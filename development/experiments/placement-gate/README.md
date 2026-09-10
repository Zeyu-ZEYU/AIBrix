# Placement gate, end to end

This exercises the check that refuses a pod which could never hold a model, on
real hardware: two GPUs, three models, and one model too many.

What it is meant to show:

1. Two models that each need more than half a card land on separate cards.
2. A third model finds no card that could ever hold it, stays Pending, and says
   so with numbers rather than "no available warm pod".
3. Its retries back off, 10s doubling towards 5m, instead of polling forever.
4. Deleting one of the first two places the third **at once**, not at the end of
   whatever wait it had reached.
5. Every model that is placed serves requests through the gateway throughout.

## The numbers, and why

The pool's card reports `hbm_total_bytes = 102625181696`, which is 95.58 GiB.
The controller subtracts a 256 MiB driver reserve, so

```
HBM_usable = 102625181696 - 268435456 = 102356746240   (95.33 GiB)
```

Each claim declares 50 GiB of maximum footprint and a 10 GiB KV floor, so its
`minimumReserveBytes` is 60 GiB, `64424509440`. One card holding one of them has

```
maximumRoomBytes = 102356746240 - 64424509440 = 37932236800   (35.33 GiB)
```

which is less than the 60 GiB a second model would need. Two cards therefore
hold exactly two models, and the third has nowhere to go.

Those declarations are far above what this model really uses:
DeepSeek-R1-Distill-Qwen-1.5B holds a few GB. That is deliberate. The gate
judges the declaration, so overstating it produces a full pool on demand, while
the engines stay small enough to start and serve normally. It also means a
placement that slips through the gate will not OOM, so a bug shows up as a
wrong decision rather than a dead engine.

## Before running

Four things have to be true first. All four change the cluster.

**The CRD carries `spec.perGPU`.** Without it the API server drops the field
silently and every claim is refused with `NumbersMissing`.

```bash
kubectl apply -f config/crd/model/model.aibrix.ai_modelclaims.yaml
kubectl get crd modelclaims.model.aibrix.ai \
  -o jsonpath='{.spec.versions[0].schema.openAPIV3Schema.properties.spec.properties.perGPU}'
```

**The controller runs this branch.** Build and push
`controller-manager`, then restart the deployment.

```bash
kubectl -n aibrix-system rollout restart deploy/aibrix-controller-manager
kubectl -n aibrix-system rollout status deploy/aibrix-controller-manager
```

**The pool has two pods.** The node has 8 GPUs and each pod requests one, so
scaling is enough; no new node is needed.

```bash
kubectl scale deploy/warm-runtime-pool-zeyu -n zeyu-dev --replicas=2
kubectl get pods -n zeyu-dev -l pool.aibrix.ai/name=zeyu-pool-a -w
```

**The old claims are gone.** `zeyu-qwen15b-a` and `zeyu-qwen15b-b` are Failed
and declare no `spec.perGPU`. Their instances are Failed too, so they no longer
blind their card, but they still confuse the reading.

```bash
kubectl delete modelclaim zeyu-qwen15b-a zeyu-qwen15b-b -n zeyu-dev
```

The pool's `pool.aibrix.ai/policy` annotation asks for a 1 GiB KV capacity with
a 60 percent floor per model, which cannot be satisfied by two models and makes
every budget round abandon itself. It does not affect this experiment, and the
log noise is a useful reminder that it is still there.

## Running it

`verify.sh` only reads. It recomputes the ledger from the same two sources the
controller uses, so its arithmetic can be compared against what the controller
actually decided.

```bash
export KUBECTL='ssh zboe ~/.pixi/bin/kubectl'
./verify.sh state
```

### 1 and 2: fill both cards, then refuse

```bash
kubectl apply -f development/experiments/placement-gate/claims.yaml
./verify.sh state
```

Expected: `gate-a` and `gate-b` Active on different pods, each card showing
`maximumRoom 35.33GiB`; `gate-c` Pending with

```
InsufficientCapacity: 2 candidate pod(s) refused; model needs 64424509440 bytes
per GPU, the roomiest pod could free at most 37932236800; retrying in 40s
```

The failure worth watching for is `gate-a` and `gate-b` landing on the *same*
pod. That would mean the gate is not being consulted at all.

### 3: the backoff

```bash
kubectl logs -n aibrix-system deploy/aibrix-controller-manager --since=10m \
  | grep -i gate-c
./verify.sh events
```

Expected: reconciles for `gate-c` at widening intervals, 10s, 20s, 40s, 80s,
160s, then 5m; and one `PlacementOutOfMemory` event whose count rises rather
than a new event each round.

### 4: room appears

```bash
date --iso-8601=seconds
kubectl delete modelclaim gate-a -n zeyu-dev
./verify.sh claims
```

Expected: `gate-c` Activating on the card `gate-a` released, within seconds,
even if it had reached the 5-minute wait. Deleting a claim removes its routing
annotation from the pod, the pod update re-enqueues every claim in the
namespace, and the changed ledger resets the backoff.

The failure worth watching for is `gate-c` sitting out its remaining wait. That
would mean the fingerprint is not tracking the ledger.

### 5: serving

Through the gateway, throughout the run:

```bash
GATEWAY=http://101.126.78.197
curl -s $GATEWAY/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gate-a","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```

Worth doing at three moments: while a and b are up and c is waiting; right
after c is placed; and against a model that was never placed, which should be
refused by the gateway rather than routed anywhere.

## Cleaning up

```bash
kubectl delete -f development/experiments/placement-gate/claims.yaml
kubectl scale deploy/warm-runtime-pool-zeyu -n zeyu-dev --replicas=1
```

## A note on pods without GPUs

Whether the gate applies to a pod is decided by the pod's ``nvidia.com/gpu``
resources, not by what its runtime reports. A pool pod that requests a card and
whose sidecar then reports no accelerator is treated as a fault and refused,
because NVML missing or a device not mounted looks identical to a CPU-only pod
from the snapshot, and that card may already be full. `verify.sh` prints each
pod's GPU count so the two cases can be told apart at a glance.

## What this does not cover

Only the first gate exists. `minimumRoomBytes` and `currentRoomBytes`, the
reservation, the shrink and the wait for it, and sleeping a neighbour to make
room are all still to be built. A model that could fit if a neighbour gave up
some KV cache is refused here exactly like one that could never fit.
