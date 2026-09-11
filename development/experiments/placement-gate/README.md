# Placement gate, end to end

One card, two models, one of which cannot fit. This exercises the check that
refuses a pod provably too small, on real hardware.

What it should show:

1. The first model is placed and serves requests.
2. The second finds no card that could ever hold it, stays Pending, and says so
   with numbers rather than "no available warm pod".
3. Removing the first places the second, with nothing else changed.

## The numbers, and why

The pool's card reports `hbm_total_bytes = 102625181696`, 95.58 GiB. The runtime
measures what an engine can actually take while nothing holds the card and
reports it as `hbm_usable_bytes`; on this pool that came to `102363824128`,
95.33 GiB, the driver keeping the other 249 MiB.

Each claim declares 50 GiB of maximum footprint and a 10 GiB KV floor, so its
`minimumReserveBytes` is 60 GiB. One card holding one of them has

```
maximumRoomBytes = 102363824128 - 64424509440 = 37939314688   (35.33 GiB)
```

which is less than the 60 GiB a second model needs. So the card holds exactly
one, and the second has nowhere to go.

Those declarations are far above what this model really uses:
DeepSeek-R1-Distill-Qwen-1.5B holds a few GB. That is deliberate. The gate
judges the declaration, so overstating it produces a full card on demand while
the engine stays small enough to start and serve. It also means a placement that
slips through the gate will not OOM, so a bug shows up as a wrong decision
rather than a dead engine.

Confirm the figure before running, since everything above is sized against it:

```bash
kubectl get --raw \
  "/api/v1/namespaces/zeyu-dev/pods/<pod>:8080/proxy/v1/runtime/snapshot" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["accelerators"])'
```

A card reporting `hbm_usable_bytes: -1` was already in use when its runtime
started, so it was never measured and no model can be placed on it. That needs
the pod restarted on an idle card, not a workaround.

## Before running

Four things, in this order.

**The old claims are gone, and this comes first.** `zeyu-qwen15b-a` and
`zeyu-qwen15b-b` declare no `spec.perGPU`. Once the new CRD makes those fields
required, any write to those objects is rejected, including the status updates
the controller needs to make, so they have to go before the schema changes.

```bash
kubectl delete modelclaim zeyu-qwen15b-a zeyu-qwen15b-b -n zeyu-dev
```

**The CRD requires `spec.perGPU`.** Without it the API server drops the field
silently and every card looks empty. With it, a claim that omits the two numbers
is refused at apply time rather than reaching a controller.

```bash
kubectl apply -f config/crd/model/model.aibrix.ai_modelclaims.yaml
kubectl get crd modelclaims.model.aibrix.ai -o json \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["spec"]["versions"][0]["schema"]["openAPIV3Schema"]["properties"]["spec"]["required"])'
```

**Both images run this branch.** The runtime reports `hbm_usable_bytes` and the
controller requires it, so `controller-manager` and `kvcached-runtime` have to
be built and pushed together. A new controller against an old runtime sees no
card size anywhere and places nothing. Both deployments already use
`imagePullPolicy: Always`, which is what makes reusing one tag work.

```bash
# on the build host
export AIBRIX_CONTAINER_REGISTRY_NAMESPACE=<registry>/aibrix
export IMAGE_TAG=zeyu-dev
export IS_MAIN_BRANCH=false      # without this, push also overwrites :nightly
make docker-build-controller-manager docker-build-kvcached-runtime
make docker-push-controller-manager docker-push-kvcached-runtime

kubectl -n aibrix-system rollout restart deploy/aibrix-controller-manager
kubectl -n zeyu-dev rollout restart deploy/warm-runtime-pool-zeyu
```

**The pool's reclaim policy is off.** `pool.aibrix.ai/policy` carries a 1 GiB KV
capacity with a 60 percent floor per model, which two models cannot satisfy, so
every budget round abandons itself as unsafe and fills the log. Worse, nothing
makes that percentage-derived floor agree with the `kvFloorBytes` a claim
declares. The gate does not need the budget loop, so remove the annotation
rather than picking a new number for it.

```bash
kubectl -n zeyu-dev annotate deploy/warm-runtime-pool-zeyu pool.aibrix.ai/policy-
```

## Running it

`verify.sh` only reads. It recomputes the ledger from the same sources the
controller uses, so its arithmetic can be compared against what the controller
actually decided.

```bash
export KUBECTL='ssh zboe ~/.pixi/bin/kubectl'
./verify.sh state
```

### 1: the first model is placed

```bash
kubectl apply -f development/experiments/placement-gate/claims.yaml
./verify.sh state
```

Expected: `gate-a` reaches Active, the card shows `maximumRoom 35.33GiB`, and
`gate-b` is Pending with

```
InsufficientCapacity: no warm pod has room: this model needs 60.0 GiB per GPU,
and of 1 candidate pod(s) the roomiest could free at most 35.3 GiB
```

The failure worth watching for is `gate-b` also reaching Active. That would mean
the gate is not being consulted at all.

### 2: the refusal is visible to an operator

```bash
./verify.sh conditions
./verify.sh events
```

Expected: one `InsufficientCapacity` event whose count rises on each retry,
rather than a new event every round.

### 3: room appears

```bash
date --iso-8601=seconds
kubectl delete modelclaim gate-a -n zeyu-dev
./verify.sh claims
```

Expected: `gate-b` reaches Activating on the card `gate-a` released. Deleting a
claim removes its routing annotation from the pod, and that pod update
re-enqueues every claim in the namespace, so this should take seconds rather
than a full requeue interval.

### 4: serving

Through the gateway, while `gate-a` is up and `gate-b` is waiting:

```bash
GATEWAY=http://<gateway-ip>
curl -s $GATEWAY/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gate-a","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```

Worth repeating against `gate-b` while it is Pending, which should be refused by
the gateway rather than routed anywhere, and again after step 3 places it.

## Cleaning up

```bash
kubectl delete -f development/experiments/placement-gate/claims.yaml
```

## A note on pods without GPUs

Whether the gate applies to a pod is decided by the pod's `nvidia.com/gpu`
resources, not by what its runtime reports. A pool pod that requests a card and
whose runtime then reports none is treated as a fault and refused, because NVML
missing or a device not mounted looks identical to a CPU-only pod from the
snapshot, and that card may already be full. `verify.sh` prints each pod's GPU
count so the two cases can be told apart at a glance.

## What this does not cover

Only the first gate exists. `minimumRoomBytes` and `currentRoomBytes`, the
reservation, the shrink and the wait for it, and sleeping a neighbour to make
room are all still to be built. A model that would fit if a neighbour gave up
some KV cache is refused here exactly like one that could never fit.
