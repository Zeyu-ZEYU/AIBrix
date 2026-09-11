# Placement gates, end to end

One card, three models. This exercises both placement gates and the KV limit
the controller holds each engine to, on real hardware.

What it should show:

1. Two models that fit the card by their floors are placed one at a time. The
   second waits, and says why, until the first engine is ready and held to its
   KV limit.
2. Each engine starts under kvcached's default limit, most of the card, and
   becomes routable only once the controller has written its 10 GiB limit.
3. A third model finds no card that could ever hold it. It is refused with
   numbers, and it is retried less and less often.
4. Deleting one of the first two places the third, once the old engine has
   exited.

## The numbers, and why

The pool's card reports `hbm_usable_bytes = 102363824128`, 95.33 GiB: the total
less what NVML says the driver and firmware reserve. Confirm it before running,
since everything below is sized against it:

```bash
kubectl get --raw \
  "/api/v1/namespaces/zeyu-dev/pods/<pod>:8080/proxy/v1/runtime/snapshot" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["accelerators"])'
```

Each claim declares 30 GiB of maximum footprint and a 10 GiB KV floor, so each
needs 40 GiB. The two rooms the gates use come out as follows.

| On the card | maximumRoom, all at floor | minimumRoom, engines as they are |
|---|---|---|
| nothing | 95.33 GiB | 95.33 GiB |
| gate-a, engine still starting | 55.33 GiB | unknown |
| gate-a, under kvcached's default of about 80 GiB | 55.33 GiB | about -15 GiB |
| gate-a, held to 10 GiB | 55.33 GiB | 55.33 GiB |
| gate-a and gate-b, both held to 10 GiB | 15.33 GiB | 15.33 GiB |

The first gate refuses a card whose maximumRoom is below what a model needs,
since no amount of waiting changes that. The second admits only a card whose
minimumRoom is at least what the model needs. So gate-b is held back in the
second and third rows and placed in the fourth, and gate-c is refused in the
last.

The declarations are far above what DeepSeek-R1-Distill-Qwen-1.5B really
uses, a few GB per engine. That is deliberate. The gates judge declarations,
so overstating them fills the card on demand while the engines stay small
enough to start and serve. A placement that slips through a gate will not
OOM either, so a bug shows up as a wrong decision rather than a dead engine.

## Before running

Five things, in this order.

**The previous experiment's claims are gone, and this comes first.** They
predate `status.instances[].kvLimitBytes`. The new CRD requires the field, so
the controller could no longer write their status, and the new controller
would read their limit as zero and write that into their engines.

```bash
kubectl delete modelclaim gate-a gate-b -n zeyu-dev
```

**The CRD requires `kvLimitBytes`.**

```bash
kubectl apply -f config/crd/model/model.aibrix.ai_modelclaims.yaml
```

**Both images run this branch.** The runtime reports a missing kvcached
segment as -1 rather than 0, and the controller relies on that, so
`controller-manager` and `kvcached-runtime` are built and pushed together.

```bash
# on the build host
export AIBRIX_CONTAINER_REGISTRY_NAMESPACE=<registry>/aibrix
export IMAGE_TAG=zeyu-dev
export IS_MAIN_BRANCH=false      # without this, push also overwrites :nightly
make docker-build-controller-manager docker-build-kvcached-runtime
make docker-push-controller-manager docker-push-kvcached-runtime
```

**Both deployments restart.** They use `imagePullPolicy: Always`, which is
what makes reusing one tag work.

```bash
kubectl -n aibrix-system rollout restart deploy/aibrix-controller-manager
kubectl -n zeyu-dev rollout restart deploy/warm-runtime-pool-zeyu
```

**The pool's reclaim policy stays off.** It writes KV limits of its own, and
the controller writes each engine's recorded limit back on every pass, so the
two would undo each other.

```bash
kubectl -n zeyu-dev get deploy warm-runtime-pool-zeyu -o json \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["metadata"].get("annotations",{}).get("pool.aibrix.ai/policy"))'
```

Expected: `None`.

## Running it

`verify.sh` only reads. It recomputes both rooms from the same sources the
controller uses, and shows each engine's segment beside the limit its
instance records, so its arithmetic can be compared with what the controller
decided.

```bash
export KUBECTL='ssh zboe ~/.pixi/bin/kubectl'
./verify.sh state
```

### 1: two models, one at a time

```bash
kubectl apply -f development/experiments/placement-gate/claims.yaml
./verify.sh claims; ./verify.sh conditions
```

Expected, in order:

- One claim, say gate-a, is Activating. The other is Pending with
  `WaitingForRoom`, and its message ends
  `on <pod>, the engine of gate-a cannot be read yet: it is starting,
  restarting, or has no kvcached segment`.
- Once gate-a's engine is ready, a `KVLimitSet` event reads
  `model gate-a on pod <pod>: KV limit set to 10.0 GiB, from <default>`. For
  a pass or two before that lands, gate-b's message may give a negative
  figure instead: `the roomiest pod that could hold it, <pod>, can vouch for
  only -14.7 GiB`.
- gate-b is placed, its engine goes through the same `KVLimitSet`, and both
  reach Active with `Scheduled` True and the reason `Placed`.

```bash
./verify.sh ledger
```

Expected: both instances show `limit 10.00GiB  segment 10.00GiB`, marked
`in force`, and the card shows `maximumRoom 15.33GiB  minimumRoom 15.33GiB`.

The failure worth watching for is gate-b placed while gate-a's engine is still
starting, or while its segment still holds the default. That would mean the
second gate is not consulted.

### 2: a model no card could hold

```bash
kubectl apply -f development/experiments/placement-gate/gate-c.yaml
./verify.sh conditions
```

Expected: gate-c is Pending with

```
InsufficientCapacity: no warm pod has room: this model needs 40.0 GiB per GPU,
and of 1 candidate pod(s) the roomiest could free at most 15.3 GiB
```

```bash
./verify.sh events
```

Expected: the `InsufficientCapacity` event's count rises more and more slowly.
The controller retries a refused claim after 10 seconds, then 20, 40, 80 and
160, and every 5 minutes after that.

### 3: room appears

```bash
date --iso-8601=seconds
kubectl delete modelclaim gate-a -n zeyu-dev
./verify.sh claims; ./verify.sh conditions
```

Expected: deleting gate-a removes its routing annotation, and that pod update
reconciles every claim at once. gate-c then passes the first gate but is held
by the second, with `the runtime reports engines no claim accounts for:
gate-a`, until gate-a's engine has exited. It is placed on the next pass after
that, about 10 seconds later, and goes through `KVLimitSet` to Active.

### 4: serving

The pool's gateway serves another team's PD deployment and cannot route to
these pods, so a request goes to an engine through the API server's pod
proxy instead. It is a POST to our own pod, so ask before running it.

```bash
PORT=$(kubectl get modelclaim gate-b -n zeyu-dev -o json \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"]["instances"][0]["port"])')
echo '{"model":"gate-b","prompt":"hello","max_tokens":16}' > /tmp/gate-b-request.json
kubectl create --raw \
  "/api/v1/namespaces/zeyu-dev/pods/<pod>:$PORT/proxy/v1/completions" \
  -f /tmp/gate-b-request.json
```

## Cleaning up

```bash
kubectl delete -f development/experiments/placement-gate/gate-c.yaml
kubectl delete -f development/experiments/placement-gate/claims.yaml
```

## A note on pods without GPUs

Whether the gates apply to a pod is decided by the pod's `nvidia.com/gpu`
resources, not by what its runtime reports. A pool pod that requests a card
and whose runtime then reports none is a fault, not a CPU-only pod: NVML
missing, or a device never mounted, looks identical to a CPU-only pod from the
snapshot, and that card may already be full. `verify.sh` prints each pod's GPU
count so the two cases can be told apart at a glance.

## What this does not cover

No engine's limit is ever raised above its floor, so a card never has KV to
give back. Shrinking a neighbour's limit to make room, putting an idle
neighbour to sleep, and a budget that grows busy engines are all still to be
built. Nor is the declared footprint checked against what an engine really
holds.
