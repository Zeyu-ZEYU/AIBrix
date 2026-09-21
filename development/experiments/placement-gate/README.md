# Placing models on a declared card

This walks one GPU through the whole of what `spec.perGPU` buys: a card that
is divided between the models on it, a model that is refused when the card
cannot hold it, and a card that is divided again when a model leaves.

It needs one warm runtime pool with a single GPU, and a model small enough to
start twice on it. The numbers below come from a run on one H20, which the
runtime measured at 95.33 GiB usable. Yours will differ; every figure here is
derived from the card, so work them out the same way.

Both claims declare far more than this model really costs. The declaration is
what placement judges, so overstating it fills the card on demand while the
engines stay small enough to start and answer normally.

## Before starting

The pool Deployment must not carry a `pool.aibrix.ai/policy` annotation with a
`reclaim` block. That policy writes KV limits of its own, and while it stands
down on Pods where a claim holds the limit, a pool is easier to read with only
one writer on it.

```bash
kubectl get deployment <your pool> -o jsonpath='{.metadata.annotations}'
kubectl get modelclaims          # no gate-a, gate-b or gate-c yet
```

## One card, two models

```bash
kubectl apply -f claims.yaml
kubectl get modelclaims -w
```

Each claim declares a 30 GiB footprint and a 10 GiB KV floor, so each needs
40 GiB and the two fit. The card has 95.33 GiB, the two footprints take 60,
and the two floors take 20, so 15.33 GiB is left to share. Each engine should
end up at 10 + 7.67 = 17.67 GiB:

```bash
kubectl get modelclaim gate-a -o jsonpath='{.status.instances[0].kvLimitBytes}'
kubectl get --raw "/api/v1/namespaces/<ns>/pods/<pool pod>:8080/proxy/v1/runtime/snapshot" \
  | jq '.models[] | {model_name, kv_capacity_bytes, kv_used_bytes}'
```

Three things are worth watching while they come up.

Each engine starts under its KV allocator's own limit, which is most of the
card: 76 GiB each, on the run this was written from. The controller pulls it
down to the planned share within a few seconds. Neither model is routable in
between, which is what stops an engine from growing into the memory held for
the other:

```bash
kubectl get pod <pool pod> -o jsonpath='{.metadata.annotations}' | jq .
# modelclaim.aibrix.ai/gate-a is {"model":"gate-a","port":0,"state":"activating"}
# until the limit is in force, and carries the real port afterwards.
```

The two footprints and the two limits come to the card exactly: 30 + 30 +
17.67 + 17.67 = 95.33 GiB. Nothing is left unassigned, and no byte is
promised twice.

Each claim says where it landed, and records the write:

```bash
kubectl get modelclaim gate-a -o jsonpath='{.status.conditions}' | jq .
# Scheduled True Placed: placed on pod <pool pod>
kubectl get events --field-selector reason=KVLimitSet
```

## A third model the card cannot hold

```bash
kubectl apply -f gate-c.yaml
kubectl describe modelclaim gate-c
```

gate-c needs 40 GiB. Even with both engines back at their floors the card
could offer only 15.33 GiB, so it is refused on the first of the two counts:

```text
no warm pod can hold this model, which needs 40.0 GiB on a card:
<pool pod> can offer at most 15.3 GiB, even with every engine on it at its floor
```

The answer does not change until the pool does, so the wait between attempts
doubles from 10 seconds up to a minute. Each refusal says how long:

```bash
kubectl get events --field-selector reason=NoMatchingPods
# ...; trying again in 10s
# ...; trying again in 20s
# ...; trying again in 40s
# ...; trying again in 1m0s
```

Over three minutes that is five attempts, where the ordinary ten-second pace
would be eighteen.

## Room given back, and taken

Delete gate-a and watch the card twice:

```bash
kubectl delete modelclaim gate-a
kubectl get modelclaims -w
```

Two things happen in order. Once gate-a's engine has exited, gate-b is given
the room it freed, without gate-b being touched: 95.33 less its own 30 GiB
footprint is 65.33 GiB. Then gate-c's wait comes up, the card can hold it, and
placing it divides the card again, putting gate-b back to 17.67 GiB before
gate-c's engine starts. On the run this was written from, gate-c landed twelve
seconds after gate-a's engine exited.

Finally, delete gate-b and watch gate-c grow to 65.33 GiB on its own, three
seconds after the other engine exited. Nothing was placed and no claim was
edited: the card is planned again on every round.

## Serving

A model held to a limit still serves normally, straight to the engine:

```bash
kubectl get --raw "/api/v1/namespaces/<ns>/pods/<pool pod>:<port>/proxy/v1/models"
```

and through the gateway. Note that a gateway whose default routing strategy is
`pd` never picks a ModelClaim Pod, since those carry no prefill or decode role
label. A `routing-strategy: random` header overrides it for one request
without changing the gateway:

```bash
curl -i http://<gateway>/v1/completions \
  -H 'Content-Type: application/json' \
  -H 'routing-strategy: random' \
  -d '{"model": "gate-c", "prompt": "The capital of France is", "max_tokens": 16}'
```

The `target-pod-ip` response header should name the engine's own port.

## Cleaning up

```bash
kubectl delete modelclaim gate-a gate-b gate-c --ignore-not-found
```
