#!/usr/bin/env bash
# Read-only checks for the placement gate experiment. Nothing here changes the
# cluster; the steps that do are listed in README.md and are run by hand.
#
#   KUBECTL='ssh zboe ~/.pixi/bin/kubectl' ./verify.sh state
#
# Every kubectl call asks for -o json and parses it here, never with a
# jsonpath: KUBECTL may be an ssh invocation, and a jsonpath's spaces and
# braces would be re-split by the remote shell.
set -uo pipefail

KUBECTL=${KUBECTL:-kubectl}
NAMESPACE=${NAMESPACE:-zeyu-dev}
POOL_LABEL=${POOL_LABEL:-pool.aibrix.ai/name=zeyu-pool-a}
CLAIMS=${CLAIMS:-"gate-a gate-b gate-c"}

# The fourth column is the pod's GPU count, read from its resources exactly as
# the controller reads it. A pod with none falls outside the memory gates; a
# pod with one whose runtime reports no card is a fault, not a CPU-only pod.
pods() {
  $KUBECTL get pods -n "$NAMESPACE" -l "$POOL_LABEL" -o json | python3 -c '
import json, sys
for pod in json.load(sys.stdin)["items"]:
    status = pod.get("status") or {}
    gpus = 0
    for container in (pod.get("spec") or {}).get("containers", []):
        resources = container.get("resources") or {}
        for section in ("limits", "requests"):
            value = (resources.get(section) or {}).get("nvidia.com/gpu")
            if value is not None:
                gpus += int(value)
                break
    print(pod["metadata"]["name"], status.get("podIP", "-"), status.get("phase", "-"), gpus)
'
}

# claim_lines prints each claim's phase, its instances as pod:phase:limit, and
# the reason on its Scheduled condition.
claim_lines() {
  $KUBECTL get modelclaims -n "$NAMESPACE" -o json | python3 -c '
import json, sys
for claim in json.load(sys.stdin)["items"]:
    name = claim["metadata"]["name"]
    status = claim.get("status") or {}
    parts = []
    for instance in status.get("instances", []):
        limit = instance.get("kvLimitBytes")
        shown = "%.1fGiB" % (limit / 2 ** 30) if isinstance(limit, int) else "?"
        parts.append("%s:%s:%s" % (instance.get("pod", "?"), instance.get("phase", "?"), shown))
    where = ",".join(parts) or "-"
    reason = "-"
    for condition in status.get("conditions", []):
        if condition.get("type") == "Scheduled":
            reason = condition.get("reason", "-")
    print("%-10s %-11s %-64s %s" % (name, status.get("phase", "-"), where, reason))
'
}

# ledger recomputes what the controller should be seeing, from the same
# sources it uses: the runtime snapshot for the card's size and each engine's
# KV figures, and claim status plus spec.perGPU for what the card owes. A
# disagreement between this and the controller's own condition message is a
# real finding, not a script bug.
ledger() {
  local claims_json pod _ip _phase gpus snapshot
  claims_json=$($KUBECTL get modelclaims -n "$NAMESPACE" -o json)
  while read -r pod _ip _phase gpus; do
    [ -z "$pod" ] && continue
    if [ "${gpus:-0}" = "0" ]; then
      printf '%s: no GPU allocated, the gates do not apply\n' "$pod"
      continue
    fi
    snapshot=$($KUBECTL get --raw "/api/v1/namespaces/$NAMESPACE/pods/$pod:8080/proxy/v1/runtime/snapshot" 2>/dev/null)
    if [ -z "$snapshot" ]; then
      printf '%s: snapshot unavailable, nothing is known about this pod\n' "$pod"
      continue
    fi
    printf '%s' "$snapshot" | CLAIMS_JSON="$claims_json" POD="$pod" python3 -c '
import json, os, sys

GIB = 2 ** 30
pod = os.environ["POD"]
snapshot = json.load(sys.stdin)
claims = json.loads(os.environ["CLAIMS_JSON"])["items"]
cards = snapshot.get("accelerators") or []
usable = min((card.get("hbm_usable_bytes", 0) for card in cards), default=0)
if usable <= 0:
    print("%s: the runtime could not size this card (hbm_usable_bytes %d)" % (pod, usable))
    sys.exit(0)
engines = snapshot.get("models") or []


def gib(value):
    return "%.2fGiB" % (value / GIB)


# The engine a claim owns: by claim UID when the runtime reports one, else by
# served name, the way the controller matches them.
def engine_for(claim):
    uid = claim["metadata"].get("uid", "")
    served = (claim.get("spec") or {}).get("modelName") or claim["metadata"]["name"]
    legacy = None
    for index, engine in enumerate(engines):
        ref = engine.get("claim_ref") or {}
        if uid and ref.get("uid"):
            if ref["uid"] == uid:
                return index
            continue
        if engine.get("model_name") == served:
            legacy = index
    return legacy


# An engine counts only while its process is alive and it is ready or asleep,
# and only when its segment exists.
def readable(engine):
    if engine is None or not engine.get("alive"):
        return False
    if not (engine.get("ready") or engine.get("phase") == "sleeping"):
        return False
    return engine.get("kv_capacity_bytes", -1) >= 0 and engine.get("kv_used_bytes", -1) >= 0


floors = 0
bounds = 0
unreadable = None
claimed = set()
lines = []
for claim in claims:
    name = claim["metadata"]["name"]
    per = (claim.get("spec") or {}).get("perGPU") or {}
    footprint = per.get("maximumFootprintBytes", 0)
    floor = per.get("kvFloorBytes", 0)
    for instance in (claim.get("status") or {}).get("instances", []):
        if instance.get("pod") != pod or instance.get("phase") == "Failed":
            continue
        limit = instance.get("kvLimitBytes", 0)
        index = engine_for(claim)
        engine = engines[index] if index is not None else None
        if index is not None:
            claimed.add(index)
        floors += footprint + floor
        if readable(engine):
            capacity = engine["kv_capacity_bytes"]
            used = engine["kv_used_bytes"]
            bound = max(limit, capacity, used)
            bounds += footprint + bound
            state = "in force" if capacity == limit else "NOT in force"
            lines.append("  %-8s limit %s  segment %s  mapped %s  upper bound %s  (%s)" % (
                name, gib(limit), gib(capacity), gib(used), gib(bound), state))
        else:
            unreadable = unreadable or name
            phase = engine.get("phase", "?") if engine else "not in the snapshot"
            lines.append("  %-8s limit %s  engine %s, not readable" % (name, gib(limit), phase))

strangers = [engine.get("model_name", "?") for index, engine in enumerate(engines)
             if engine.get("alive") and index not in claimed]
summary = "%s: usable %s  maximumRoom %s" % (pod, gib(usable), gib(usable - floors))
if strangers:
    summary += "  minimumRoom unknown: the runtime reports engines no claim accounts for: " + ", ".join(strangers)
elif unreadable:
    summary += "  minimumRoom unknown: the engine of %s cannot be read yet" % unreadable
else:
    summary += "  minimumRoom " + gib(usable - bounds)
print(summary)
for line in lines:
    print(line)
'
  done < <(pods)
}

conditions() {
  $KUBECTL get modelclaims -n "$NAMESPACE" -o json | CLAIMS="$CLAIMS" python3 -c '
import json, os, sys
wanted = set(os.environ["CLAIMS"].split())
for claim in json.load(sys.stdin)["items"]:
    name = claim["metadata"]["name"]
    if wanted and name not in wanted:
        continue
    print("--- " + name)
    for condition in (claim.get("status") or {}).get("conditions", []):
        if condition.get("type") == "Scheduled":
            print("    %s %s: %s" % (condition.get("status", "-"), condition.get("reason", "-"), condition.get("message", "")))
'
}

events() {
  $KUBECTL get events -n "$NAMESPACE" -o json | python3 -c '
import json, sys
interesting = ("Capacity", "WaitingForRoom", "NoMatching", "Activating", "Activated",
               "KVLimit", "Unhealthy", "EngineFailed")
rows = []
for event in json.load(sys.stdin)["items"]:
    reason = event.get("reason", "")
    if not any(word in reason for word in interesting):
        continue
    stamp = event.get("lastTimestamp") or event.get("eventTime") or ""
    obj = event.get("involvedObject") or {}
    rows.append((stamp, event.get("count", 1), reason, obj.get("name", ""), event.get("message", "")))
for stamp, count, reason, name, message in sorted(rows)[-20:]:
    print("%s  x%-3s %-22s %-8s %s" % (stamp, count, reason, name, message))
'
}

case "${1:-state}" in
  state)
    echo "=== pool pods ==="; pods
    echo; echo "=== claims ==="; claim_lines
    echo; echo "=== ledger, recomputed independently ==="; ledger
    echo; echo "=== Scheduled conditions ==="; conditions
    ;;
  ledger)     ledger ;;
  claims)     claim_lines ;;
  conditions) conditions ;;
  events)     events ;;
  *)
    echo "usage: $0 [state|ledger|claims|conditions|events]" >&2
    exit 2
    ;;
esac
