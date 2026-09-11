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

# The driver reserve the controller subtracts from a card's total.
DRIVER_RESERVE_BYTES=268435456

# The fourth column is the pod's GPU count, read from its resources exactly as
# the controller reads it. A pod with none falls outside the memory gate; a pod
# with one whose runtime reports no card is a fault, not a CPU-only pod.
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

claim_lines() {
  $KUBECTL get modelclaims -n "$NAMESPACE" -o json | python3 -c '
import json, sys
for claim in json.load(sys.stdin)["items"]:
    name = claim["metadata"]["name"]
    status = claim.get("status") or {}
    parts = []
    for instance in status.get("instances", []):
        parts.append(instance.get("pod", "?") + ":" + instance.get("phase", "?"))
    where = ",".join(parts) or "-"
    reason = "-"
    for condition in status.get("conditions", []):
        if condition.get("type") == "Scheduled":
            reason = condition.get("reason", "-")
    phase = status.get("phase", "-")
    print(f"{name:<10} {phase:<11} {where:<48} {reason}")
'
}

# ledger recomputes what the controller should be seeing, from the same two
# sources it uses: the runtime snapshot for card size, and claim status plus
# spec.perGPU for what the card already owes. A disagreement between this and
# the controller's own condition message is a real finding, not a script bug.
ledger() {
  local claims_json pod ip gpus total snapshot
  claims_json=$($KUBECTL get modelclaims -n "$NAMESPACE" -o json)
  while read -r pod ip _phase gpus; do
    [ -z "$pod" ] && continue
    if [ "${gpus:-0}" = "0" ]; then
      printf '%-42s no GPU allocated, the gate does not apply\n' "$pod"
      continue
    fi
    snapshot=$($KUBECTL get --raw "/api/v1/namespaces/$NAMESPACE/pods/$pod:8080/proxy/v1/runtime/snapshot" 2>/dev/null)
    if [ -z "$snapshot" ]; then
      printf '%-42s snapshot unavailable\n' "$pod"
      continue
    fi
    total=$(printf '%s' "$snapshot" | python3 -c '
import json, sys
cards = json.load(sys.stdin).get("accelerators") or []
print(min((card["hbm_total_bytes"] for card in cards), default=0))
')
    printf '%s' "$claims_json" | POD="$pod" TOTAL="$total" RESERVE="$DRIVER_RESERVE_BYTES" python3 -c '
import json, os, sys
pod = os.environ["POD"]
total = int(os.environ["TOTAL"])
reserve = int(os.environ["RESERVE"])
usable = total - reserve if total > reserve else 0
charged = 0
lines = []
blind = None
for claim in json.load(sys.stdin)["items"]:
    name = claim["metadata"]["name"]
    per = (claim.get("spec") or {}).get("perGPU") or {}
    footprint = per.get("maximumFootprintBytes")
    floor = per.get("kvFloorBytes")
    for instance in (claim.get("status") or {}).get("instances", []):
        if instance.get("pod") != pod or instance.get("phase") == "Failed":
            continue
        if not footprint or not floor:
            blind = blind or name
            continue
        charged += footprint + floor
        size = (footprint + floor) / 2**30
        lines.append(f"{name}={size:.0f}GiB")
if total == 0:
    print(f"{pod:<42} has a GPU but the runtime reported no usable card: refused")
elif blind:
    print(f"{pod:<42} unreadable: {blind} declares no perGPU")
else:
    detail = " [" + ", ".join(lines) + "]" if lines else ""
    u = usable / 2**30
    c = charged / 2**30
    room = (usable - charged) / 2**30
    print(f"{pod:<42} usable {u:.2f}GiB  charged {c:.2f}GiB  maximumRoom {room:.2f}GiB{detail}")
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
            print("    " + condition.get("reason", "-") + ": " + condition.get("message", ""))
'
}

events() {
  $KUBECTL get events -n "$NAMESPACE" -o json | python3 -c '
import json, sys
interesting = ("Placement", "Ledger", "Numbers", "Capacity", "Activating", "NoMatching")
rows = []
for event in json.load(sys.stdin)["items"]:
    reason = event.get("reason", "")
    if not any(word in reason for word in interesting):
        continue
    stamp = event.get("lastTimestamp") or event.get("eventTime") or ""
    obj = event.get("involvedObject") or {}
    rows.append((stamp, event.get("count", 1), reason, obj.get("name", ""), event.get("message", "")))
for stamp, count, reason, name, message in sorted(rows)[-15:]:
    print(f"{stamp}  x{count:<3} {reason:<22} {name:<10} {message}")
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
