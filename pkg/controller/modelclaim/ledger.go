/*
Copyright 2026 The Aibrix Team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package modelclaim

import (
	"context"

	modelv1alpha1 "github.com/vllm-project/aibrix/api/model/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// The ledger is the control plane's own account of GPU memory. It is built
// from what users declared on their ModelClaims and from how large the cards
// are, never from what the cards currently hold, so it does not move when
// traffic moves.
//
// A card offers three different amounts of room depending on how much
// disruption is acceptable, and they always order this way:
//
//	minimumRoomBytes  <=  currentRoomBytes  <=  maximumRoomBytes
//
// The smallest assumes nobody is touched. The middle one assumes every engine
// gives back the KV cache it is not using, which needs a live reading and so
// is never stored. The largest assumes every engine drops to the KV floor it
// declared, which can only be reached by putting engines to sleep.
//
// Only maximumRoomBytes exists here, because it is the one that can prove a
// placement impossible rather than merely inconvenient: needing more than it
// cannot be fixed by waiting. The other two arrive with the machinery that can
// act on them.

// hbmUsableUnknown is what HBMUsableBytes holds whenever the ledger cannot say
// how large the card is. It is negative rather than zero because zero is a
// number arithmetic accepts: a reader who skipped the state check would
// compute a plausible-looking card from zero, and an obviously broken one from
// this. Only ledgerComplete carries a real size.
const hbmUsableUnknown int64 = -1

// ledgerState says whether a ledger can answer how much room its card has,
// and when it cannot, why. The five values are mutually exclusive, so a ledger
// is in exactly one of them and no combination has to be reasoned about.
//
// Only ledgerComplete permits an answer. The rest are not the same as each
// other: a pod with no GPU is one this account does not apply to, while the
// others are pods it applies to but could not read.
type ledgerState int

const (
	// ledgerUnknown is the zero value, and deliberately so. Nothing was
	// learned about this pod: its runtime did not answer, the claim list
	// failed, or the pod is not in the account at all. Looking up a pod that
	// was never tracked yields this, and the honest thing for that lookup to
	// say is that it knows nothing. A zero meaning "complete" would let an
	// untracked pod pass as an empty card.
	//
	// This usually clears on its own. A pod that was still starting, or a
	// request that timed out, has an answer on the next reconcile.
	ledgerUnknown ledgerState = iota
	// ledgerComplete means the card's size is known and room can be computed.
	ledgerComplete
	// ledgerNoGPU means Kubernetes allocated this pod no GPU, so there is no
	// GPU memory to keep an account of. This is not a gap in the account; it
	// is a pod the account does not cover.
	ledgerNoGPU
	// ledgerGPUMeasureFailed means the runtime answered, the pod does hold
	// GPUs, and the runtime could not size at least one of them: NVML did not
	// report the driver's reservation, or reported no accelerator at all, or
	// fewer than this model's parallelism spans.
	//
	// Unlike ledgerUnknown this rarely clears on its own. Its usual causes are
	// a driver or NVML binding too old to report the reservation, and a device
	// that never reached the container, and each needs a person to look.
	ledgerGPUMeasureFailed
)

// String is the phrase an operator reads, so each value says what happened
// rather than naming itself.
func (s ledgerState) String() string {
	switch s {
	case ledgerComplete:
		return "complete"
	case ledgerNoGPU:
		return "the pod holds no GPU"
	case ledgerGPUMeasureFailed:
		return "the runtime could not measure this pod's GPUs"
	default:
		return "nothing is known about this pod"
	}
}

// ledgerInstance is one engine instance already placed on a card, carrying the
// two numbers its ModelClaim declared. It is an instance and not a claim: one
// claim's instances can sit on different cards, and only the ones on this card
// consume its memory.
type ledgerInstance struct {
	Claim                 types.NamespacedName
	MaximumFootprintBytes int64
	KVFloorBytes          int64
}

// podLedger is the control plane's account of the GPU memory behind one warm
// pod: how much of the card can hold an engine, and which instances have
// already been promised part of it. Every instance occupies all of the pod's
// cards, so the pod is described by its tightest one and each instance is
// charged its declared per-GPU cost against that card.
//
// Under tensor parallelism that is exact, since the ranks are identical. Under
// pipeline parallelism it is conservative: the stages are not equal, a claim
// declares its heaviest, and the lighter cards are charged more than they
// hold. Telling those apart would mean keeping the account per accelerator
// rather than per pod, which buys nothing in a homogeneous pool.
type podLedger struct {
	State ledgerState
	// HBMUsableBytes is the card's size, and is hbmUsableUnknown unless State
	// is ledgerComplete.
	HBMUsableBytes int64
	Instances      []ledgerInstance
}

// ledgerFor returns a pod's account, and an unknown one for a pod that has
// none. Every read goes through here rather than indexing the map directly: a
// bare lookup yields the zero value, whose HBMUsableBytes is zero rather than
// hbmUsableUnknown, and zero is a number arithmetic accepts. Only ledgerFor
// keeps the invariant that a size is real exactly when the state is complete.
func ledgerFor(ledgers map[string]podLedger, pod string) podLedger {
	if ledger, found := ledgers[pod]; found {
		return ledger
	}
	return podLedger{State: ledgerUnknown, HBMUsableBytes: hbmUsableUnknown}
}

// chargeable reports whether an instance can still be added to this pod's
// account. Once a card cannot be sized, or its account already has a hole,
// further entries change nothing: the total was already unknowable.
func (l podLedger) chargeable() bool {
	return l.State == ledgerComplete
}

// MaximumRoomBytes is the most memory this card could ever offer a new
// instance: what is left once every instance already on it is down to the KV
// floor its claim declared. Reaching it means putting engines to sleep, so a
// model needing more than this cannot be placed here by waiting. The value can
// be negative if the card is already promised more than it has.
//
// The second return is false when the ledger cannot answer, in which case the
// first has no meaning. Zero with a true second return is a real answer: a
// card whose instances have been promised exactly all of it.
func (l podLedger) MaximumRoomBytes() (int64, bool) {
	if l.State != ledgerComplete {
		return 0, false
	}
	room := l.HBMUsableBytes
	for _, instance := range l.Instances {
		room -= instance.MaximumFootprintBytes + instance.KVFloorBytes
	}
	return room, true
}

// claimMinimumReserveBytes is what one instance of this claim takes off a card
// and does not give back while it is awake: its maximum footprint plus its KV
// floor. An engine's KV limit can be lowered towards the floor but never past
// it, so this is a lower bound on occupancy rather than an estimate of it.
//
// It describes one instance on one card. A claim with several instances spends
// this much on each card it lands on.
//
// Both numbers are required by the CRD and validated as positive, so this does
// not check for them. The API server is the only place that check belongs.
func claimMinimumReserveBytes(pm *modelv1alpha1.ModelClaim) int64 {
	return pm.Spec.PerGPU.MaximumFootprintBytes + pm.Spec.PerGPU.KVFloorBytes
}

// hbmUsableBytes is how much of a pod's GPU memory can ever hold an engine,
// taken from what the runtime measured rather than derived here. A pod with
// several cards is described by its tightest one: which card an engine lands
// on is decided by the device plugin, not by us, so the smallest is the only
// safe reading. In a homogeneous pool every card is the same size and the
// choice costs nothing.
//
// One card the runtime could not size makes the whole pod unsizable. Taking
// the cards it could read and ignoring the rest would describe a pod that does
// not exist.
func hbmUsableBytes(snapshot *RuntimeSnapshot) (int64, bool) {
	if snapshot == nil || len(snapshot.Accelerators) == 0 {
		return hbmUsableUnknown, false
	}
	usable := int64(0)
	known := false
	for _, accelerator := range snapshot.Accelerators {
		if accelerator.HBMUsableBytes <= 0 {
			return hbmUsableUnknown, false
		}
		if !known || accelerator.HBMUsableBytes < usable {
			usable, known = accelerator.HBMUsableBytes, true
		}
	}
	return usable, known
}

// collectPodLedgers builds one ledger per candidate pod. Card size comes from
// the placement states already gathered from runtime snapshots; the instances
// come from ModelClaim status, which the controller alone writes, so the
// account survives a controller restart and does not depend on every engine
// being reachable at the moment a decision is made.
func (r *ModelClaimReconciler) collectPodLedgers(
	ctx context.Context,
	namespace string,
	candidates []corev1.Pod,
	states map[string]PodPlacementState,
) map[string]podLedger {
	ledgers := make(map[string]podLedger, len(candidates))
	for i := range candidates {
		pod := &candidates[i]
		state, found := states[pod.Name]
		switch {
		case podGPUCount(*pod) == 0:
			ledgers[pod.Name] = podLedger{State: ledgerNoGPU, HBMUsableBytes: hbmUsableUnknown}
		case !found:
			ledgers[pod.Name] = podLedger{State: ledgerUnknown, HBMUsableBytes: hbmUsableUnknown}
		case !state.HBMUsableKnown:
			ledgers[pod.Name] = podLedger{State: ledgerGPUMeasureFailed, HBMUsableBytes: hbmUsableUnknown}
		default:
			ledgers[pod.Name] = podLedger{
				State:          ledgerComplete,
				HBMUsableBytes: state.HBMUsableBytes,
			}
		}
	}

	// Deliberately not the cached client. An instance written moments ago may
	// not have reached the informer yet, and an instance missing from the
	// account is memory a second claim would be told is free.
	reader := client.Reader(r.Client)
	if r.LedgerReader != nil {
		reader = r.LedgerReader
	}
	list := &modelv1alpha1.ModelClaimList{}
	if err := reader.List(ctx, list, client.InNamespace(namespace)); err != nil {
		// Without the claim list every ledger would understate what its card
		// already owes, which is the one direction that overcommits a GPU.
		klog.ErrorS(err, "collect pod ledgers: list model claims", "namespace", namespace)
		for name := range ledgers {
			ledgers[name] = podLedger{State: ledgerUnknown, HBMUsableBytes: hbmUsableUnknown}
		}
		return ledgers
	}

	for i := range list.Items {
		claim := &list.Items[i]
		key := types.NamespacedName{Namespace: claim.Namespace, Name: claim.Name}
		for _, instance := range claim.Status.Instances {
			// A failed instance has exhausted its restarts and its engine
			// process is gone, so its memory is back with the card. Charging
			// the card for it would take a slice of GPU out of circulation for
			// as long as the claim exists, and nothing would ever put it back.
			// An activating instance is charged, and deliberately: placement
			// has already committed those bytes, and waiting for readiness
			// would let a second claim be placed against the same memory.
			if instance.Phase == modelv1alpha1.ModelClaimFailed {
				continue
			}
			ledger := ledgerFor(ledgers, instance.Pod)
			if !ledger.chargeable() {
				continue
			}
			ledger.Instances = append(ledger.Instances, ledgerInstance{
				Claim:                 key,
				MaximumFootprintBytes: claim.Spec.PerGPU.MaximumFootprintBytes,
				KVFloorBytes:          claim.Spec.PerGPU.KVFloorBytes,
			})
			ledgers[instance.Pod] = ledger
		}
	}
	return ledgers
}
