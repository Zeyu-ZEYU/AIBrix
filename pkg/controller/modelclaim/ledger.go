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

// defaultDriverReserveBytes is how much of a card never becomes usable memory:
// the CUDA context and driver structures that exist before any engine starts.
// Measured at 249 MiB on the 96 GB class card this was developed against, and
// rounded up so the estimate errs towards leaving room rather than claiming
// it. It is a property of the hardware and the driver, not of AIBrix, so it is
// passed in rather than read from here directly; a pool on different cards
// will eventually have to say its own figure.
const defaultDriverReserveBytes int64 = 256 << 20

// ledgerMissing says what a ledger could not find out. A ledger missing
// anything cannot say how much room a card has, and a decision that would
// claim a placement is impossible must not be made from an incomplete account.
type ledgerMissing int

const (
	// missingNothing is the zero value on purpose: an account is complete
	// until something is found to be absent.
	missingNothing ledgerMissing = iota
	// missingSnapshot means the runtime sidecar did not answer, so nothing at
	// all is known about this pod's cards.
	missingSnapshot
	// missingCardSize means the pod was allocated GPUs but the runtime did not
	// report a usable size for them: it saw no accelerator at all, or not as
	// many as this model's parallelism spans.
	missingCardSize
	// missingClaimNumbers means an instance already on this card belongs to a
	// ModelClaim that never declared spec.perGPU, so its share of the card
	// cannot be counted.
	missingClaimNumbers
)

func (m ledgerMissing) String() string {
	switch m {
	case missingSnapshot:
		return "runtime snapshot unavailable"
	case missingCardSize:
		return "the runtime reported no usable size for this pod's GPUs"
	case missingClaimNumbers:
		return "an instance on this pod has no declared spec.perGPU"
	default:
		return ""
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
// already been promised part of it. A pod that spans several cards is
// described by its tightest one, and that is exact rather than conservative:
// every instance on such a pod occupies all of its cards, so the cards differ
// only in size.
type podLedger struct {
	HBMUsableBytes int64
	Instances      []ledgerInstance
	Missing        ledgerMissing
	// UndeclaredClaim names the claim that made this ledger incomplete, so an
	// operator is told which ModelClaim to fix rather than only that one
	// exists. Set only with missingClaimNumbers.
	UndeclaredClaim types.NamespacedName
	// NoGPU records that Kubernetes allocated this pod no GPU at all, read
	// from the pod's nvidia.com/gpu resources rather than from anything the
	// runtime said about itself. There is then no GPU memory to keep an
	// account of. It is deliberately not inferred from an empty accelerator
	// list: a pod that does hold a card reports the same empty list when NVML
	// is missing or the device was never mounted into the container, and
	// treating that pod as GPU-free would hide a card in an unknown state.
	NoGPU bool
}

// accountable reports whether this pod has a card the ledger can keep an
// account of. A pod with no GPU, or one whose size could not be established,
// has nothing to charge an instance against.
func (l podLedger) accountable() bool {
	return !l.NoGPU && l.HBMUsableBytes > 0
}

// MaximumRoomBytes is the most memory this card could ever offer a new
// instance: what is left once every instance already on it is down to the KV
// floor its claim declared. Reaching it means putting engines to sleep, so a
// model needing more than this cannot be placed here by waiting. The value can
// be negative if the card is already promised more than it has.
//
// The second return is false when the ledger is incomplete, in which case the
// first has no meaning.
func (l podLedger) MaximumRoomBytes() (int64, bool) {
	if l.Missing != missingNothing {
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
// The second return is false when the claim did not declare both numbers.
func claimMinimumReserveBytes(pm *modelv1alpha1.ModelClaim) (int64, bool) {
	if pm == nil || pm.Spec.PerGPU == nil {
		return 0, false
	}
	footprint, floor := pm.Spec.PerGPU.MaximumFootprintBytes, pm.Spec.PerGPU.KVFloorBytes
	if footprint == nil || floor == nil || *footprint <= 0 || *floor <= 0 {
		return 0, false
	}
	return *footprint + *floor, true
}

// hbmUsableBytes is how much of a pod's GPU memory can ever hold an engine.
// It follows the same rule placementStateFromSnapshot uses for free memory: a
// single-GPU engine wants the largest device, a fixed parallelism group spans
// every device and is limited by the smallest.
//
// reserveBytes is what the driver keeps for itself on each card. It is a
// property of the hardware, so the caller supplies it.
func hbmUsableBytes(snapshot *RuntimeSnapshot, parallelism, reserveBytes int64) (int64, bool) {
	if snapshot == nil || len(snapshot.Accelerators) == 0 {
		return 0, false
	}
	if parallelism < 1 {
		parallelism = 1
	}
	if parallelism > 1 && int64(len(snapshot.Accelerators)) != parallelism {
		return 0, false
	}
	usable := int64(0)
	known := false
	for _, accelerator := range snapshot.Accelerators {
		if accelerator.HBMTotalBytes <= reserveBytes {
			continue
		}
		candidate := accelerator.HBMTotalBytes - reserveBytes
		if !known || (parallelism == 1 && candidate > usable) ||
			(parallelism > 1 && candidate < usable) {
			usable, known = candidate, true
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
			ledgers[pod.Name] = podLedger{NoGPU: true}
		case !found:
			ledgers[pod.Name] = podLedger{Missing: missingSnapshot}
		case !state.HBMUsableKnown:
			ledgers[pod.Name] = podLedger{Missing: missingCardSize}
		default:
			ledgers[pod.Name] = podLedger{HBMUsableBytes: state.HBMUsableBytes}
		}
	}

	list := &modelv1alpha1.ModelClaimList{}
	if err := r.List(ctx, list, client.InNamespace(namespace)); err != nil {
		// Without the claim list every ledger would understate what its card
		// already owes, which is the one direction that overcommits a GPU.
		klog.ErrorS(err, "collect pod ledgers: list model claims", "namespace", namespace)
		for name, ledger := range ledgers {
			ledger.Missing = missingSnapshot
			ledgers[name] = ledger
		}
		return ledgers
	}

	for i := range list.Items {
		claim := &list.Items[i]
		_, declared := claimMinimumReserveBytes(claim)
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
			ledger, tracked := ledgers[instance.Pod]
			if !tracked || !ledger.accountable() {
				continue
			}
			if !declared {
				if ledger.Missing == missingNothing {
					ledger.Missing = missingClaimNumbers
					ledger.UndeclaredClaim = key
				}
				ledgers[instance.Pod] = ledger
				continue
			}
			ledger.Instances = append(ledger.Instances, ledgerInstance{
				Claim:                 key,
				MaximumFootprintBytes: *claim.Spec.PerGPU.MaximumFootprintBytes,
				KVFloorBytes:          *claim.Spec.PerGPU.KVFloorBytes,
			})
			ledgers[instance.Pod] = ledger
		}
	}
	return ledgers
}
