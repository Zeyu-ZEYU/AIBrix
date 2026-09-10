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
	"fmt"
	"hash/fnv"
	"sort"

	modelv1alpha1 "github.com/vllm-project/aibrix/api/model/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// The ledger is the control plane's own account of one GPU: which instances it
// has already promised space to, and how much space that leaves. It is built
// from what users declared on their ModelClaims, not from what the card
// currently reports, so it does not move when traffic moves. NVML supplies one
// number only, how large the card is.
//
// The design document this implements calls the three derived quantities
// room_waterline, room_now and room_floor. This code names them by size
// instead, because the gates run in size order and the names then say why:
//
//	room_waterline -> minimumRoomBytes   nobody is touched            (not yet implemented)
//	room_now       -> currentRoomBytes   lines lowered, no eviction   (not yet implemented)
//	room_floor     -> maximumRoomBytes   everyone at their KV floor, reachable only by sleep
//
// minimumRoomBytes <= currentRoomBytes <= maximumRoomBytes always holds,
// because an instance's KV waterline is never below its KV floor. Only
// maximumRoomBytes exists today: it is the one gate that can prove a placement
// impossible rather than merely inconvenient.

// driverReserveBytes is the slice of a card that never becomes usable memory:
// the CUDA context and driver structures that exist before any engine starts.
// Measured at 0.24 GiB on the 96 GB class card this was developed against, and
// rounded up so the estimate errs towards leaving room rather than claiming it.
const driverReserveBytes int64 = 256 << 20

// ledgerMissing says what a ledger could not find out. A ledger that is
// missing anything cannot answer how much room a card has, and placement
// refuses the pod rather than guessing, because the gate it feeds claims a
// placement is impossible.
type ledgerMissing int

const (
	// missingNothing is the zero value on purpose: a ledger is complete until
	// something is found to be absent.
	missingNothing ledgerMissing = iota
	// missingSnapshot means the runtime sidecar did not answer, so nothing at
	// all is known about this pod's cards. Silence is not evidence that a card
	// is empty, which is why it refuses rather than admits.
	missingSnapshot
	// missingCardSize means the sidecar answered and does have accelerators,
	// but not the number this model's parallelism spans, so there is no
	// meaningful card size to charge the model against.
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
		return "the pod's accelerators do not match the model's parallelism"
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

// podLedger is the account of the GPU behind one warm pod. On a pod with
// several cards it describes the tightest of them, which is the correct
// reading for an engine that spans all of them.
type podLedger struct {
	HBMUsableBytes int64
	Instances      []ledgerInstance
	Missing        ledgerMissing
	// UndeclaredClaim names the instance's claim that made this ledger
	// incomplete, so an operator is told which ModelClaim to fix rather than
	// only that one exists. Set only with missingClaimNumbers.
	UndeclaredClaim types.NamespacedName
	// NoAccelerator records that the runtime answered and reported no GPU at
	// all. There is then no GPU memory to keep an account of, and a gate about
	// GPU memory has nothing to say about this pod. It is the CPU-only and
	// mock-engine case, and it is not the same as a runtime that stayed
	// silent.
	NoAccelerator bool
}

// accountable reports whether this pod has a card the ledger can keep an
// account of. A pod with no GPU, or one whose size could not be established,
// has nothing to charge an instance against.
func (l podLedger) accountable() bool {
	return !l.NoAccelerator && l.HBMUsableBytes > 0
}

// MaximumRoomBytes is the most memory this card could ever offer a new
// instance: what is left once every instance already on it is down to its KV
// floor. Reaching it means putting engines to sleep, so a placement that needs
// more than this is impossible rather than merely delayed. The value can be
// negative if the card is already promised more than it has.
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
// floor. The engine's KV limit can be lowered towards the floor but never past
// it, so this is a lower bound on occupancy, not an estimate of it.
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
func hbmUsableBytes(snapshot *RuntimeSnapshot, parallelism int64) (int64, bool) {
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
		if accelerator.HBMTotalBytes <= driverReserveBytes {
			continue
		}
		candidate := accelerator.HBMTotalBytes - driverReserveBytes
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
// account survives a controller restart and does not depend on engines being
// reachable.
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
		case !found:
			ledgers[pod.Name] = podLedger{Missing: missingSnapshot}
		case state.NoAccelerator:
			ledgers[pod.Name] = podLedger{NoAccelerator: true}
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

// ledgerFingerprint summarizes every candidate's account in one value, so a
// retry can tell "nothing has moved" from "the account changed". It covers the
// set of candidate pods, each one's room, and whether it could be read at all.
// Pods are visited in name order and the room figures are ledger quantities
// rather than live readings, so the fingerprint is stable while the cards are.
func ledgerFingerprint(ledgers map[string]podLedger) uint64 {
	names := make([]string, 0, len(ledgers))
	for name := range ledgers {
		names = append(names, name)
	}
	sort.Strings(names)
	digest := fnv.New64a()
	for _, name := range names {
		room, known := ledgers[name].MaximumRoomBytes()
		fmt.Fprintf(digest, "%s=%d,%t,%d;", name, room, known, ledgers[name].Missing)
	}
	return digest.Sum64()
}
