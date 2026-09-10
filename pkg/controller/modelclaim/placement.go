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
	"fmt"
	"sort"
	"strings"

	modelv1alpha1 "github.com/vllm-project/aibrix/api/model/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
)

// servedModelName returns the model name clients address, defaulting to the
// object name when Spec.ModelName is unset.
func servedModelName(pm *modelv1alpha1.ModelClaim) string {
	if pm.Spec.ModelName != nil && *pm.Spec.ModelName != "" {
		return *pm.Spec.ModelName
	}
	return pm.Name
}

// desiredReplicas resolves the target number of active instances.
func desiredReplicas(pm *modelv1alpha1.ModelClaim) int32 {
	if pm.Spec.Replicas != nil {
		return *pm.Spec.Replicas
	}
	return 1
}

// ipcNameFor derives the kvcached shared-memory segment name for a model. It
// must be unique per GPU so co-tenant engines do not collide on /dev/shm, and
// it is sanitized to match kvcached's own normalization (it replaces characters
// like '.' and '/' with '-'); otherwise kvctl operations would target a
// different segment name than the engine actually created.
func ipcNameFor(pm *modelv1alpha1.ModelClaim) string {
	return "kvc_" + sanitizeIPCName(pm.Name)
}

// sanitizeIPCName maps any character outside [A-Za-z0-9_-] to '-', matching how
// kvcached normalizes the KVCACHED_IPC_NAME (verified on real hardware: a name
// like "kvc_qwen3-0.6b" becomes "kvc_qwen3-0-6b").
func sanitizeIPCName(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '_', r == '-':
			b.WriteRune(r)
		default:
			b.WriteRune('-')
		}
	}
	return b.String()
}

// LocalityProvider is an optional advisory estimate for bringing a model's
// weights onto a node. Runtime snapshots supply the authoritative per-pod
// artifact signal; this interface remains for future node-level hints. A zero
// cost means "already hot", and uniformLocality preserves load-only fallback.
type LocalityProvider interface {
	Cost(model, nodeName string) float64
}

// uniformLocality is the default: every node looks equally cheap, so locality
// never tips the decision and placement falls back to load + name ordering.
type uniformLocality struct{}

func (uniformLocality) Cost(model, nodeName string) float64 { return 0 }

// selectPodForActivation picks a warm pod to attach the model to. Among pods not
// already hosting this model, it chooses the lowest-cost pod, ranked
// lexicographically by (locality cost, current model load, name): prefer a node
// where the weights are already hot, then the least-loaded pod for density
// spread, breaking remaining ties by name for determinism.
//
// A nil provider is treated as uniform (load-only), preserving the existing
// deterministic fallback when runtime observations are unavailable.
func selectPodForActivation(candidates []corev1.Pod, alreadyOn map[string]bool, load map[string]int, model string, locality LocalityProvider) (*corev1.Pod, error) {
	return selectPodForActivationWithState(candidates, alreadyOn, load, model, locality, nil)
}

// selectPodForActivationWithState first prefers a pod that already has the
// artifact locally, then live GPU/KV observations, and finally the Phase-1
// locality/load/name rank. Missing runtime state is safe: it simply falls back
// to the existing deterministic placement behavior.
func selectPodForActivationWithState(
	candidates []corev1.Pod,
	alreadyOn map[string]bool,
	load map[string]int,
	model string,
	locality LocalityProvider,
	states map[string]PodPlacementState,
) (*corev1.Pod, error) {
	ordered := rankCandidates(candidates, alreadyOn, load, model, locality, states)
	if len(ordered) == 0 {
		return nil, fmt.Errorf("no available candidate warm pod for model")
	}
	return ordered[0], nil
}

// rankCandidates orders every pod that could take this model, best first. It
// only ranks: preference and admission are separate questions, and mixing them
// would let artifact locality win a pod that has no room at all.
//
// The comparison is the one the previous single-winner search used, so the
// head of this list is the pod that search would have returned. It ends in a
// name comparison and is therefore a strict total order over distinct pods,
// which makes the sort deterministic.
func rankCandidates(
	candidates []corev1.Pod,
	alreadyOn map[string]bool,
	load map[string]int,
	model string,
	locality LocalityProvider,
	states map[string]PodPlacementState,
) []*corev1.Pod {
	if locality == nil {
		locality = uniformLocality{}
	}
	ordered := make([]*corev1.Pod, 0, len(candidates))
	for i := range candidates {
		if alreadyOn[candidates[i].Name] {
			continue
		}
		ordered = append(ordered, &candidates[i])
	}
	sort.SliceStable(ordered, func(i, j int) bool {
		a, b := ordered[i], ordered[j]
		stateA, stateB := states[a.Name], states[b.Name]
		if placementStateLess(stateA, stateB) {
			return true
		}
		if placementStateLess(stateB, stateA) {
			return false
		}
		return rankLess(
			locality.Cost(model, a.Spec.NodeName), load[a.Name], a.Name,
			locality.Cost(model, b.Spec.NodeName), load[b.Name], b.Name,
		)
	})
	return ordered
}

// Refusal reasons are fixed strings because they reach an operator as a
// condition reason. They separate two situations an operator has to act on
// differently: a card that provably cannot hold the model needs capacity
// added or a model removed, while a ledger that could not be read needs a
// sidecar or a ModelClaim fixed.
const (
	refusalInsufficientCapacity = "InsufficientCapacity"
	refusalLedgerIncomplete     = "LedgerIncomplete"
)

// podRefusal records why one pod could not take the model, so the claim can
// say which pods were considered and what stopped each of them.
type podRefusal struct {
	Pod              string
	Reason           string
	MaximumRoomBytes int64
	RoomKnown        bool
	Missing          ledgerMissing
	UndeclaredClaim  types.NamespacedName
}

// selectPodForPlacement walks the ranked pods and returns the first one whose
// ledger proves it can hold a model needing minimumReserveBytes.
//
// Walking matters: ranking puts artifact locality first, so the best-ranked
// pod is routinely the fullest one, and room does not decrease down the list.
// A pod that fails here is skipped, not fatal to the round.
//
// This is the first of the design's gates, the only one that can prove a
// placement impossible: maximumRoomBytes is what the card could offer if every
// engine on it went down to its KV floor, which needs sleeps and is therefore
// the most room that will ever exist there. Needing more than that cannot be
// fixed by waiting. The gates that decide whether a placement is merely
// inconvenient, against currentRoomBytes and minimumRoomBytes, are not
// implemented yet, so a pod that passes here is placed on directly.
func selectPodForPlacement(
	ordered []*corev1.Pod,
	ledgers map[string]podLedger,
	minimumReserveBytes int64,
) (*corev1.Pod, []podRefusal) {
	refusals := make([]podRefusal, 0, len(ordered))
	for _, pod := range ordered {
		ledger, tracked := ledgers[pod.Name]
		if !tracked {
			ledger = podLedger{Missing: missingSnapshot}
		}
		room, known := ledger.MaximumRoomBytes()
		if !known {
			refusals = append(refusals, podRefusal{
				Pod:             pod.Name,
				Reason:          refusalLedgerIncomplete,
				Missing:         ledger.Missing,
				UndeclaredClaim: ledger.UndeclaredClaim,
			})
			continue
		}
		if room < minimumReserveBytes {
			refusals = append(refusals, podRefusal{
				Pod:              pod.Name,
				Reason:           refusalInsufficientCapacity,
				MaximumRoomBytes: room,
				RoomKnown:        true,
			})
			continue
		}
		return pod, refusals
	}
	return nil, refusals
}

// summarizeRefusals turns the per-pod record into the one reason and message a
// claim carries. An unreadable ledger outranks a full card, because a pod
// nobody could measure may well have had room.
func summarizeRefusals(refusals []podRefusal, minimumReserveBytes int64) (string, string) {
	if len(refusals) == 0 {
		return "NoMatchingPods", "no warm pod matched the claim's selector"
	}
	reason := refusalInsufficientCapacity
	bestRoom, bestKnown := int64(0), false
	var incomplete *podRefusal
	for i := range refusals {
		refusal := &refusals[i]
		if refusal.Reason == refusalLedgerIncomplete && incomplete == nil {
			incomplete = refusal
		}
		if refusal.RoomKnown && (!bestKnown || refusal.MaximumRoomBytes > bestRoom) {
			bestRoom, bestKnown = refusal.MaximumRoomBytes, true
		}
	}
	if incomplete != nil {
		reason = refusalLedgerIncomplete
		message := fmt.Sprintf("%d candidate pod(s) refused; pod %s could not be judged: %s",
			len(refusals), incomplete.Pod, incomplete.Missing)
		if incomplete.Missing == missingClaimNumbers {
			message += fmt.Sprintf(" (%s)", incomplete.UndeclaredClaim)
		}
		return reason, message
	}
	return reason, fmt.Sprintf(
		"%d candidate pod(s) refused; model needs %d bytes per GPU, the roomiest pod could free at most %d",
		len(refusals), minimumReserveBytes, bestRoom)
}

// placementStateLess returns whether a ranks ahead of b using live runtime
// state. `false` in both directions means the legacy locality/load tie-breaker
// decides the winner.
func placementStateLess(a, b PodPlacementState) bool {
	if a.ArtifactCached != b.ArtifactCached {
		return a.ArtifactCached
	}
	if a.SnapshotKnown != b.SnapshotKnown {
		return a.SnapshotKnown
	}
	if a.MemoryKnown != b.MemoryKnown {
		return a.MemoryKnown
	}
	if a.MemoryKnown && a.HBMFreeBytes != b.HBMFreeBytes {
		return a.HBMFreeBytes > b.HBMFreeBytes
	}
	if a.SnapshotKnown && a.KVUsedBytes != b.KVUsedBytes {
		return a.KVUsedBytes < b.KVUsedBytes
	}
	if a.SnapshotKnown && a.ModelCount != b.ModelCount {
		return a.ModelCount < b.ModelCount
	}
	return false
}

// rankLess reports whether candidate (loc,load,name) ranks before
// (bLoc,bLoad,bName): lower locality cost first, then lower load, then lower name.
func rankLess(loc float64, load int, name string, bLoc float64, bLoad int, bName string) bool {
	if loc != bLoc {
		return loc < bLoc
	}
	if load != bLoad {
		return load < bLoad
	}
	return name < bName
}
