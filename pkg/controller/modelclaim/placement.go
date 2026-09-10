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

// Placement asks two different questions about a warm pod, and one comparison
// cannot answer both. Whether a pod *can* host this model is a hard
// constraint: failing one takes the pod out of the running entirely. Which of
// the surviving pods is *preferable* is a soft one: it only orders them, and
// any of them would serve.
//
// Keeping them apart matters for two reasons. A preference must never outvote
// a constraint, which a single ordering cannot guarantee. And a refusal has to
// be explainable: an ordering can say a pod came last, never why it was
// unusable.
//
// Hard constraints also live in listCandidateWarmPods, which drops pods that
// are not usable warm pool members at all: not enabled, not Running, without
// an IP, terminating, or holding a number of GPUs that does not match this
// model's parallelism. Those define the candidate set. Everything below
// decides among candidates.

// podRefusal records a pod the filter turned down, together with the figure
// the decision was made from, so a claim can say why it is waiting rather than
// only that it is.
//
// It carries no reason field because there is only one refusal to record: a
// card provably too full. Pods dropped for already hosting this model are not
// listed, since "the model is already there" asks nothing of an operator and
// would only dilute the real cause.
type podRefusal struct {
	Pod              string
	MaximumRoomBytes int64
}

// filterCandidates applies the hard constraints and returns the pods that
// could host this model, in the order they arrived. Preference is not its job,
// so it deliberately does not reorder.
//
// minimumReserveBytes is what one instance of the model costs on a card, and
// is always positive: spec.perGPU is required and both of its fields are
// validated above zero, so a claim that reached a controller has them.
func filterCandidates(
	candidates []corev1.Pod,
	alreadyOn map[string]bool,
	ledgers map[string]podLedger,
	minimumReserveBytes int64,
) ([]*corev1.Pod, []podRefusal) {
	feasible := make([]*corev1.Pod, 0, len(candidates))
	var refusals []podRefusal
	for i := range candidates {
		pod := &candidates[i]
		// A model's instances have to sit on different pods: a second engine
		// for the same model on the same card would compete with the first for
		// memory it is already counted as holding.
		if alreadyOn[pod.Name] {
			continue
		}
		if room, tooFull := provablyTooFull(ledgers[pod.Name], minimumReserveBytes); tooFull {
			refusals = append(refusals, podRefusal{Pod: pod.Name, MaximumRoomBytes: room})
			continue
		}
		feasible = append(feasible, pod)
	}
	return feasible, refusals
}

// provablyTooFull reports whether this card can be shown to have no room for a
// model needing minimumReserveBytes, and if so how much it does have.
//
// maximumRoomBytes is what the card would offer if every instance on it
// dropped to the KV floor its claim declared, which needs those engines put to
// sleep. It is therefore the most room that will ever exist there, and needing
// more than it cannot be fixed by waiting.
//
// It answers false whenever the question cannot be settled, which is the whole
// of its caution: a pod Kubernetes gave no GPU is not judged on GPU memory,
// and a ledger that could not be read is a different constraint, not this one.
// Both admit the pod exactly as it was admitted before this check existed.
func provablyTooFull(ledger podLedger, minimumReserveBytes int64) (int64, bool) {
	if ledger.State == ledgerNoGPU {
		return 0, false
	}
	room, known := ledger.MaximumRoomBytes()
	if !known {
		return 0, false
	}
	return room, room < minimumReserveBytes
}

// summarizeRefusals states, in one line an operator can act on, how far the
// pool is from holding this model. It reports the roomiest refused pod rather
// than listing every one: the gap that matters is the smallest one, and a list
// would grow with the pool.
func summarizeRefusals(refusals []podRefusal, minimumReserveBytes int64) string {
	roomiest := refusals[0].MaximumRoomBytes
	for _, refusal := range refusals[1:] {
		if refusal.MaximumRoomBytes > roomiest {
			roomiest = refusal.MaximumRoomBytes
		}
	}
	return fmt.Sprintf(
		"no warm pod has room: this model needs %s per GPU, and of %d candidate pod(s) the roomiest could free at most %s",
		gibibytes(minimumReserveBytes), len(refusals), gibibytes(roomiest),
	)
}

// gibibytes renders a byte count for someone reading a status condition.
// Placement figures are GPU-sized, so GiB to one decimal stays readable while
// keeping two of them comparable at a glance.
func gibibytes(bytes int64) string {
	return fmt.Sprintf("%.1f GiB", float64(bytes)/float64(1<<30))
}

// rankCandidates orders feasible pods, best first, sorting in place and
// returning the same slice.
//
// The comparison is the one the previous single-winner search used. It ends in
// a name comparison, and distinct pods have distinct names, so it is a strict
// total order. That makes the sort deterministic, and it makes the head of
// this list exactly the pod that search would have returned.
func rankCandidates(
	feasible []*corev1.Pod,
	load map[string]int,
	model string,
	locality LocalityProvider,
	states map[string]PodPlacementState,
) []*corev1.Pod {
	if locality == nil {
		locality = uniformLocality{}
	}
	sort.SliceStable(feasible, func(i, j int) bool {
		a, b := feasible[i], feasible[j]
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
	return feasible
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
