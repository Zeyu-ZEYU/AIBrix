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
	ordered := rankCandidates(filterCandidates(candidates, alreadyOn), load, model, locality, states)
	if len(ordered) == 0 {
		return nil, fmt.Errorf("no available candidate warm pod for model")
	}
	return ordered[0], nil
}

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

// filterCandidates applies the hard constraints and returns the pods that
// could host this model, in the order they arrived. Preference is not its job,
// so it deliberately does not reorder.
func filterCandidates(candidates []corev1.Pod, alreadyOn map[string]bool) []*corev1.Pod {
	feasible := make([]*corev1.Pod, 0, len(candidates))
	for i := range candidates {
		// A model's instances have to sit on different pods: a second engine
		// for the same model on the same card would compete with the first for
		// memory it is already counted as holding.
		if alreadyOn[candidates[i].Name] {
			continue
		}
		feasible = append(feasible, &candidates[i])
	}
	return feasible
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
