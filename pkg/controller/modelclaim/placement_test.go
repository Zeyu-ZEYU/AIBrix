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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	modelv1alpha1 "github.com/vllm-project/aibrix/api/model/v1alpha1"
)

func namedPod(name string) corev1.Pod {
	return corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: name}}
}

func ptrPod(name string) *corev1.Pod {
	pod := namedPod(name)
	return &pod
}

func TestSelectPodForActivation_LeastLoaded(t *testing.T) {
	cands := []corev1.Pod{namedPod("a"), namedPod("b"), namedPod("c")}
	load := map[string]int{"a": 2, "b": 0, "c": 1}
	got, err := selectPodForActivation(cands, map[string]bool{}, load, "m", uniformLocality{})
	require.NoError(t, err)
	assert.Equal(t, "b", got.Name)
}

func TestSelectPodForActivation_SkipsAlreadyOn(t *testing.T) {
	cands := []corev1.Pod{namedPod("a"), namedPod("b")}
	load := map[string]int{"a": 0, "b": 5}
	got, err := selectPodForActivation(cands, map[string]bool{"a": true}, load, "m", uniformLocality{})
	require.NoError(t, err)
	assert.Equal(t, "b", got.Name, "a is excluded even though least loaded")
}

func TestSelectPodForActivation_TieBreakByName(t *testing.T) {
	cands := []corev1.Pod{namedPod("z"), namedPod("a")}
	load := map[string]int{"z": 0, "a": 0}
	got, err := selectPodForActivation(cands, map[string]bool{}, load, "m", uniformLocality{})
	require.NoError(t, err)
	assert.Equal(t, "a", got.Name)
}

func TestSelectPodForActivation_NoCapacity(t *testing.T) {
	cands := []corev1.Pod{namedPod("a")}
	_, err := selectPodForActivation(cands, map[string]bool{"a": true}, map[string]int{}, "m", uniformLocality{})
	assert.Error(t, err)
}

func TestServedModelName(t *testing.T) {
	pm := &modelv1alpha1.ModelClaim{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
	assert.Equal(t, "foo", servedModelName(pm))
	name := "bar"
	pm.Spec.ModelName = &name
	assert.Equal(t, "bar", servedModelName(pm))
}

func TestIpcNameFor(t *testing.T) {
	pm := &modelv1alpha1.ModelClaim{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
	assert.Equal(t, "kvc_foo", ipcNameFor(pm))

	// Sanitized to match kvcached's normalization (verified on real hardware):
	// '.' and '/' become '-', existing '-' is kept.
	dotted := &modelv1alpha1.ModelClaim{ObjectMeta: metav1.ObjectMeta{Name: "qwen3-0.6b"}}
	assert.Equal(t, "kvc_qwen3-0-6b", ipcNameFor(dotted))
	slashed := &modelv1alpha1.ModelClaim{ObjectMeta: metav1.ObjectMeta{Name: "Qwen/Qwen2-7B"}}
	assert.Equal(t, "kvc_Qwen-Qwen2-7B", ipcNameFor(slashed))
}

func TestDesiredReplicas(t *testing.T) {
	pm := &modelv1alpha1.ModelClaim{}
	assert.Equal(t, int32(1), desiredReplicas(pm))
	one := int32(1)
	pm.Spec.Replicas = &one
	assert.Equal(t, int32(1), desiredReplicas(pm))
}

// fakeLocality maps nodeName -> load cost for tests (0 = weights already hot).
type fakeLocality map[string]float64

func (f fakeLocality) Cost(model, nodeName string) float64 { return f[nodeName] }

func podOnNode(name, node string) corev1.Pod {
	return corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec:       corev1.PodSpec{NodeName: node},
	}
}

func TestSelectPodForActivation_LocalityDominatesLoad(t *testing.T) {
	// "hot" sits on a node whose store already has the weights (cost 0) but is
	// busier; "cold" is idle but on a node that must stage weights (cost 5).
	cands := []corev1.Pod{podOnNode("cold", "n-cold"), podOnNode("hot", "n-hot")}
	load := map[string]int{"cold": 0, "hot": 3}
	loc := fakeLocality{"n-hot": 0, "n-cold": 5}
	got, err := selectPodForActivation(cands, map[string]bool{}, load, "m", loc)
	require.NoError(t, err)
	assert.Equal(t, "hot", got.Name, "lower locality cost wins over lower load")
}

func TestSelectPodForActivation_LoadBreaksEqualLocality(t *testing.T) {
	// Two nodes equally hot (cost 0): fall back to least-loaded.
	cands := []corev1.Pod{podOnNode("a", "n1"), podOnNode("b", "n2")}
	load := map[string]int{"a": 2, "b": 1}
	loc := fakeLocality{"n1": 0, "n2": 0}
	got, err := selectPodForActivation(cands, map[string]bool{}, load, "m", loc)
	require.NoError(t, err)
	assert.Equal(t, "b", got.Name)
}

func TestSelectPodForActivation_NilLocalityIsUniform(t *testing.T) {
	// A nil provider must not panic and must behave like load-only selection.
	cands := []corev1.Pod{podOnNode("a", "n1"), podOnNode("b", "n2")}
	load := map[string]int{"a": 5, "b": 0}
	got, err := selectPodForActivation(cands, map[string]bool{}, load, "m", nil)
	require.NoError(t, err)
	assert.Equal(t, "b", got.Name)
}

func TestSelectPodForActivationWithStatePrefersLiveRuntimeState(t *testing.T) {
	candidates := []corev1.Pod{namedPod("cold"), namedPod("hot")}
	states := map[string]PodPlacementState{
		"cold": {
			SnapshotKnown: true,
			MemoryKnown:   true,
			HBMFreeBytes:  900,
			KVUsedBytes:   10,
			ModelCount:    1,
		},
		"hot": {
			SnapshotKnown:  true,
			ArtifactCached: true,
			MemoryKnown:    true,
			HBMFreeBytes:   100,
			KVUsedBytes:    100,
			ModelCount:     3,
		},
	}

	got, err := selectPodForActivationWithState(
		candidates, map[string]bool{}, map[string]int{}, "m", uniformLocality{}, states,
	)
	require.NoError(t, err)
	assert.Equal(t, "hot", got.Name, "cached artifact wins before live memory tie-breakers")
}

func TestSelectPodForActivationWithStateRanksMemoryAndKV(t *testing.T) {
	candidates := []corev1.Pod{namedPod("busy"), namedPod("free")}
	states := map[string]PodPlacementState{
		"busy": {
			SnapshotKnown: true,
			MemoryKnown:   true,
			HBMFreeBytes:  500,
			KVUsedBytes:   10,
			ModelCount:    1,
		},
		"free": {
			SnapshotKnown: true,
			MemoryKnown:   true,
			HBMFreeBytes:  600,
			KVUsedBytes:   100,
			ModelCount:    3,
		},
	}

	got, err := selectPodForActivationWithState(
		candidates, map[string]bool{}, map[string]int{}, "m", uniformLocality{}, states,
	)
	require.NoError(t, err)
	assert.Equal(t, "free", got.Name, "higher free HBM wins before KV/model-count tie-breakers")
}

func TestSelectPodForActivationWithStateFallsBackForUnknownSnapshots(t *testing.T) {
	candidates := []corev1.Pod{namedPod("busy"), namedPod("idle")}
	got, err := selectPodForActivationWithState(
		candidates,
		map[string]bool{},
		map[string]int{"busy": 2, "idle": 0},
		"m",
		uniformLocality{},
		map[string]PodPlacementState{},
	)
	require.NoError(t, err)
	assert.Equal(t, "idle", got.Name)
}

func TestUniformLocality_AlwaysZero(t *testing.T) {
	assert.Zero(t, uniformLocality{}.Cost("m", "any-node"))
}

func TestPruneDeadInstances(t *testing.T) {
	pm := &modelv1alpha1.ModelClaim{}
	pm.Status.Instances = []modelv1alpha1.ModelClaimInstance{
		{Pod: "alive", Port: 20000},
		{Pod: "gone", Port: 20001},
	}
	pruneDeadInstances(pm, []corev1.Pod{namedPod("alive")})
	require.Len(t, pm.Status.Instances, 1)
	assert.Equal(t, "alive", pm.Status.Instances[0].Pod,
		"instance on a vanished warm pod must be dropped so re-activation can run")

	// No candidates at all: every instance is stale.
	pruneDeadInstances(pm, nil)
	assert.Empty(t, pm.Status.Instances)
}

// roomLedger is a card with a known size and nothing on it, so a test states
// the room it wants directly.
func roomLedger(room int64) podLedger {
	return podLedger{HBMUsableBytes: room}
}

func TestRankCandidatesHeadMatchesTheSingleWinnerSearch(t *testing.T) {
	// Sorting has to reproduce the pod the old single-pass search returned, or
	// turning the gate off would silently change placement.
	cases := []struct {
		name      string
		pods      []corev1.Pod
		load      map[string]int
		alreadyOn map[string]bool
		states    map[string]PodPlacementState
	}{
		{
			name: "least loaded",
			pods: []corev1.Pod{namedPod("a"), namedPod("b"), namedPod("c")},
			load: map[string]int{"a": 2, "b": 0, "c": 1},
		},
		{
			name:      "skips already on",
			pods:      []corev1.Pod{namedPod("a"), namedPod("b")},
			load:      map[string]int{"a": 0, "b": 5},
			alreadyOn: map[string]bool{"a": true},
		},
		{
			name: "name breaks a full tie",
			pods: []corev1.Pod{namedPod("z"), namedPod("a")},
			load: map[string]int{"z": 0, "a": 0},
		},
		{
			name: "runtime state outranks load",
			pods: []corev1.Pod{namedPod("cold"), namedPod("hot")},
			load: map[string]int{"cold": 0, "hot": 9},
			states: map[string]PodPlacementState{
				"hot":  {SnapshotKnown: true, ArtifactCached: true},
				"cold": {SnapshotKnown: true},
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			alreadyOn := tc.alreadyOn
			if alreadyOn == nil {
				alreadyOn = map[string]bool{}
			}
			want, err := selectPodForActivationWithState(
				tc.pods, alreadyOn, tc.load, "m", uniformLocality{}, tc.states)
			require.NoError(t, err)
			ordered := rankCandidates(tc.pods, alreadyOn, tc.load, "m", uniformLocality{}, tc.states)
			require.NotEmpty(t, ordered)
			assert.Equal(t, want.Name, ordered[0].Name)
		})
	}
}

func TestRankCandidatesOrdersEveryEligiblePod(t *testing.T) {
	pods := []corev1.Pod{namedPod("c"), namedPod("a"), namedPod("b"), namedPod("taken")}
	load := map[string]int{"a": 1, "b": 1, "c": 0, "taken": 0}
	ordered := rankCandidates(pods, map[string]bool{"taken": true}, load, "m", uniformLocality{}, nil)

	names := make([]string, 0, len(ordered))
	for _, pod := range ordered {
		names = append(names, pod.Name)
	}
	assert.Equal(t, []string{"c", "a", "b"}, names,
		"least loaded first, then by name, and an occupied pod is not in the list")
}

func TestSelectPodForPlacementWalksPastTheFullestPod(t *testing.T) {
	// The ranking puts artifact locality first, so its favourite pod is
	// routinely the fullest. Walking down the list is the whole point.
	ordered := []*corev1.Pod{ptrPod("hot"), ptrPod("warm"), ptrPod("empty")}
	ledgers := map[string]podLedger{
		"hot":   roomLedger(1 * gibibyte),
		"warm":  roomLedger(3 * gibibyte),
		"empty": roomLedger(40 * gibibyte),
	}

	pod, refusals := selectPodForPlacement(ordered, ledgers, 6*gibibyte)
	require.NotNil(t, pod)
	assert.Equal(t, "empty", pod.Name)
	require.Len(t, refusals, 2)
	assert.Equal(t, "hot", refusals[0].Pod)
	assert.Equal(t, refusalInsufficientCapacity, refusals[0].Reason)
	assert.Equal(t, 1*gibibyte, refusals[0].MaximumRoomBytes)
	assert.Equal(t, "warm", refusals[1].Pod)
}

func TestSelectPodForPlacementTakesTheFirstPodThatFits(t *testing.T) {
	ordered := []*corev1.Pod{ptrPod("first"), ptrPod("second")}
	ledgers := map[string]podLedger{
		"first":  roomLedger(40 * gibibyte),
		"second": roomLedger(80 * gibibyte),
	}

	pod, refusals := selectPodForPlacement(ordered, ledgers, 6*gibibyte)
	require.NotNil(t, pod)
	assert.Equal(t, "first", pod.Name, "ranking decides among pods that fit, not size")
	assert.Empty(t, refusals)
}

func TestSelectPodForPlacementExactFitIsAccepted(t *testing.T) {
	ordered := []*corev1.Pod{ptrPod("exact")}
	ledgers := map[string]podLedger{"exact": roomLedger(6 * gibibyte)}

	pod, _ := selectPodForPlacement(ordered, ledgers, 6*gibibyte)
	require.NotNil(t, pod)
	assert.Equal(t, "exact", pod.Name)
}

func TestSelectPodForPlacementRefusesUnreadableLedgers(t *testing.T) {
	ordered := []*corev1.Pod{ptrPod("blind"), ptrPod("undeclared"), ptrPod("untracked")}
	ledgers := map[string]podLedger{
		"blind": {Missing: missingSnapshot},
		"undeclared": {
			HBMUsableBytes:  80 * gibibyte,
			Missing:         missingClaimNumbers,
			UndeclaredClaim: types.NamespacedName{Namespace: testNamespace, Name: "silent"},
		},
	}

	pod, refusals := selectPodForPlacement(ordered, ledgers, 6*gibibyte)
	assert.Nil(t, pod)
	require.Len(t, refusals, 3)
	for _, refusal := range refusals {
		assert.Equal(t, refusalLedgerIncomplete, refusal.Reason)
	}
	assert.Equal(t, missingClaimNumbers, refusals[1].Missing)
	assert.Equal(t, "silent", refusals[1].UndeclaredClaim.Name)
	assert.Equal(t, missingSnapshot, refusals[2].Missing,
		"a pod with no ledger at all is treated as unmeasured, never as empty")
}

func TestSummarizeRefusals(t *testing.T) {
	t.Run("no candidates at all", func(t *testing.T) {
		reason, message := summarizeRefusals(nil, 6*gibibyte)
		assert.Equal(t, "NoMatchingPods", reason)
		assert.Contains(t, message, "selector")
	})

	t.Run("every card is genuinely full", func(t *testing.T) {
		reason, message := summarizeRefusals([]podRefusal{
			{Pod: "a", Reason: refusalInsufficientCapacity, MaximumRoomBytes: 1, RoomKnown: true},
			{Pod: "b", Reason: refusalInsufficientCapacity, MaximumRoomBytes: 5, RoomKnown: true},
		}, 6*gibibyte)
		assert.Equal(t, refusalInsufficientCapacity, reason)
		assert.Contains(t, message, "at most 5", "the roomiest pod is the useful one to report")
	})

	t.Run("an unreadable ledger outranks a full card", func(t *testing.T) {
		reason, message := summarizeRefusals([]podRefusal{
			{Pod: "a", Reason: refusalInsufficientCapacity, MaximumRoomBytes: 1, RoomKnown: true},
			{
				Pod: "b", Reason: refusalLedgerIncomplete, Missing: missingClaimNumbers,
				UndeclaredClaim: types.NamespacedName{Namespace: testNamespace, Name: "silent"},
			},
		}, 6*gibibyte)
		assert.Equal(t, refusalLedgerIncomplete, reason,
			"a pod nobody could measure may well have had room")
		assert.Contains(t, message, "silent", "the claim to fix has to be named")
	})
}

func TestPlacementEventReason(t *testing.T) {
	assert.Equal(t, "PlacementOutOfMemory", placementEventReason(refusalInsufficientCapacity))
	assert.Equal(t, refusalLedgerIncomplete, placementEventReason(refusalLedgerIncomplete))
	assert.Equal(t, reasonNumbersMissing, placementEventReason(reasonNumbersMissing))
}
