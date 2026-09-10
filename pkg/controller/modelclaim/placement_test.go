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

// mustFilter runs the filter with no ledger, for the tests that are about the
// other constraints and say nothing about memory.
func mustFilter(candidates []corev1.Pod, alreadyOn map[string]bool) []*corev1.Pod {
	feasible, refusals := filterCandidates(candidates, alreadyOn, nil, 0)
	if len(refusals) != 0 {
		panic("filter refused a pod for memory with no ledger and no declared cost")
	}
	return feasible
}

// topChoice composes the filter and the ranker the way the controller does,
// with no ledger, so the tests written before the split keep asserting exactly
// what they always did. Nil means every candidate was filtered out.
func topChoice(
	candidates []corev1.Pod,
	alreadyOn map[string]bool,
	load map[string]int,
	model string,
	locality LocalityProvider,
	states map[string]PodPlacementState,
) *corev1.Pod {
	feasible, _ := filterCandidates(candidates, alreadyOn, nil, 0)
	ordered := rankCandidates(feasible, load, model, locality, states)
	if len(ordered) == 0 {
		return nil
	}
	return ordered[0]
}

func TestSelectPodForActivation_LeastLoaded(t *testing.T) {
	cands := []corev1.Pod{namedPod("a"), namedPod("b"), namedPod("c")}
	load := map[string]int{"a": 2, "b": 0, "c": 1}
	got := topChoice(cands, map[string]bool{}, load, "m", uniformLocality{}, nil)
	require.NotNil(t, got)
	assert.Equal(t, "b", got.Name)
}

func TestSelectPodForActivation_SkipsAlreadyOn(t *testing.T) {
	cands := []corev1.Pod{namedPod("a"), namedPod("b")}
	load := map[string]int{"a": 0, "b": 5}
	got := topChoice(cands, map[string]bool{"a": true}, load, "m", uniformLocality{}, nil)
	require.NotNil(t, got)
	assert.Equal(t, "b", got.Name, "a is excluded even though least loaded")
}

func TestSelectPodForActivation_TieBreakByName(t *testing.T) {
	cands := []corev1.Pod{namedPod("z"), namedPod("a")}
	load := map[string]int{"z": 0, "a": 0}
	got := topChoice(cands, map[string]bool{}, load, "m", uniformLocality{}, nil)
	require.NotNil(t, got)
	assert.Equal(t, "a", got.Name)
}

func TestSelectPodForActivation_NoCapacity(t *testing.T) {
	cands := []corev1.Pod{namedPod("a")}
	assert.Nil(t, topChoice(cands, map[string]bool{"a": true}, map[string]int{}, "m", uniformLocality{}, nil))
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
	got := topChoice(cands, map[string]bool{}, load, "m", loc, nil)
	require.NotNil(t, got)
	assert.Equal(t, "hot", got.Name, "lower locality cost wins over lower load")
}

func TestSelectPodForActivation_LoadBreaksEqualLocality(t *testing.T) {
	// Two nodes equally hot (cost 0): fall back to least-loaded.
	cands := []corev1.Pod{podOnNode("a", "n1"), podOnNode("b", "n2")}
	load := map[string]int{"a": 2, "b": 1}
	loc := fakeLocality{"n1": 0, "n2": 0}
	got := topChoice(cands, map[string]bool{}, load, "m", loc, nil)
	require.NotNil(t, got)
	assert.Equal(t, "b", got.Name)
}

func TestSelectPodForActivation_NilLocalityIsUniform(t *testing.T) {
	// A nil provider must not panic and must behave like load-only selection.
	cands := []corev1.Pod{podOnNode("a", "n1"), podOnNode("b", "n2")}
	load := map[string]int{"a": 5, "b": 0}
	got := topChoice(cands, map[string]bool{}, load, "m", nil, nil)
	require.NotNil(t, got)
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

	got := topChoice(
		candidates, map[string]bool{}, map[string]int{}, "m", uniformLocality{}, states,
	)
	require.NotNil(t, got)
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

	got := topChoice(
		candidates, map[string]bool{}, map[string]int{}, "m", uniformLocality{}, states,
	)
	require.NotNil(t, got)
	assert.Equal(t, "free", got.Name, "higher free HBM wins before KV/model-count tie-breakers")
}

func TestSelectPodForActivationWithStateFallsBackForUnknownSnapshots(t *testing.T) {
	candidates := []corev1.Pod{namedPod("busy"), namedPod("idle")}
	got := topChoice(
		candidates,
		map[string]bool{},
		map[string]int{"busy": 2, "idle": 0},
		"m",
		uniformLocality{},
		map[string]PodPlacementState{},
	)
	require.NotNil(t, got)
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

func TestFilterCandidatesDropsPodsAlreadyHostingTheModel(t *testing.T) {
	candidates := []corev1.Pod{namedPod("a"), namedPod("b"), namedPod("c")}
	feasible := mustFilter(candidates, map[string]bool{"b": true})

	names := make([]string, 0, len(feasible))
	for _, pod := range feasible {
		names = append(names, pod.Name)
	}
	assert.Equal(t, []string{"a", "c"}, names)
}

func TestFilterCandidatesKeepsTheInputOrder(t *testing.T) {
	// Ordering is the ranker's job. If the filter reordered, a preference
	// would be expressed by a function that is supposed to express only
	// feasibility.
	candidates := []corev1.Pod{namedPod("z"), namedPod("a"), namedPod("m")}
	feasible := mustFilter(candidates, map[string]bool{})

	require.Len(t, feasible, 3)
	assert.Equal(t, "z", feasible[0].Name)
	assert.Equal(t, "a", feasible[1].Name)
	assert.Equal(t, "m", feasible[2].Name)
}

func TestFilterCandidatesOnEmptyInput(t *testing.T) {
	assert.Empty(t, mustFilter(nil, map[string]bool{}))
	assert.Empty(t, mustFilter([]corev1.Pod{namedPod("a")}, map[string]bool{"a": true}))
}

func rankedNames(pods []*corev1.Pod) []string {
	names := make([]string, 0, len(pods))
	for _, pod := range pods {
		names = append(names, pod.Name)
	}
	return names
}

func TestRankCandidatesOrdersEveryPodNotJustTheWinner(t *testing.T) {
	// The whole point of ranking instead of picking: there is a second and a
	// third choice, and a later gate can walk to them.
	cases := []struct {
		name   string
		pods   []corev1.Pod
		load   map[string]int
		states map[string]PodPlacementState
		want   []string
	}{
		{
			name: "least loaded first, name breaks ties",
			pods: []corev1.Pod{namedPod("c"), namedPod("a"), namedPod("b")},
			load: map[string]int{"a": 1, "b": 1, "c": 0},
			want: []string{"c", "a", "b"},
		},
		{
			name: "a cached artifact outranks every load difference",
			pods: []corev1.Pod{namedPod("empty"), namedPod("cached")},
			load: map[string]int{"empty": 0, "cached": 9},
			states: map[string]PodPlacementState{
				"cached": {SnapshotKnown: true, ArtifactCached: true},
				"empty":  {SnapshotKnown: true},
			},
			want: []string{"cached", "empty"},
		},
		{
			name: "a known snapshot outranks an unknown one",
			pods: []corev1.Pod{namedPod("dark"), namedPod("seen")},
			states: map[string]PodPlacementState{
				"seen": {SnapshotKnown: true},
			},
			want: []string{"seen", "dark"},
		},
		{
			name: "more free memory first, then less KV in use, then fewer models",
			pods: []corev1.Pod{namedPod("busy"), namedPod("roomy"), namedPod("quiet")},
			states: map[string]PodPlacementState{
				"roomy": {SnapshotKnown: true, MemoryKnown: true, HBMFreeBytes: 40 * gibibyte},
				"quiet": {SnapshotKnown: true, MemoryKnown: true, HBMFreeBytes: 10 * gibibyte, KVUsedBytes: 1},
				"busy":  {SnapshotKnown: true, MemoryKnown: true, HBMFreeBytes: 10 * gibibyte, KVUsedBytes: 9},
			},
			want: []string{"roomy", "quiet", "busy"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			feasible := mustFilter(tc.pods, map[string]bool{})
			ordered := rankCandidates(feasible, tc.load, "m", uniformLocality{}, tc.states)
			assert.Equal(t, tc.want, rankedNames(ordered))
		})
	}
}

// legacySelectPodForActivation reproduces the single-winner search this split
// replaced. It is kept here so the refactor can be compared against it over
// generated inputs rather than a handful of examples.
func legacySelectPodForActivation(
	candidates []corev1.Pod,
	alreadyOn map[string]bool,
	load map[string]int,
	model string,
	locality LocalityProvider,
	states map[string]PodPlacementState,
) *corev1.Pod {
	if locality == nil {
		locality = uniformLocality{}
	}
	var best *corev1.Pod
	var bestState PodPlacementState
	var bestLoc float64
	var bestLoad int
	for i := range candidates {
		pod := &candidates[i]
		if alreadyOn[pod.Name] {
			continue
		}
		state := states[pod.Name]
		loc := locality.Cost(model, pod.Spec.NodeName)
		l := load[pod.Name]
		if best == nil || placementStateLess(state, bestState) ||
			(!placementStateLess(bestState, state) && rankLess(loc, l, pod.Name, bestLoc, bestLoad, best.Name)) {
			best, bestState, bestLoc, bestLoad = pod, state, loc, l
		}
	}
	return best
}

// deterministicSeq is a small linear congruential generator. The cases below
// are generated rather than listed, but they must be reproducible, so this
// avoids a seeded global source that another test could disturb.
type deterministicSeq struct{ state uint64 }

func (s *deterministicSeq) next(n int) int {
	s.state = s.state*6364136223846793005 + 1442695040888963407
	return int((s.state >> 33) % uint64(n))
}

func (s *deterministicSeq) boolean() bool { return s.next(2) == 0 }

// TestRankCandidatesHeadMatchesTheSingleWinnerSearch is the pin on this
// refactor. The comparison ends in a name comparison and pod names are
// distinct, so it is a strict total order, and under a strict total order the
// minimum equals the head of the sorted list. This asserts that equality over
// many generated pools instead of trusting the argument.
func TestRankCandidatesHeadMatchesTheSingleWinnerSearch(t *testing.T) {
	seq := &deterministicSeq{state: 20260910}
	nodes := []string{"node-a", "node-b", "node-c"}
	locality := fakeLocality{"node-a": 0, "node-b": 1, "node-c": 2}

	for round := 0; round < 1000; round++ {
		count := 1 + seq.next(6)
		names := make([]string, count)
		for i := range names {
			names[i] = fmt.Sprintf("pod-%d", i)
		}
		// Shuffle, so the input order cannot be what makes the two agree.
		for i := count - 1; i > 0; i-- {
			j := seq.next(i + 1)
			names[i], names[j] = names[j], names[i]
		}

		candidates := make([]corev1.Pod, 0, count)
		states := map[string]PodPlacementState{}
		load := map[string]int{}
		alreadyOn := map[string]bool{}
		for _, name := range names {
			pod := namedPod(name)
			pod.Spec.NodeName = nodes[seq.next(len(nodes))]
			candidates = append(candidates, pod)
			load[name] = seq.next(4)
			if seq.next(5) == 0 {
				alreadyOn[name] = true
			}
			// A pod with no entry at all is a real case: the zero value is
			// what an unreachable sidecar leaves behind.
			if seq.next(4) == 0 {
				continue
			}
			states[name] = PodPlacementState{
				ArtifactCached: seq.boolean(),
				SnapshotKnown:  seq.boolean(),
				MemoryKnown:    seq.boolean(),
				HBMFreeBytes:   int64(seq.next(3)) * gibibyte,
				KVUsedBytes:    int64(seq.next(3)) * gibibyte,
				ModelCount:     seq.next(3),
			}
		}

		var localityArg LocalityProvider
		if seq.boolean() {
			localityArg = locality
		}
		want := legacySelectPodForActivation(candidates, alreadyOn, load, "m", localityArg, states)
		ordered := rankCandidates(
			mustFilter(candidates, alreadyOn), load, "m", localityArg, states,
		)

		if want == nil {
			require.Emptyf(t, ordered, "round %d: legacy found nothing but ranking returned pods", round)
			continue
		}
		require.NotEmptyf(t, ordered, "round %d: legacy chose %s but ranking returned nothing", round, want.Name)
		require.Equalf(t, want.Name, ordered[0].Name, "round %d", round)
	}
}

// roomLedger is a readable card with nothing on it, so a test states the room
// it wants directly instead of building instances to consume it.
func roomLedger(room int64) podLedger {
	return podLedger{State: ledgerComplete, HBMUsableBytes: room}
}

func TestFilterCandidatesRefusesACardThatCannotHoldTheModel(t *testing.T) {
	candidates := []corev1.Pod{namedPod("full"), namedPod("roomy")}
	ledgers := map[string]podLedger{
		"full":  roomLedger(35 * gibibyte),
		"roomy": roomLedger(80 * gibibyte),
	}

	feasible, refusals := filterCandidates(candidates, map[string]bool{}, ledgers, 60*gibibyte)

	require.Len(t, feasible, 1)
	assert.Equal(t, "roomy", feasible[0].Name)
	require.Len(t, refusals, 1)
	assert.Equal(t, "full", refusals[0].Pod)
	assert.Equal(t, 35*gibibyte, refusals[0].MaximumRoomBytes)
}

func TestFilterCandidatesAcceptsAnExactFit(t *testing.T) {
	// The refusal is strictly "less than", so a card with exactly enough is
	// placed on rather than turned away by a rounding argument.
	candidates := []corev1.Pod{namedPod("exact")}
	ledgers := map[string]podLedger{"exact": roomLedger(60 * gibibyte)}

	feasible, refusals := filterCandidates(candidates, map[string]bool{}, ledgers, 60*gibibyte)

	require.Len(t, feasible, 1)
	assert.Empty(t, refusals)
}

// TestFilterCandidatesOnlyRefusesWhatItCanProve covers the caution in this
// constraint. Each of these cases was admitted before the check existed and
// has to stay admitted: the check only ever adds refusals.
func TestFilterCandidatesOnlyRefusesWhatItCanProve(t *testing.T) {
	cases := []struct {
		name    string
		ledger  podLedger
		reserve int64
	}{
		{
			name:    "a pod Kubernetes gave no GPU is not judged on GPU memory",
			ledger:  podLedger{State: ledgerNoGPU},
			reserve: 600 * gibibyte,
		},
		{
			name:    "a silent sidecar is a different constraint, not this one",
			ledger:  podLedger{State: ledgerUnread},
			reserve: 600 * gibibyte,
		},
		{
			name:    "a card whose size could not be read is not proof of anything",
			ledger:  podLedger{State: ledgerNoCardSize},
			reserve: 600 * gibibyte,
		},
		{
			name:    "a pod with no ledger entry at all",
			reserve: 600 * gibibyte,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			candidates := []corev1.Pod{namedPod("pod")}
			ledgers := map[string]podLedger{}
			if tc.name != "a pod with no ledger entry at all" {
				ledgers["pod"] = tc.ledger
			}

			feasible, refusals := filterCandidates(candidates, map[string]bool{}, ledgers, tc.reserve)

			assert.Len(t, feasible, 1, "the pod must still be a candidate")
			assert.Empty(t, refusals)
		})
	}
}

func TestFilterCandidatesReportsAnOvercommittedCard(t *testing.T) {
	// A card already promised more than it has reports negative room. That is
	// information, not an error, and it must reach the operator intact.
	candidates := []corev1.Pod{namedPod("oversold")}
	ledgers := map[string]podLedger{"oversold": {
		State:          ledgerComplete,
		HBMUsableBytes: 10 * gibibyte,
		Instances: []ledgerInstance{{
			Claim:                 types.NamespacedName{Namespace: testNamespace, Name: "big"},
			MaximumFootprintBytes: 20 * gibibyte,
			KVFloorBytes:          4 * gibibyte,
		}},
	}}

	feasible, refusals := filterCandidates(candidates, map[string]bool{}, ledgers, 1*gibibyte)

	assert.Empty(t, feasible)
	require.Len(t, refusals, 1)
	assert.Equal(t, -14*gibibyte, refusals[0].MaximumRoomBytes)
}

func TestFilterCandidatesStillDropsPodsAlreadyHostingTheModel(t *testing.T) {
	// Being already host is not a refusal: it asks nothing of an operator, so
	// it must not appear among the reasons the claim reports.
	candidates := []corev1.Pod{namedPod("hosting"), namedPod("full")}
	ledgers := map[string]podLedger{
		"hosting": roomLedger(80 * gibibyte),
		"full":    roomLedger(1 * gibibyte),
	}

	feasible, refusals := filterCandidates(
		candidates, map[string]bool{"hosting": true}, ledgers, 60*gibibyte)

	assert.Empty(t, feasible)
	require.Len(t, refusals, 1)
	assert.Equal(t, "full", refusals[0].Pod)
}

func TestSummarizeRefusals(t *testing.T) {
	message := summarizeRefusals([]podRefusal{
		{Pod: "a", MaximumRoomBytes: 1 * gibibyte},
		{Pod: "b", MaximumRoomBytes: 35*gibibyte + gibibyte/3},
		{Pod: "c", MaximumRoomBytes: 20 * gibibyte},
	}, 60*gibibyte)

	assert.Contains(t, message, "needs 60.0 GiB per GPU")
	assert.Contains(t, message, "3 candidate pod(s)")
	assert.Contains(t, message, "at most 35.3 GiB", "the roomiest refusal is the useful one")
	assert.NotContains(t, message, "20.0 GiB", "listing every pod would grow with the pool")
}

func TestGibibytes(t *testing.T) {
	assert.Equal(t, "0.0 GiB", gibibytes(0))
	assert.Equal(t, "1.0 GiB", gibibytes(gibibyte))
	assert.Equal(t, "35.3 GiB", gibibytes(37932236800))
	assert.Equal(t, "-14.0 GiB", gibibytes(-14*gibibyte))
}

func TestProvablyTooFullExemptsAPodWithNoGPUBeforeSizingItsCard(t *testing.T) {
	// The exemption is checked before the card is sized, and that ordering
	// matters even though both paths admit today. A pod with no GPU is exempt
	// because GPU memory does not apply to it; a card of no known size is
	// admitted only because nothing was proved. When the second starts being
	// refused rather than admitted, the first must stay exempt.
	//
	// Constructing a ledger the collector would never produce is the only way
	// to hold the two apart while they still agree.
	_, tooFull := provablyTooFull(podLedger{State: ledgerNoGPU, HBMUsableBytes: 1}, 600*gibibyte)
	assert.False(t, tooFull, "a pod with no GPU is not judged on GPU memory")

	_, tooFull = provablyTooFull(podLedger{State: ledgerComplete, HBMUsableBytes: 1}, 600*gibibyte)
	assert.True(t, tooFull, "the same card without the exemption is provably too full")
}

func TestFilterCandidatesRefusesAFullCardButNotAnUnknownOne(t *testing.T) {
	// Zero room and unknown room both arrive as the number zero. They must not
	// be treated alike: a card promised exactly all of itself is provably too
	// small for anything, while a card nobody could read proves nothing.
	candidates := []corev1.Pod{namedPod("full"), namedPod("unread")}
	ledgers := map[string]podLedger{
		"full": {
			State:          ledgerComplete,
			HBMUsableBytes: 24 * gibibyte,
			Instances: []ledgerInstance{{
				Claim:                 types.NamespacedName{Namespace: testNamespace, Name: "a"},
				MaximumFootprintBytes: 20 * gibibyte,
				KVFloorBytes:          4 * gibibyte,
			}},
		},
		"unread": {State: ledgerUnread},
	}

	feasible, refusals := filterCandidates(candidates, map[string]bool{}, ledgers, gibibyte)

	require.Len(t, feasible, 1)
	assert.Equal(t, "unread", feasible[0].Name)
	require.Len(t, refusals, 1)
	assert.Equal(t, "full", refusals[0].Pod)
	assert.Equal(t, int64(0), refusals[0].MaximumRoomBytes)
}
