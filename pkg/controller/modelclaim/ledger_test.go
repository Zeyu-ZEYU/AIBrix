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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	modelv1alpha1 "github.com/vllm-project/aibrix/api/model/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

const (
	gibibyte          int64 = 1 << 30
	testHBMTotalBytes int64 = 80 << 30
)

// testUsableBytes is what a test card reports as usable: the driver's cut is
// measured by the runtime, so a test states the figure rather than deriving it.
const testUsableBytes int64 = testHBMTotalBytes - (256 << 20)

func TestClaimMinimumReserveBytes(t *testing.T) {
	// spec.perGPU is required and both fields are validated positive, so this
	// is addition and nothing else. A claim that reaches a controller has the
	// numbers; nothing here re-checks what the API server already refused.
	cases := []struct {
		name      string
		footprint int64
		floor     int64
		want      int64
	}{
		{name: "the two declared numbers", footprint: 4 * gibibyte, floor: 2 * gibibyte, want: 6 * gibibyte},
		{name: "the smallest the CRD allows", footprint: 1, floor: 1, want: 2},
		{name: "a large model", footprint: 640 * gibibyte, floor: 80 * gibibyte, want: 720 * gibibyte},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			claim := &modelv1alpha1.ModelClaim{Spec: modelv1alpha1.ModelClaimSpec{
				PerGPU: modelv1alpha1.ModelClaimPerGPU{
					MaximumFootprintBytes: tc.footprint,
					KVFloorBytes:          tc.floor,
				},
			}}
			assert.Equal(t, tc.want, claimMinimumReserveBytes(claim))
		})
	}
}

func TestMaximumRoomBytes(t *testing.T) {
	line := func(name string, footprint, floor int64) ledgerInstance {
		return ledgerInstance{
			Claim:                 types.NamespacedName{Namespace: testNamespace, Name: name},
			MaximumFootprintBytes: footprint,
			KVFloorBytes:          floor,
		}
	}
	cases := []struct {
		name      string
		ledger    podLedger
		want      int64
		wantKnown bool
	}{
		{
			name:      "an empty card offers all of itself",
			ledger:    podLedger{State: ledgerComplete, HBMUsableBytes: 80 * gibibyte},
			want:      80 * gibibyte,
			wantKnown: true,
		},
		{
			name: "each instance costs its footprint plus its floor",
			ledger: podLedger{
				State:          ledgerComplete,
				HBMUsableBytes: 80 * gibibyte,
				Instances:      []ledgerInstance{line("a", 20*gibibyte, 4*gibibyte)},
			},
			want:      56 * gibibyte,
			wantKnown: true,
		},
		{
			name: "instances accumulate",
			ledger: podLedger{
				State:          ledgerComplete,
				HBMUsableBytes: 80 * gibibyte,
				Instances: []ledgerInstance{
					line("a", 20*gibibyte, 4*gibibyte),
					line("b", 30*gibibyte, 6*gibibyte),
				},
			},
			want:      20 * gibibyte,
			wantKnown: true,
		},
		{
			name: "a card promised exactly all of itself has zero room, and knows it",
			ledger: podLedger{
				State:          ledgerComplete,
				HBMUsableBytes: 24 * gibibyte,
				Instances:      []ledgerInstance{line("a", 20*gibibyte, 4*gibibyte)},
			},
			want:      0,
			wantKnown: true,
		},
		{
			name: "an overcommitted card reports negative room rather than zero",
			ledger: podLedger{
				State:          ledgerComplete,
				HBMUsableBytes: 10 * gibibyte,
				Instances:      []ledgerInstance{line("a", 20*gibibyte, 4*gibibyte)},
			},
			want:      -14 * gibibyte,
			wantKnown: true,
		},
		{
			name:   "the zero value knows nothing, it does not report a full card",
			ledger: podLedger{},
		},
		{
			name:   "a pod with no GPU has no answer either",
			ledger: podLedger{State: ledgerNoGPU},
		},
		{
			name:   "no snapshot means no answer",
			ledger: podLedger{State: ledgerUnknown},
		},
		{
			name:   "no usable card size means no answer",
			ledger: podLedger{State: ledgerGPUMeasureFailed},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, known := tc.ledger.MaximumRoomBytes()
			assert.Equal(t, tc.wantKnown, known)
			assert.Equal(t, tc.want, got)
		})
	}
}

func TestHBMUsableBytes(t *testing.T) {
	card := func(usable int64) RuntimeAcceleratorSnapshot {
		return RuntimeAcceleratorSnapshot{
			ID:             "GPU",
			HBMTotalBytes:  80 * gibibyte,
			HBMUsableBytes: usable,
		}
	}
	cases := []struct {
		name      string
		snapshot  *RuntimeSnapshot
		want      int64
		wantKnown bool
	}{
		{name: "nil snapshot", want: hbmUsableUnknown},
		{name: "no accelerators", snapshot: &RuntimeSnapshot{}, want: hbmUsableUnknown},
		{
			name:      "one card, as the runtime measured it",
			snapshot:  &RuntimeSnapshot{Accelerators: []RuntimeAcceleratorSnapshot{card(79 * gibibyte)}},
			want:      79 * gibibyte,
			wantKnown: true,
		},
		{
			name: "several cards are described by the tightest",
			snapshot: &RuntimeSnapshot{Accelerators: []RuntimeAcceleratorSnapshot{
				card(79 * gibibyte), card(39 * gibibyte), card(59 * gibibyte),
			}},
			want:      39 * gibibyte,
			wantKnown: true,
		},
		{
			name: "a card the runtime could not measure makes the pod unsizable",
			snapshot: &RuntimeSnapshot{Accelerators: []RuntimeAcceleratorSnapshot{
				card(79 * gibibyte), card(hbmUsableUnknown),
			}},
			want: hbmUsableUnknown,
		},
		{
			name: "so does a runtime too old to report the figure at all",
			snapshot: &RuntimeSnapshot{Accelerators: []RuntimeAcceleratorSnapshot{
				card(79 * gibibyte), card(0),
			}},
			want: hbmUsableUnknown,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, known := hbmUsableBytes(tc.snapshot)
			assert.Equal(t, tc.wantKnown, known)
			assert.Equal(t, tc.want, got)
		})
	}
}

// gpuPods returns warm pods that each hold one card, with distinct IPs and a
// snapshot registered for each so the fake runtime can answer for them
// separately.
func gpuPods(runtime *fakeRuntime, names ...string) []corev1.Pod {
	if runtime.snapshots == nil {
		runtime.snapshots = map[string]*RuntimeSnapshot{}
	}
	pods := make([]corev1.Pod, 0, len(names))
	for i, name := range names {
		pod := warmPodWithGPUs(name, "b300-pool-a", 1)
		pod.Status.PodIP = fmt.Sprintf("10.0.0.%d", i+1)
		runtime.snapshots[pod.Status.PodIP] = &RuntimeSnapshot{
			Accelerators: []RuntimeAcceleratorSnapshot{{
				ID:             "GPU-0",
				HBMTotalBytes:  testHBMTotalBytes,
				HBMFreeBytes:   testHBMTotalBytes,
				HBMUsableBytes: testUsableBytes,
			}},
		}
		pods = append(pods, *pod)
	}
	return pods
}

// ledgerClaim builds a claim already placed on a pod, so a ledger has
// something to account for.
func ledgerClaim(name, pod string, footprint, floor int64) *modelv1alpha1.ModelClaim {
	claim := &modelv1alpha1.ModelClaim{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: testNamespace},
		Spec: modelv1alpha1.ModelClaimSpec{
			PodSelector: &metav1.LabelSelector{},
			ArtifactURL: "huggingface://test/" + name,
		},
		Status: modelv1alpha1.ModelClaimStatus{
			Instances: []modelv1alpha1.ModelClaimInstance{
				{Pod: pod, Port: 8100, Phase: modelv1alpha1.ModelClaimActive},
			},
		},
	}
	claim.Spec.PerGPU = modelv1alpha1.ModelClaimPerGPU{
		MaximumFootprintBytes: footprint,
		KVFloorBytes:          floor,
	}
	return claim
}

func collectLedgers(t *testing.T, r *ModelClaimReconciler, candidates []corev1.Pod) map[string]podLedger {
	t.Helper()
	ctx := context.Background()
	states := r.collectPlacementStates(ctx, candidates, "artifact", 1)
	return r.collectPodLedgers(ctx, testNamespace, candidates, states)
}

func TestCollectPodLedgers(t *testing.T) {
	t.Run("an instance on a candidate pod is charged to its card", func(t *testing.T) {
		resident := ledgerClaim("resident", "warm-1", 20*gibibyte, 4*gibibyte)
		r, runtime := newReconciler(t, resident)
		ledgers := collectLedgers(t, r, gpuPods(runtime, "warm-1", "warm-2"))

		require.Len(t, ledgers["warm-1"].Instances, 1)
		assert.Equal(t, "resident", ledgers["warm-1"].Instances[0].Claim.Name)
		room, known := ledgers["warm-1"].MaximumRoomBytes()
		require.True(t, known)
		assert.Equal(t, testUsableBytes-24*gibibyte, room)

		empty, known := ledgers["warm-2"].MaximumRoomBytes()
		require.True(t, known)
		assert.Equal(t, testUsableBytes, empty)
	})

	t.Run("instances from different claims accumulate on one card", func(t *testing.T) {
		first := ledgerClaim("first", "warm-1", 20*gibibyte, 4*gibibyte)
		second := ledgerClaim("second", "warm-1", 10*gibibyte, 2*gibibyte)
		r, runtime := newReconciler(t, first, second)
		ledgers := collectLedgers(t, r, gpuPods(runtime, "warm-1"))

		require.Len(t, ledgers["warm-1"].Instances, 2)
		room, known := ledgers["warm-1"].MaximumRoomBytes()
		require.True(t, known)
		assert.Equal(t, testUsableBytes-36*gibibyte, room)
	})

	t.Run("a failed instance no longer holds the memory it was charged", func(t *testing.T) {
		dead := ledgerClaim("dead", "warm-1", 20*gibibyte, 4*gibibyte)
		dead.Status.Instances[0].Phase = modelv1alpha1.ModelClaimFailed
		r, runtime := newReconciler(t, dead)
		ledgers := collectLedgers(t, r, gpuPods(runtime, "warm-1"))

		assert.Empty(t, ledgers["warm-1"].Instances)
		room, known := ledgers["warm-1"].MaximumRoomBytes()
		require.True(t, known)
		assert.Equal(t, testUsableBytes, room, "the card is whole again")
	})

	t.Run("an activating instance is charged before its engine is ready", func(t *testing.T) {
		// Placement has already committed the memory; waiting for readiness
		// would let a second claim be placed against the same bytes.
		booting := ledgerClaim("booting", "warm-1", 20*gibibyte, 4*gibibyte)
		booting.Status.Instances[0].Phase = modelv1alpha1.ModelClaimActivating
		r, runtime := newReconciler(t, booting)
		ledgers := collectLedgers(t, r, gpuPods(runtime, "warm-1"))

		require.Len(t, ledgers["warm-1"].Instances, 1)
		room, _ := ledgers["warm-1"].MaximumRoomBytes()
		assert.Equal(t, testUsableBytes-24*gibibyte, room)
	})

	t.Run("an instance on a pod outside the candidates is ignored", func(t *testing.T) {
		elsewhere := ledgerClaim("elsewhere", "other-pod", 20*gibibyte, 4*gibibyte)
		r, runtime := newReconciler(t, elsewhere)
		ledgers := collectLedgers(t, r, gpuPods(runtime, "warm-1"))

		room, known := ledgers["warm-1"].MaximumRoomBytes()
		require.True(t, known)
		assert.Equal(t, testUsableBytes, room)
	})

	t.Run("a pod whose sidecar did not answer is unreadable", func(t *testing.T) {
		r, runtime := newReconciler(t)
		candidates := gpuPods(runtime, "warm-1")
		runtime.nilSnapshots = map[string]bool{candidates[0].Status.PodIP: true}
		ledgers := collectLedgers(t, r, candidates)

		assert.Equal(t, ledgerUnknown, ledgers["warm-1"].State)
		_, known := ledgers["warm-1"].MaximumRoomBytes()
		assert.False(t, known)
	})

	t.Run("a pod with a card whose runtime saw none is refused, not admitted", func(t *testing.T) {
		// NVML missing from the image, or a device never mounted into the
		// container, looks exactly like a CPU-only pod from the snapshot. The
		// card is there and may already be full, so it must not be waved
		// through.
		r, runtime := newReconciler(t)
		candidates := gpuPods(runtime, "warm-1")
		runtime.snapshots[candidates[0].Status.PodIP] = &RuntimeSnapshot{
			Accelerators: []RuntimeAcceleratorSnapshot{},
		}
		ledgers := collectLedgers(t, r, candidates)

		assert.NotEqual(t, ledgerNoGPU, ledgers["warm-1"].State,
			"a pod that holds a card is never mistaken for one without")
		assert.Equal(t, ledgerGPUMeasureFailed, ledgers["warm-1"].State)
	})

	t.Run("a pod Kubernetes gave no GPU is outside the account", func(t *testing.T) {
		r, _ := newReconciler(t)
		candidates := []corev1.Pod{*cpuOnlyWarmPod("warm-1", "b300-pool-a")}
		ledgers := collectLedgers(t, r, candidates)

		assert.Equal(t, ledgerNoGPU, ledgers["warm-1"].State,
			"having no GPU is a state of its own, not a gap in the account")
		assert.False(t, ledgers["warm-1"].chargeable())
	})

	t.Run("an instance on a pod with no GPU is not charged anywhere", func(t *testing.T) {
		resident := ledgerClaim("resident", "warm-1", 20*gibibyte, 4*gibibyte)
		r, _ := newReconciler(t, resident)
		candidates := []corev1.Pod{*cpuOnlyWarmPod("warm-1", "b300-pool-a")}
		ledgers := collectLedgers(t, r, candidates)

		assert.Empty(t, ledgers["warm-1"].Instances)
		assert.Equal(t, ledgerNoGPU, ledgers["warm-1"].State)
	})
}

// TestCollectPodLedgersMarksEveryUnsizedCard is the invariant the sentinel
// exists for: a real size appears only alongside ledgerComplete, and every
// other state carries hbmUsableUnknown. Zero would be a number arithmetic
// accepts, so a reader that skipped the state check would get a believable
// answer; this way it gets an obviously broken one.
func TestCollectPodLedgersMarksEveryUnsizedCard(t *testing.T) {
	r, runtime := newReconciler(t)
	// One call, so the pods get distinct IPs and each can be given its own
	// runtime behaviour.
	withCards := gpuPods(runtime, "sized", "blind", "sightless")
	runtime.nilSnapshots = map[string]bool{withCards[1].Status.PodIP: true}
	runtime.snapshots[withCards[2].Status.PodIP] = &RuntimeSnapshot{
		Accelerators: []RuntimeAcceleratorSnapshot{},
	}
	candidates := append(withCards, *cpuOnlyWarmPod("card-free", "b300-pool-a"))
	ledgers := collectLedgers(t, r, candidates)

	want := map[string]ledgerState{
		"sized":     ledgerComplete,
		"blind":     ledgerUnknown,
		"sightless": ledgerGPUMeasureFailed,
		"card-free": ledgerNoGPU,
	}
	for name, state := range want {
		ledger := ledgers[name]
		require.Equalf(t, state, ledger.State, "pod %s", name)
		if state == ledgerComplete {
			assert.Greaterf(t, ledger.HBMUsableBytes, int64(0),
				"pod %s: a complete ledger carries a real size", name)
			continue
		}
		assert.Equalf(t, hbmUsableUnknown, ledger.HBMUsableBytes,
			"pod %s: an unsized card must not report a number arithmetic accepts", name)
	}
}

func TestLedgerForNeverYieldsTheBareZeroValue(t *testing.T) {
	// A pod missing from the account must come back unknown with the sentinel,
	// not as the zero value whose size is zero. Zero is a number the room
	// arithmetic accepts; the sentinel is not.
	ledgers := map[string]podLedger{
		"tracked": {State: ledgerComplete, HBMUsableBytes: 80 * gibibyte},
	}

	known := ledgerFor(ledgers, "tracked")
	assert.Equal(t, ledgerComplete, known.State)
	assert.Equal(t, 80*gibibyte, known.HBMUsableBytes)

	missing := ledgerFor(ledgers, "never-seen")
	assert.Equal(t, ledgerUnknown, missing.State)
	assert.Equal(t, hbmUsableUnknown, missing.HBMUsableBytes,
		"a bare map lookup would have given zero here")

	_, answerable := missing.MaximumRoomBytes()
	assert.False(t, answerable)
	assert.False(t, missing.chargeable())
}

// TestCollectPodLedgersKeepsTheSizeSentinelInvariant states the rule the
// sentinel exists for, over every state the collector can produce: a size is
// real exactly when the state is complete.
func TestCollectPodLedgersKeepsTheSizeSentinelInvariant(t *testing.T) {
	r, runtime := newReconciler(t)
	withCards := gpuPods(runtime, "sized", "silent", "unmeasured")
	runtime.nilSnapshots = map[string]bool{withCards[1].Status.PodIP: true}
	runtime.snapshots[withCards[2].Status.PodIP] = &RuntimeSnapshot{
		Accelerators: []RuntimeAcceleratorSnapshot{
			{ID: "GPU-0", HBMTotalBytes: testHBMTotalBytes, HBMUsableBytes: hbmUsableUnknown},
		},
	}
	candidates := append(withCards, *cpuOnlyWarmPod("card-free", "b300-pool-a"))
	ledgers := collectLedgers(t, r, candidates)

	want := map[string]ledgerState{
		"sized":      ledgerComplete,
		"silent":     ledgerUnknown,
		"unmeasured": ledgerGPUMeasureFailed,
		"card-free":  ledgerNoGPU,
	}
	for name, state := range want {
		ledger := ledgerFor(ledgers, name)
		require.Equalf(t, state, ledger.State, "pod %s", name)
		if state == ledgerComplete {
			assert.Greaterf(t, ledger.HBMUsableBytes, int64(0), "pod %s", name)
			continue
		}
		assert.Equalf(t, hbmUsableUnknown, ledger.HBMUsableBytes, "pod %s", name)
	}
}
