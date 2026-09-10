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
	"k8s.io/utils/ptr"
)

const (
	gibibyte          int64 = 1 << 30
	testHBMTotalBytes int64 = 80 << 30
)

// testUsableBytes is what a test card offers once the driver has taken its cut.
var testUsableBytes = testHBMTotalBytes - defaultDriverReserveBytes

func TestClaimMinimumReserveBytes(t *testing.T) {
	cases := []struct {
		name      string
		perGPU    *modelv1alpha1.ModelClaimPerGPU
		want      int64
		wantFound bool
	}{
		{name: "no perGPU block", perGPU: nil},
		{
			name:   "footprint only",
			perGPU: &modelv1alpha1.ModelClaimPerGPU{MaximumFootprintBytes: ptr.To(4 * gibibyte)},
		},
		{
			name:   "floor only",
			perGPU: &modelv1alpha1.ModelClaimPerGPU{KVFloorBytes: ptr.To(2 * gibibyte)},
		},
		{
			name: "zero is not a declaration",
			perGPU: &modelv1alpha1.ModelClaimPerGPU{
				MaximumFootprintBytes: ptr.To(int64(0)),
				KVFloorBytes:          ptr.To(2 * gibibyte),
			},
		},
		{
			name: "both declared",
			perGPU: &modelv1alpha1.ModelClaimPerGPU{
				MaximumFootprintBytes: ptr.To(4 * gibibyte),
				KVFloorBytes:          ptr.To(2 * gibibyte),
			},
			want:      6 * gibibyte,
			wantFound: true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			claim := &modelv1alpha1.ModelClaim{Spec: modelv1alpha1.ModelClaimSpec{PerGPU: tc.perGPU}}
			got, found := claimMinimumReserveBytes(claim)
			assert.Equal(t, tc.wantFound, found)
			assert.Equal(t, tc.want, got)
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
			ledger:    podLedger{HBMUsableBytes: 80 * gibibyte},
			want:      80 * gibibyte,
			wantKnown: true,
		},
		{
			name: "each instance costs its footprint plus its floor",
			ledger: podLedger{
				HBMUsableBytes: 80 * gibibyte,
				Instances:      []ledgerInstance{line("a", 20*gibibyte, 4*gibibyte)},
			},
			want:      56 * gibibyte,
			wantKnown: true,
		},
		{
			name: "instances accumulate",
			ledger: podLedger{
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
			name: "an overcommitted card reports negative room rather than zero",
			ledger: podLedger{
				HBMUsableBytes: 10 * gibibyte,
				Instances:      []ledgerInstance{line("a", 20*gibibyte, 4*gibibyte)},
			},
			want:      -14 * gibibyte,
			wantKnown: true,
		},
		{
			name:   "no snapshot means no answer",
			ledger: podLedger{Missing: missingSnapshot},
		},
		{
			name:   "no usable card size means no answer",
			ledger: podLedger{Missing: missingCardSize},
		},
		{
			name: "an undeclared neighbour means no answer even with a known card",
			ledger: podLedger{
				HBMUsableBytes: 80 * gibibyte,
				Missing:        missingClaimNumbers,
			},
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
	card := func(total int64) RuntimeAcceleratorSnapshot {
		return RuntimeAcceleratorSnapshot{ID: "GPU", HBMTotalBytes: total}
	}
	const reserve = defaultDriverReserveBytes
	cases := []struct {
		name        string
		snapshot    *RuntimeSnapshot
		parallelism int64
		want        int64
		wantKnown   bool
	}{
		{name: "nil snapshot"},
		{name: "no accelerators", snapshot: &RuntimeSnapshot{}},
		{
			name:     "a card smaller than the driver reserve is not usable",
			snapshot: &RuntimeSnapshot{Accelerators: []RuntimeAcceleratorSnapshot{card(1 << 20)}},
		},
		{
			name:      "one card, minus the driver reserve",
			snapshot:  &RuntimeSnapshot{Accelerators: []RuntimeAcceleratorSnapshot{card(80 * gibibyte)}},
			want:      80*gibibyte - reserve,
			wantKnown: true,
		},
		{
			name: "a single-GPU engine takes the largest card",
			snapshot: &RuntimeSnapshot{Accelerators: []RuntimeAcceleratorSnapshot{
				card(40 * gibibyte), card(80 * gibibyte),
			}},
			want:      80*gibibyte - reserve,
			wantKnown: true,
		},
		{
			name: "a parallel engine is limited by the smallest card it spans",
			snapshot: &RuntimeSnapshot{Accelerators: []RuntimeAcceleratorSnapshot{
				card(40 * gibibyte), card(80 * gibibyte),
			}},
			parallelism: 2,
			want:        40*gibibyte - reserve,
			wantKnown:   true,
		},
		{
			name: "a parallel engine that does not match the visible cards is unknown",
			snapshot: &RuntimeSnapshot{Accelerators: []RuntimeAcceleratorSnapshot{
				card(40 * gibibyte), card(80 * gibibyte),
			}},
			parallelism: 4,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, known := hbmUsableBytes(tc.snapshot, tc.parallelism, reserve)
			assert.Equal(t, tc.wantKnown, known)
			assert.Equal(t, tc.want, got)
		})
	}
}

func TestHBMUsableBytesTakesTheReserveFromItsCaller(t *testing.T) {
	// The driver's cut belongs to the hardware, not to this package, so a pool
	// on different cards can be given a different figure without touching the
	// arithmetic.
	snapshot := &RuntimeSnapshot{Accelerators: []RuntimeAcceleratorSnapshot{
		{ID: "GPU", HBMTotalBytes: 80 * gibibyte},
	}}
	got, known := hbmUsableBytes(snapshot, 1, gibibyte)
	require.True(t, known)
	assert.Equal(t, 79*gibibyte, got)
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
			Accelerators: []RuntimeAcceleratorSnapshot{
				{ID: "GPU-0", HBMTotalBytes: testHBMTotalBytes, HBMFreeBytes: testHBMTotalBytes},
			},
		}
		pods = append(pods, *pod)
	}
	return pods
}

// ledgerClaim builds a claim already placed on a pod, so a ledger has
// something to account for. A zero footprint or floor means the claim declared
// nothing, which is how a pre-existing claim looks.
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
	if footprint > 0 && floor > 0 {
		claim.Spec.PerGPU = &modelv1alpha1.ModelClaimPerGPU{
			MaximumFootprintBytes: ptr.To(footprint),
			KVFloorBytes:          ptr.To(floor),
		}
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

	t.Run("a failed instance whose claim declared nothing does not blind the card", func(t *testing.T) {
		// A claim from before spec.perGPU existed, whose engine then died,
		// must not make its card unusable for everyone else forever.
		legacy := ledgerClaim("legacy", "warm-1", 0, 0)
		legacy.Status.Instances[0].Phase = modelv1alpha1.ModelClaimFailed
		r, runtime := newReconciler(t, legacy)
		ledgers := collectLedgers(t, r, gpuPods(runtime, "warm-1"))

		assert.Equal(t, missingNothing, ledgers["warm-1"].Missing)
		room, known := ledgers["warm-1"].MaximumRoomBytes()
		require.True(t, known)
		assert.Equal(t, testUsableBytes, room)
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

	t.Run("an undeclared neighbour makes the card unreadable and names itself", func(t *testing.T) {
		silent := ledgerClaim("silent", "warm-1", 0, 0)
		r, runtime := newReconciler(t, silent)
		ledgers := collectLedgers(t, r, gpuPods(runtime, "warm-1"))

		assert.Equal(t, missingClaimNumbers, ledgers["warm-1"].Missing)
		assert.Equal(t, "silent", ledgers["warm-1"].UndeclaredClaim.Name)
		_, known := ledgers["warm-1"].MaximumRoomBytes()
		assert.False(t, known)
	})

	t.Run("a pod whose sidecar did not answer is unreadable", func(t *testing.T) {
		r, runtime := newReconciler(t)
		candidates := gpuPods(runtime, "warm-1")
		runtime.nilSnapshots = map[string]bool{candidates[0].Status.PodIP: true}
		ledgers := collectLedgers(t, r, candidates)

		assert.Equal(t, missingSnapshot, ledgers["warm-1"].Missing)
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

		assert.False(t, ledgers["warm-1"].NoGPU)
		assert.Equal(t, missingCardSize, ledgers["warm-1"].Missing)
	})

	t.Run("a pod Kubernetes gave no GPU is outside the account", func(t *testing.T) {
		r, _ := newReconciler(t)
		candidates := []corev1.Pod{*warmPod("warm-1", "b300-pool-a", true, corev1.PodRunning)}
		ledgers := collectLedgers(t, r, candidates)

		assert.True(t, ledgers["warm-1"].NoGPU)
		assert.Equal(t, missingNothing, ledgers["warm-1"].Missing,
			"having no GPU is not a gap in the account")
		assert.False(t, ledgers["warm-1"].accountable())
	})

	t.Run("an instance on a pod with no GPU is not charged anywhere", func(t *testing.T) {
		resident := ledgerClaim("resident", "warm-1", 20*gibibyte, 4*gibibyte)
		r, _ := newReconciler(t, resident)
		candidates := []corev1.Pod{*warmPod("warm-1", "b300-pool-a", true, corev1.PodRunning)}
		ledgers := collectLedgers(t, r, candidates)

		assert.Empty(t, ledgers["warm-1"].Instances)
		assert.True(t, ledgers["warm-1"].NoGPU)
	})
}
