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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	modelv1alpha1 "github.com/vllm-project/aibrix/api/model/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
)

const gibibyte int64 = 1 << 30

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
			name:      "empty card offers all of itself",
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
			name: "an overcommitted card reports a negative room rather than zero",
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
			name:      "a card smaller than the driver reserve is not usable",
			snapshot:  &RuntimeSnapshot{Accelerators: []RuntimeAcceleratorSnapshot{card(1 << 20)}},
			wantKnown: false,
		},
		{
			name:      "one card, minus the driver reserve",
			snapshot:  &RuntimeSnapshot{Accelerators: []RuntimeAcceleratorSnapshot{card(80 * gibibyte)}},
			want:      80*gibibyte - driverReserveBytes,
			wantKnown: true,
		},
		{
			name: "a single-GPU engine takes the largest card",
			snapshot: &RuntimeSnapshot{Accelerators: []RuntimeAcceleratorSnapshot{
				card(40 * gibibyte), card(80 * gibibyte),
			}},
			want:      80*gibibyte - driverReserveBytes,
			wantKnown: true,
		},
		{
			name: "a parallel engine is limited by the smallest card it spans",
			snapshot: &RuntimeSnapshot{Accelerators: []RuntimeAcceleratorSnapshot{
				card(40 * gibibyte), card(80 * gibibyte),
			}},
			parallelism: 2,
			want:        40*gibibyte - driverReserveBytes,
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
			got, known := hbmUsableBytes(tc.snapshot, tc.parallelism)
			assert.Equal(t, tc.wantKnown, known)
			assert.Equal(t, tc.want, got)
		})
	}
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
			Instances: []modelv1alpha1.ModelClaimInstance{{Pod: pod, Port: 8100}},
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

func TestCollectPodLedgers(t *testing.T) {
	pods := func(names ...string) []corev1.Pod {
		out := make([]corev1.Pod, 0, len(names))
		for i, name := range names {
			pod := warmPod(name, "b300-pool-a", true, corev1.PodRunning)
			pod.Status.PodIP = "10.0.0." + string(rune('1'+i))
			out = append(out, *pod)
		}
		return out
	}
	usable := testHBMTotalBytes - driverReserveBytes

	t.Run("an instance on a candidate pod is charged to its card", func(t *testing.T) {
		resident := ledgerClaim("resident", "warm-1", 20*gibibyte, 4*gibibyte)
		r, _ := newReconciler(t, resident)
		candidates := pods("warm-1", "warm-2")
		states := r.collectPlacementStates(context.Background(), candidates, "artifact", 1)
		ledgers := r.collectPodLedgers(context.Background(), testNamespace, candidates, states)

		require.Len(t, ledgers["warm-1"].Instances, 1)
		room, known := ledgers["warm-1"].MaximumRoomBytes()
		require.True(t, known)
		assert.Equal(t, usable-24*gibibyte, room)

		empty, known := ledgers["warm-2"].MaximumRoomBytes()
		require.True(t, known)
		assert.Equal(t, usable, empty)
	})

	t.Run("an instance on a pod outside the candidates is ignored", func(t *testing.T) {
		elsewhere := ledgerClaim("elsewhere", "other-pod", 20*gibibyte, 4*gibibyte)
		r, _ := newReconciler(t, elsewhere)
		candidates := pods("warm-1")
		states := r.collectPlacementStates(context.Background(), candidates, "artifact", 1)
		ledgers := r.collectPodLedgers(context.Background(), testNamespace, candidates, states)

		room, known := ledgers["warm-1"].MaximumRoomBytes()
		require.True(t, known)
		assert.Equal(t, usable, room)
	})

	t.Run("an undeclared neighbour makes the card unreadable and names itself", func(t *testing.T) {
		silent := ledgerClaim("silent", "warm-1", 0, 0)
		r, _ := newReconciler(t, silent)
		candidates := pods("warm-1")
		states := r.collectPlacementStates(context.Background(), candidates, "artifact", 1)
		ledgers := r.collectPodLedgers(context.Background(), testNamespace, candidates, states)

		assert.Equal(t, missingClaimNumbers, ledgers["warm-1"].Missing)
		assert.Equal(t, "silent", ledgers["warm-1"].UndeclaredClaim.Name)
		_, known := ledgers["warm-1"].MaximumRoomBytes()
		assert.False(t, known)
	})

	t.Run("a pod whose sidecar did not answer is unreadable", func(t *testing.T) {
		r, runtime := newReconciler(t)
		candidates := pods("warm-1")
		runtime.nilSnapshots = map[string]bool{candidates[0].Status.PodIP: true}
		states := r.collectPlacementStates(context.Background(), candidates, "artifact", 1)
		ledgers := r.collectPodLedgers(context.Background(), testNamespace, candidates, states)

		assert.Equal(t, missingSnapshot, ledgers["warm-1"].Missing)
		_, known := ledgers["warm-1"].MaximumRoomBytes()
		assert.False(t, known)
	})
}

func TestLedgerFingerprintTracksTheAccount(t *testing.T) {
	base := map[string]podLedger{
		"warm-1": {HBMUsableBytes: 80 * gibibyte},
		"warm-2": {HBMUsableBytes: 80 * gibibyte},
	}
	same := map[string]podLedger{
		"warm-2": {HBMUsableBytes: 80 * gibibyte},
		"warm-1": {HBMUsableBytes: 80 * gibibyte},
	}
	assert.Equal(t, ledgerFingerprint(base), ledgerFingerprint(same),
		"map iteration order must not change the fingerprint")

	moved := map[string]podLedger{
		"warm-1": {HBMUsableBytes: 80 * gibibyte},
		"warm-2": {
			HBMUsableBytes: 80 * gibibyte,
			Instances: []ledgerInstance{{
				Claim:                 types.NamespacedName{Namespace: testNamespace, Name: "new"},
				MaximumFootprintBytes: 20 * gibibyte,
				KVFloorBytes:          4 * gibibyte,
			}},
		},
	}
	assert.NotEqual(t, ledgerFingerprint(base), ledgerFingerprint(moved),
		"an instance arriving must change the fingerprint")

	// One card gaining exactly what another lost still has to register, which
	// is why the fingerprint hashes each pod rather than summing rooms.
	shifted := map[string]podLedger{
		"warm-1": {HBMUsableBytes: 60 * gibibyte},
		"warm-2": {HBMUsableBytes: 100 * gibibyte},
	}
	assert.NotEqual(t, ledgerFingerprint(base), ledgerFingerprint(shifted))

	fewer := map[string]podLedger{"warm-1": {HBMUsableBytes: 80 * gibibyte}}
	assert.NotEqual(t, ledgerFingerprint(base), ledgerFingerprint(fewer),
		"a pod leaving the pool must change the fingerprint")

	unreadable := map[string]podLedger{
		"warm-1": {HBMUsableBytes: 80 * gibibyte},
		"warm-2": {Missing: missingSnapshot},
	}
	assert.NotEqual(t, ledgerFingerprint(base), ledgerFingerprint(unreadable),
		"losing sight of a pod must change the fingerprint")
}
