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
)

func TestPlacementStateFromSnapshot(t *testing.T) {
	snapshot := &RuntimeSnapshot{
		Accelerators: []RuntimeAcceleratorSnapshot{
			{ID: "GPU-0", HBMFreeBytes: 800},
			{ID: "GPU-1", HBMFreeBytes: 300},
		},
		Models: []RuntimeSnapshotModel{
			{ModelName: "m1", KVUsedBytes: 10},
			{ModelName: "m2", KVUsedBytes: 25},
		},
		CachedArtifacts: []string{"hf://Org/M1"},
	}

	singleGPUState := placementStateFromSnapshot(snapshot, "hf://Org/M1", 1)
	groupState := placementStateFromSnapshot(snapshot, "hf://Org/M1", 2)

	assert.True(t, singleGPUState.SnapshotKnown)
	assert.True(t, singleGPUState.ArtifactCached)
	assert.True(t, singleGPUState.MemoryKnown)
	assert.Equal(t, int64(800), singleGPUState.HBMFreeBytes)
	assert.Equal(t, int64(300), groupState.HBMFreeBytes)
	assert.Equal(t, int64(35), groupState.KVUsedBytes)
	assert.Equal(t, 2, groupState.ModelCount)
}

// TestPlacementReadsEveryDecisionAfresh pins that placement never decides on
// a copy. An engine that went back to kvcached's default between two
// decisions, as it does when it restarts, is seen at the second one.
func TestPlacementReadsEveryDecisionAfresh(t *testing.T) {
	resident := ledgerClaim("resident", "warm-1", 20*gibibyte, 4*gibibyte)
	resident.UID = "resident-uid"
	resident.Status.Instances[0].KVLimitBytes = 4 * gibibyte
	r, runtime := newReconciler(t, resident)
	candidates := gpuPods(runtime, "warm-1")
	stated := runtime.snapshots[candidates[0].Status.PodIP]
	stated.Models = []RuntimeSnapshotModel{engineOf(resident, 4*gibibyte, gibibyte)}

	before, known := collectLedgers(t, r, candidates)["warm-1"].MinimumRoomBytes()
	assert.True(t, known)

	stated.Models[0].KVCapacityBytes = 80 * gibibyte
	after, known := collectLedgers(t, r, candidates)["warm-1"].MinimumRoomBytes()
	assert.True(t, known)

	assert.Equal(t, before-76*gibibyte, after,
		"the second decision must see the limit the engine holds now")
}

func TestPlacementStateFromSnapshotSkipsEnginesWithoutASegment(t *testing.T) {
	snapshot := &RuntimeSnapshot{
		Models: []RuntimeSnapshotModel{
			{ModelName: "serving", KVUsedBytes: 10, KVCapacityBytes: 100},
			{ModelName: "starting", KVUsedBytes: -1, KVCapacityBytes: -1},
		},
	}

	state := placementStateFromSnapshot(snapshot, "", 1)

	assert.Equal(t, int64(10), state.KVUsedBytes)
	assert.Equal(t, 2, state.ModelCount)
}
