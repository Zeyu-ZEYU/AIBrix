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

// PodPlacementState is the scheduling-relevant summary of one warm pod. It is
// not persisted in a CRD because runtime sidecars are the authoritative source.
type PodPlacementState struct {
	SnapshotKnown  bool
	ArtifactCached bool
	MemoryKnown    bool
	HBMFreeBytes   int64
	KVUsedBytes    int64
	ModelCount     int
	// HBMUsableBytes is how much of the pod's GPU memory can ever hold an
	// engine: the card's total minus what the driver keeps for itself. Unlike
	// HBMFreeBytes it does not move with traffic, so the ledger can be built
	// on it. HBMUsableKnown separates "the card has no usable memory" from
	// "the snapshot did not say".
	HBMUsableBytes int64
	HBMUsableKnown bool
	// Engines are the engines the snapshot reported, as it reported them. The
	// ledger reads each one's KV figures from here.
	Engines []RuntimeSnapshotModel
}

func placementStateFromSnapshot(snapshot *RuntimeSnapshot, artifactURL string, parallelism int64) PodPlacementState {
	state := PodPlacementState{
		SnapshotKnown: true,
		ModelCount:    len(snapshot.Models),
		Engines:       snapshot.Models,
	}
	for _, cached := range snapshot.CachedArtifacts {
		if cached == artifactURL {
			state.ArtifactCached = true
			break
		}
	}
	if parallelism < 1 {
		parallelism = 1
	}
	if parallelism > 1 && int64(len(snapshot.Accelerators)) != parallelism {
		return state
	}
	for _, accelerator := range snapshot.Accelerators {
		// A single-GPU engine needs the largest available single-device slot. A
		// fixed TP/PP group uses every visible GPU, so its safe headroom is the
		// least-free rank rather than a misleading aggregate or maximum.
		if !state.MemoryKnown || (parallelism == 1 && accelerator.HBMFreeBytes > state.HBMFreeBytes) ||
			(parallelism > 1 && accelerator.HBMFreeBytes < state.HBMFreeBytes) {
			state.HBMFreeBytes = accelerator.HBMFreeBytes
			state.MemoryKnown = true
		}
	}
	state.HBMUsableBytes, state.HBMUsableKnown = hbmUsableBytes(snapshot)
	for _, model := range snapshot.Models {
		// An engine whose segment does not exist yet reports a negative figure.
		// It has built no KV cache, so it adds nothing here.
		if model.KVUsedBytes < 0 {
			continue
		}
		state.KVUsedBytes += model.KVUsedBytes
	}
	return state
}
