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
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
)

// bigClaim declares a model that needs more than half a test card, so two of
// them fill a two-pod pool and a third has nowhere to go.
func bigClaim(name string) *modelv1alpha1.ModelClaim {
	claim := sampleModelClaim()
	claim.Name = name
	claim.Spec.ModelName = ptr.To(name)
	claim.Spec.PerGPU = modelv1alpha1.ModelClaimPerGPU{
		MaximumFootprintBytes: 40 * gibibyte,
		KVFloorBytes:          4 * gibibyte,
	}
	return withFinalizer(claim)
}

func twoWarmPods() (*corev1.Pod, *corev1.Pod) {
	first := warmPod("warm-1", "b300-pool-a", true, corev1.PodRunning)
	first.Status.PodIP = "10.0.0.1"
	second := warmPod("warm-2", "b300-pool-a", true, corev1.PodRunning)
	second.Status.PodIP = "10.0.0.2"
	return first, second
}

func scheduledCondition(t *testing.T, claim *modelv1alpha1.ModelClaim) *metav1.Condition {
	t.Helper()
	return meta.FindStatusCondition(claim.Status.Conditions,
		string(modelv1alpha1.ModelClaimConditionTypeScheduled))
}

// TestPlacementFillsBothCardsThenRefuses is the whole constraint seen from
// outside: two models that each need more than half a card land on separate
// cards, and a third is told, with numbers, that neither card could ever hold
// it.
func TestPlacementFillsBothCardsThenRefuses(t *testing.T) {
	first, second, third := bigClaim("first"), bigClaim("second"), bigClaim("third")
	warm1, warm2 := twoWarmPods()
	r, runtime := newReconciler(t, first, second, third, warm1, warm2)

	reconcileOnce(t, r, "first")
	reconcileOnce(t, r, "second")

	firstPod := getModel(t, r, "first").Status.Instances[0].Pod
	secondPod := getModel(t, r, "second").Status.Instances[0].Pod
	assert.NotEqual(t, firstPod, secondPod,
		"two models that each need more than half a card cannot share one")
	require.Len(t, runtime.activateCalls, 2)

	reconcileOnce(t, r, "third")

	waiting := getModel(t, r, "third")
	assert.Empty(t, waiting.Status.Instances)
	assert.Len(t, runtime.activateCalls, 2, "a refused claim must not start an engine")
	condition := scheduledCondition(t, waiting)
	require.NotNil(t, condition)
	assert.Equal(t, metav1.ConditionFalse, condition.Status)
	assert.Equal(t, reasonInsufficientCapacity, condition.Reason)
	assert.Contains(t, condition.Message, "needs 44.0 GiB per GPU")
	assert.Contains(t, condition.Message, "2 candidate pod(s)")
}

// TestPlacementFreesUpWhenAModelLeaves closes the loop: the refusal is about
// the ledger, so removing a line from it makes the same claim placeable with
// no other change.
func TestPlacementFreesUpWhenAModelLeaves(t *testing.T) {
	ctx := context.Background()
	first, third := bigClaim("first"), bigClaim("third")
	warm1 := warmPod("warm-1", "b300-pool-a", true, corev1.PodRunning)
	r, _ := newReconciler(t, first, third, warm1)

	reconcileOnce(t, r, "first")
	require.Len(t, getModel(t, r, "first").Status.Instances, 1)

	reconcileOnce(t, r, "third")
	require.Empty(t, getModel(t, r, "third").Status.Instances)

	require.NoError(t, r.Delete(ctx, getModel(t, r, "first")))
	reconcileOnce(t, r, "first")

	reconcileOnce(t, r, "third")
	placed := getModel(t, r, "third")
	require.Len(t, placed.Status.Instances, 1)
	assert.Equal(t, "warm-1", placed.Status.Instances[0].Pod)
}

// TestPlacementDistinguishesNoRoomFromNoPods keeps the two situations apart in
// what the operator sees, because they call for different actions.
func TestPlacementDistinguishesNoRoomFromNoPods(t *testing.T) {
	claim := bigClaim("lonely")
	r, _ := newReconciler(t, claim)

	reconcileOnce(t, r, "lonely")

	condition := scheduledCondition(t, getModel(t, r, "lonely"))
	require.NotNil(t, condition)
	assert.Equal(t, "NoMatchingPods", condition.Reason,
		"a claim that matched nothing is not a capacity problem")
}

// TestPlacementIgnoresMemoryOnACardFreePod pins the one admission this check
// still makes on purpose, for the mock and CPU-only pools the e2e suite
// deploys: a pod with no GPU is not judged on GPU memory.
func TestPlacementIgnoresMemoryOnACardFreePod(t *testing.T) {
	claim := bigClaim("huge")
	pool := cpuOnlyWarmPod("warm-1", "b300-pool-a")
	r, runtime := newReconciler(t, claim, pool)

	reconcileOnce(t, r, "huge")

	require.Len(t, getModel(t, r, "huge").Status.Instances, 1)
	assert.Len(t, runtime.activateCalls, 1)
}

// TestPlacementWalksPastItsFavouriteCard is the property the single-winner
// search could not have had: ranking prefers the pod with the cached artifact,
// that pod has no room, and the round still succeeds on a later one.
func TestPlacementWalksPastItsFavouriteCard(t *testing.T) {
	arriving := bigClaim("arriving")
	resident := ledgerClaim("resident", "warm-1", 40*gibibyte, 4*gibibyte)
	warm1, warm2 := twoWarmPods()
	r, runtime := newReconciler(t, resident, arriving, warm1, warm2)
	runtime.snapshots = map[string]*RuntimeSnapshot{
		warm1.Status.PodIP: {
			Accelerators: []RuntimeAcceleratorSnapshot{{
				ID: "GPU-0", HBMTotalBytes: testHBMTotalBytes,
				HBMFreeBytes: testHBMTotalBytes, HBMUsableBytes: testUsableBytes,
			}},
			CachedArtifacts: []string{arriving.Spec.ArtifactURL},
		},
	}

	states := r.collectPlacementStates(
		context.Background(), []corev1.Pod{*warm1, *warm2}, arriving.Spec.ArtifactURL, 1)
	ordered := rankCandidates(
		mustFilter([]corev1.Pod{*warm1, *warm2}, map[string]bool{}),
		map[string]int{}, "arriving", uniformLocality{}, states,
	)
	require.Equal(t, "warm-1", ordered[0].Name,
		"the cached artifact makes the occupied card the favourite")

	reconcileOnce(t, r, "arriving")

	placed := getModel(t, r, "arriving")
	require.Len(t, placed.Status.Instances, 1)
	assert.Equal(t, "warm-2", placed.Status.Instances[0].Pod,
		"placement passes over its favourite card because that card has no room")
	require.Len(t, runtime.activateCalls, 1)
}

// TestPlacementRecordsTheInstanceBeforeStartingTheEngine pins the order these
// two steps happen in. The card is spent the moment the engine starts, and the
// instance is the only record of that, so it has to be durable first. Were the
// order reversed and the write then failed, an engine would be running with
// nothing naming it and the next reconcile would start a second one.
func TestPlacementRecordsTheInstanceBeforeStartingTheEngine(t *testing.T) {
	claim := bigClaim("early")
	warm1, _ := twoWarmPods()
	r, runtime := newReconciler(t, claim, warm1)

	var recordedWhenEngineStarted []modelv1alpha1.ModelClaimInstance
	runtime.onActivate = func(*ActivateRequest) {
		recordedWhenEngineStarted = getModel(t, r, "early").Status.Instances
	}

	reconcileOnce(t, r, "early")

	require.Len(t, recordedWhenEngineStarted, 1,
		"the API server must already hold the instance when the engine starts")
	assert.Equal(t, "warm-1", recordedWhenEngineStarted[0].Pod)
	assert.Equal(t, modelv1alpha1.ModelClaimActivating, recordedWhenEngineStarted[0].Phase)
	assert.Equal(t, int32(0), recordedWhenEngineStarted[0].Port,
		"the port is not known yet, and zero keeps the model unroutable")
	assert.Equal(t, claim.Spec.PerGPU.KVFloorBytes, recordedWhenEngineStarted[0].KVLimitBytes,
		"the instance is recorded at its KV floor, before the engine starts")

	placed := getModel(t, r, "early").Status.Instances
	require.Len(t, placed, 1)
	assert.NotZero(t, placed[0].Port, "the real port lands once the engine is up")
}

// TestPlacementReleasesTheCardWhenTheEngineFailsToStart is the other half:
// the reservation is undone on a failure we can see. Left in place it would
// hold the card for good, since nothing can tell an engine that never started
// from one still booting.
func TestPlacementReleasesTheCardWhenTheEngineFailsToStart(t *testing.T) {
	claim := bigClaim("doomed")
	warm1, _ := twoWarmPods()
	r, runtime := newReconciler(t, claim, warm1)
	runtime.failActivate = true

	_, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Namespace: testNamespace, Name: "doomed"},
	})
	require.NoError(t, err, "an activation failure is reported on the claim, not returned")

	got := getModel(t, r, "doomed")
	assert.Empty(t, got.Status.Instances, "the card must go back to the pool")
	assert.Equal(t, modelv1alpha1.ModelClaimFailed, got.Status.Phase)
}

// claimNeeding declares a model with the given per-GPU cost.
func claimNeeding(name string, footprint, floor int64) *modelv1alpha1.ModelClaim {
	claim := bigClaim(name)
	claim.Spec.PerGPU = modelv1alpha1.ModelClaimPerGPU{
		MaximumFootprintBytes: footprint,
		KVFloorBytes:          floor,
	}
	return claim
}

// TestPlacementWaitsWhileAnEngineOnTheCardIsStarting covers two models that
// fit one card by their floors. They still cannot start side by side: nothing
// can be read of the first engine until it is ready, and until its limit is
// written it runs under kvcached's default. The second waits, says why, and
// lands as soon as the first engine is held to its limit.
func TestPlacementWaitsWhileAnEngineOnTheCardIsStarting(t *testing.T) {
	first := claimNeeding("first", 20*gibibyte, 4*gibibyte)
	second := claimNeeding("second", 20*gibibyte, 4*gibibyte)
	warm1 := warmPod("warm-1", "b300-pool-a", true, corev1.PodRunning)
	r, runtime := newReconciler(t, first, second, warm1)
	runtime.notReady = true

	reconcileOnce(t, r, "first")
	require.Len(t, getModel(t, r, "first").Status.Instances, 1)

	reconcileOnce(t, r, "second")

	waiting := getModel(t, r, "second")
	assert.Empty(t, waiting.Status.Instances)
	condition := scheduledCondition(t, waiting)
	require.NotNil(t, condition)
	assert.Equal(t, reasonWaitingForRoom, condition.Reason)
	assert.Contains(t, condition.Message, "the engine of first cannot be read yet")

	// Once it is ready, the first engine's limit is written, and that is
	// all the second needs.
	runtime.notReady = false
	reconcileOnce(t, r, "first")
	reconcileOnce(t, r, "second")

	placed := getModel(t, r, "second")
	require.Len(t, placed.Status.Instances, 1)
	assert.Equal(t, "warm-1", placed.Status.Instances[0].Pod)
}

// TestPlacementWaitsForAnEngineUnderKVCachedsDefault covers a resident engine
// whose segment still allows far more than its floor. It could grow into the
// room a newcomer needs, so the newcomer waits, with the figure. Once the
// resident is held to its limit the room is there.
func TestPlacementWaitsForAnEngineUnderKVCachedsDefault(t *testing.T) {
	resident := ledgerClaim("resident", "warm-1", 20*gibibyte, 4*gibibyte)
	resident.UID = "resident-uid"
	arriving := claimNeeding("arriving", 20*gibibyte, 4*gibibyte)
	warm1 := warmPod("warm-1", "b300-pool-a", true, corev1.PodRunning)
	r, runtime := newReconciler(t, resident, arriving, warm1)
	runtime.snapshots = map[string]*RuntimeSnapshot{
		warm1.Status.PodIP: {
			Accelerators: []RuntimeAcceleratorSnapshot{{
				ID: "GPU-0", HBMTotalBytes: testHBMTotalBytes, HBMUsableBytes: testUsableBytes,
			}},
			Models: []RuntimeSnapshotModel{engineOf(resident, 70*gibibyte, gibibyte)},
		},
	}

	reconcileOnce(t, r, "arriving")

	waiting := getModel(t, r, "arriving")
	assert.Empty(t, waiting.Status.Instances)
	condition := scheduledCondition(t, waiting)
	require.NotNil(t, condition)
	assert.Equal(t, reasonWaitingForRoom, condition.Reason)
	assert.Contains(t, condition.Message, "the roomiest pod that could hold it, warm-1, can vouch for only")

	runtime.snapshots[warm1.Status.PodIP].Models[0].KVCapacityBytes = 4 * gibibyte
	reconcileOnce(t, r, "arriving")

	require.Len(t, getModel(t, r, "arriving").Status.Instances, 1)
}

// TestPlacementWaitsRatherThanGivesUp covers a pool where one card could hold
// the model once it can show room and another never could. The claim is
// waiting, not refused, and the refused card is still counted.
func TestPlacementWaitsRatherThanGivesUp(t *testing.T) {
	full := ledgerClaim("full", "warm-1", 70*gibibyte, 4*gibibyte)
	starting := ledgerClaim("starting", "warm-2", 20*gibibyte, 4*gibibyte)
	starting.Status.Instances[0].Phase = modelv1alpha1.ModelClaimActivating
	arriving := claimNeeding("arriving", 20*gibibyte, 4*gibibyte)
	warm1, warm2 := twoWarmPods()
	r, _ := newReconciler(t, full, starting, arriving, warm1, warm2)

	reconcileOnce(t, r, "arriving")

	condition := scheduledCondition(t, getModel(t, r, "arriving"))
	require.NotNil(t, condition)
	assert.Equal(t, reasonWaitingForRoom, condition.Reason)
	assert.Contains(t, condition.Message, "on warm-2, the engine of starting cannot be read yet")
	assert.Contains(t, condition.Message, "1 more pod(s) could never hold it")
}
