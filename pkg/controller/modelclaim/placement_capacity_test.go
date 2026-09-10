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
	"k8s.io/utils/ptr"
)

// bigClaim declares a model that needs more than half a test card, so two of
// them fill a two-pod pool and a third has nowhere to go.
func bigClaim(name string) *modelv1alpha1.ModelClaim {
	claim := sampleModelClaim()
	claim.Name = name
	claim.Spec.ModelName = ptr.To(name)
	claim.Spec.PerGPU = &modelv1alpha1.ModelClaimPerGPU{
		MaximumFootprintBytes: ptr.To(40 * gibibyte),
		KVFloorBytes:          ptr.To(4 * gibibyte),
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

// TestPlacementIgnoresMemoryForAClaimThatDeclaredNone pins the promise this
// change makes: it only ever adds refusals. A claim from before spec.perGPU
// existed is placed exactly as it was.
func TestPlacementIgnoresMemoryForAClaimThatDeclaredNone(t *testing.T) {
	silent := withFinalizer(sampleModelClaim())
	silent.Spec.PerGPU = nil
	resident := ledgerClaim("resident", "warm-1", 90*gibibyte, 4*gibibyte)
	warm1 := warmPod("warm-1", "b300-pool-a", true, corev1.PodRunning)
	r, runtime := newReconciler(t, silent, resident, warm1)

	reconcileOnce(t, r, silent.Name)

	placed := getModel(t, r, silent.Name)
	require.Len(t, placed.Status.Instances, 1,
		"a claim with nothing declared is placed on a card the ledger calls full")
	assert.Len(t, runtime.activateCalls, 1)
}

// TestPlacementIgnoresMemoryOnACardFreePod is the other half of that promise,
// for the mock and CPU-only pools the e2e suite deploys.
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
			Accelerators: []RuntimeAcceleratorSnapshot{
				{ID: "GPU-0", HBMTotalBytes: testHBMTotalBytes, HBMFreeBytes: testHBMTotalBytes},
			},
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
