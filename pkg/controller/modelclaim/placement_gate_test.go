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
	"time"

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
	claim.Spec.PerGPU = &modelv1alpha1.ModelClaimPerGPU{
		MaximumFootprintBytes: ptr.To(40 * gibibyte),
		KVFloorBytes:          ptr.To(4 * gibibyte),
	}
	return withFinalizer(claim)
}

func twoWarmPods(t *testing.T) (*corev1.Pod, *corev1.Pod) {
	t.Helper()
	first := warmPod("warm-1", "b300-pool-a", true, corev1.PodRunning)
	first.Status.PodIP = "10.0.0.1"
	second := warmPod("warm-2", "b300-pool-a", true, corev1.PodRunning)
	second.Status.PodIP = "10.0.0.2"
	return first, second
}

func reconcileFor(t *testing.T, r *ModelClaimReconciler, name string) ctrl.Result {
	t.Helper()
	result, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Namespace: testNamespace, Name: name},
	})
	require.NoError(t, err)
	return result
}

func scheduledCondition(t *testing.T, claim *modelv1alpha1.ModelClaim) *metav1.Condition {
	t.Helper()
	return meta.FindStatusCondition(claim.Status.Conditions,
		string(modelv1alpha1.ModelClaimConditionTypeScheduled))
}

// TestPlacementSpreadsThenWaitsThenPlacesOnFreedRoom walks the whole gate. Two
// models that each need more than half a card land on separate cards; a third
// finds no card that could ever hold it and waits, doubling its wait; and when
// one of the first two goes away it is placed on the card that came free.
func TestPlacementSpreadsThenWaitsThenPlacesOnFreedRoom(t *testing.T) {
	ctx := context.Background()
	first, second, third := bigClaim("first"), bigClaim("second"), bigClaim("third")
	warm1, warm2 := twoWarmPods(t)
	r, runtime := newReconciler(t, first, second, third, warm1, warm2)

	reconcileOnce(t, r, "first")
	reconcileOnce(t, r, "second")

	firstPod := getModel(t, r, "first").Status.Instances[0].Pod
	secondPod := getModel(t, r, "second").Status.Instances[0].Pod
	assert.NotEqual(t, firstPod, secondPod,
		"two models that each need more than half a card cannot share one")
	require.Len(t, runtime.activateCalls, 2)

	// The third model needs 44 GiB and neither card could free that much even
	// by putting its resident engine to sleep, so it is refused everywhere.
	result := reconcileFor(t, r, "third")
	waiting := getModel(t, r, "third")
	assert.Empty(t, waiting.Status.Instances)
	assert.Len(t, runtime.activateCalls, 2, "a refused claim must not start an engine")
	condition := scheduledCondition(t, waiting)
	require.NotNil(t, condition)
	assert.Equal(t, metav1.ConditionFalse, condition.Status)
	assert.Equal(t, refusalInsufficientCapacity, condition.Reason)
	assert.Contains(t, condition.Message, "2 candidate pod(s) refused")
	assert.Equal(t, placementBackoffBase, result.RequeueAfter)

	// Nothing has changed, so the wait doubles instead of hammering the pool.
	assert.Equal(t, 20*time.Second, reconcileFor(t, r, "third").RequeueAfter)
	assert.Equal(t, 40*time.Second, reconcileFor(t, r, "third").RequeueAfter)

	// The first model is deleted. Its instance is deactivated and its line
	// leaves the ledger, which is exactly the change the third was waiting on.
	require.NoError(t, r.Delete(ctx, getModel(t, r, "first")))
	reconcileOnce(t, r, "first")

	result = reconcileFor(t, r, "third")
	placed := getModel(t, r, "third")
	require.Len(t, placed.Status.Instances, 1)
	assert.Equal(t, firstPod, placed.Status.Instances[0].Pod,
		"the third model takes the card the first one released")
	assert.Equal(t, DefaultRequeueDuration, result.RequeueAfter,
		"a placed claim goes back to the ordinary reconcile pace")
}

// TestPlacementRefusesAClaimWithNoDeclaredNumbers covers the admission check:
// the control plane will not place a model whose cost it was never told.
func TestPlacementRefusesAClaimWithNoDeclaredNumbers(t *testing.T) {
	silent := withFinalizer(sampleModelClaim())
	silent.Spec.PerGPU = nil
	warm1, warm2 := twoWarmPods(t)
	r, runtime := newReconciler(t, silent, warm1, warm2)

	result := reconcileFor(t, r, silent.Name)

	got := getModel(t, r, silent.Name)
	assert.Empty(t, got.Status.Instances)
	assert.Empty(t, runtime.activateCalls)
	condition := scheduledCondition(t, got)
	require.NotNil(t, condition)
	assert.Equal(t, reasonNumbersMissing, condition.Reason)
	assert.Contains(t, condition.Message, "spec.perGPU")
	assert.Equal(t, placementBackoffBase, result.RequeueAfter)
}

// TestPlacementRefusesACardWithAnUndeclaredResident covers the other half of
// fail-closed: a card is only usable while every engine on it can be accounted
// for. An unmeasured neighbour hides an unknown amount of memory.
func TestPlacementRefusesACardWithAnUndeclaredResident(t *testing.T) {
	resident := ledgerClaim("resident", "warm-1", 0, 0)
	arriving := bigClaim("arriving")
	warm1 := warmPod("warm-1", "b300-pool-a", true, corev1.PodRunning)
	warm1.Status.PodIP = "10.0.0.1"
	r, runtime := newReconciler(t, resident, arriving, warm1)

	reconcileFor(t, r, "arriving")

	got := getModel(t, r, "arriving")
	assert.Empty(t, got.Status.Instances)
	assert.Empty(t, runtime.activateCalls)
	condition := scheduledCondition(t, got)
	require.NotNil(t, condition)
	assert.Equal(t, refusalLedgerIncomplete, condition.Reason)
	assert.Contains(t, condition.Message, "resident",
		"the operator is told which claim to fix")
}

// TestPlacementWalksPastAFullCardToAnEmptyOne is the property the single
// winner search could not have: the best-ranked pod is refused and the round
// still succeeds on a later one.
func TestPlacementWalksPastAFullCardToAnEmptyOne(t *testing.T) {
	// warm-1 is where the artifact is already cached, so ranking prefers it,
	// but a resident model has taken most of that card.
	arriving := bigClaim("arriving")
	resident := ledgerClaim("resident", "warm-1", 40*gibibyte, 4*gibibyte)
	warm1, warm2 := twoWarmPods(t)
	r, runtime := newReconciler(t, resident, arriving, warm1, warm2)
	runtime.snapshots = map[string]*RuntimeSnapshot{
		warm1.Status.PodIP: {
			Accelerators: []RuntimeAcceleratorSnapshot{
				{ID: "GPU-0", HBMTotalBytes: testHBMTotalBytes, HBMFreeBytes: testHBMTotalBytes},
			},
			CachedArtifacts: []string{arriving.Spec.ArtifactURL},
		},
	}

	ordered := rankCandidates(
		[]corev1.Pod{*warm1, *warm2}, map[string]bool{}, map[string]int{},
		"arriving", uniformLocality{},
		r.collectPlacementStates(context.Background(), []corev1.Pod{*warm1, *warm2},
			arriving.Spec.ArtifactURL, 1),
	)
	require.Equal(t, "warm-1", ordered[0].Name,
		"the cached artifact makes the full card the favourite")

	reconcileFor(t, r, "arriving")

	got := getModel(t, r, "arriving")
	require.Len(t, got.Status.Instances, 1)
	assert.Equal(t, "warm-2", got.Status.Instances[0].Pod,
		"placement walks past its favourite card because that card has no room")
	require.Len(t, runtime.activateCalls, 1)
}
