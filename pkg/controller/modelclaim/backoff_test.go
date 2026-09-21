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
	"time"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"
)

func TestPlacementBackoffWaitsLongerAfterEachRefusal(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	backoff := newPlacementBackoff(func() time.Time { return now })
	claim := types.NamespacedName{Namespace: testNamespace, Name: "qwen"}

	assert.Equal(t, DefaultRequeueDuration, backoff.refused(claim))
	assert.Equal(t, 2*DefaultRequeueDuration, backoff.refused(claim))
	assert.Equal(t, 4*DefaultRequeueDuration, backoff.refused(claim))
}

func TestPlacementBackoffStopsDoublingAtItsCeiling(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	backoff := newPlacementBackoff(func() time.Time { return now })
	claim := types.NamespacedName{Namespace: testNamespace, Name: "qwen"}

	wait := time.Duration(0)
	for i := 0; i < 40; i++ {
		wait = backoff.refused(claim)
	}

	assert.Equal(t, maximumPlacementBackoff, wait)
}

func TestPlacementBackoffHoldsAClaimUntilItsTurn(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	backoff := newPlacementBackoff(func() time.Time { return now })
	claim := types.NamespacedName{Namespace: testNamespace, Name: "qwen"}

	due, left := backoff.due(claim)
	assert.True(t, due)
	assert.Zero(t, left)

	backoff.refused(claim)
	due, left = backoff.due(claim)
	assert.False(t, due)
	assert.Equal(t, DefaultRequeueDuration, left)

	now = now.Add(DefaultRequeueDuration)
	due, _ = backoff.due(claim)
	assert.True(t, due)
}

func TestPlacementBackoffStartsOverOnceAClaimIsPlaced(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	backoff := newPlacementBackoff(func() time.Time { return now })
	claim := types.NamespacedName{Namespace: testNamespace, Name: "qwen"}

	backoff.refused(claim)
	backoff.refused(claim)
	backoff.placed(claim)

	due, _ := backoff.due(claim)
	assert.True(t, due)
	assert.Equal(t, DefaultRequeueDuration, backoff.refused(claim))
}
