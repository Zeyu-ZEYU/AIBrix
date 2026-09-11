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

func TestPlacementBackoffDuration(t *testing.T) {
	cases := []struct {
		attempts int
		want     time.Duration
	}{
		{attempts: 0, want: 10 * time.Second},
		{attempts: 1, want: 10 * time.Second},
		{attempts: 2, want: 20 * time.Second},
		{attempts: 3, want: 40 * time.Second},
		{attempts: 4, want: 80 * time.Second},
		{attempts: 5, want: 160 * time.Second},
		{attempts: 6, want: 5 * time.Minute},
		{attempts: 40, want: 5 * time.Minute},
	}
	for _, tc := range cases {
		assert.Equal(t, tc.want, placementBackoffDuration(tc.attempts),
			"attempt %d", tc.attempts)
	}
}

func TestPlacementBackoffDoublesRoundByRound(t *testing.T) {
	backoff := newPlacementBackoff()
	claim := types.NamespacedName{Namespace: testNamespace, Name: "refused"}

	assert.Equal(t, 10*time.Second, backoff.Next(claim))
	assert.Equal(t, 20*time.Second, backoff.Next(claim))
	assert.Equal(t, 40*time.Second, backoff.Next(claim))
}

func TestPlacementBackoffIsPerClaim(t *testing.T) {
	backoff := newPlacementBackoff()
	first := types.NamespacedName{Namespace: testNamespace, Name: "first"}
	second := types.NamespacedName{Namespace: testNamespace, Name: "second"}

	backoff.Next(first)
	backoff.Next(first)
	assert.Equal(t, 10*time.Second, backoff.Next(second),
		"one claim's refusals must not slow another down")
}

func TestPlacementBackoffForget(t *testing.T) {
	backoff := newPlacementBackoff()
	claim := types.NamespacedName{Namespace: testNamespace, Name: "refused"}

	backoff.Next(claim)
	backoff.Next(claim)
	backoff.Forget(claim)
	assert.Equal(t, 10*time.Second, backoff.Next(claim), "a forgotten claim starts again")
	assert.Len(t, backoff.attempts, 1)

	backoff.Forget(claim)
	assert.Empty(t, backoff.attempts, "a forgotten claim leaves nothing behind")
}

func TestPlacementBackoffNilIsUsable(t *testing.T) {
	var backoff *placementBackoff
	claim := types.NamespacedName{Namespace: testNamespace, Name: "refused"}
	assert.Equal(t, placementBackoffBase, backoff.Next(claim))
	assert.NotPanics(t, func() { backoff.Forget(claim) })
}
