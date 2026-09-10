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

func TestPlacementBackoffDoublesWhileNothingChanges(t *testing.T) {
	backoff := newPlacementBackoff()
	claim := types.NamespacedName{Namespace: testNamespace, Name: "waiting"}

	assert.Equal(t, 10*time.Second, backoff.Next(claim, 1))
	assert.Equal(t, 20*time.Second, backoff.Next(claim, 1))
	assert.Equal(t, 40*time.Second, backoff.Next(claim, 1))
}

func TestPlacementBackoffResetsWhenTheLedgerMoves(t *testing.T) {
	backoff := newPlacementBackoff()
	claim := types.NamespacedName{Namespace: testNamespace, Name: "waiting"}

	backoff.Next(claim, 1)
	backoff.Next(claim, 1)
	assert.Equal(t, 40*time.Second, backoff.Next(claim, 1))

	// A neighbour left, or a card appeared: the claim is refused against a
	// different account and must try again promptly rather than sit out the
	// wait it had earned against the old one.
	assert.Equal(t, 10*time.Second, backoff.Next(claim, 2),
		"a changed fingerprint restarts the backoff")
	assert.Equal(t, 20*time.Second, backoff.Next(claim, 2))
}

func TestPlacementBackoffIsPerClaim(t *testing.T) {
	backoff := newPlacementBackoff()
	first := types.NamespacedName{Namespace: testNamespace, Name: "first"}
	second := types.NamespacedName{Namespace: testNamespace, Name: "second"}

	backoff.Next(first, 1)
	backoff.Next(first, 1)
	assert.Equal(t, 10*time.Second, backoff.Next(second, 1),
		"one claim's failures must not slow another down")
}

func TestPlacementBackoffForget(t *testing.T) {
	backoff := newPlacementBackoff()
	claim := types.NamespacedName{Namespace: testNamespace, Name: "waiting"}

	backoff.Next(claim, 1)
	backoff.Next(claim, 1)
	backoff.Forget(claim)
	assert.Equal(t, 10*time.Second, backoff.Next(claim, 1))
	assert.Len(t, backoff.entries, 1)

	backoff.Forget(claim)
	assert.Empty(t, backoff.entries, "a forgotten claim leaves nothing behind")
}

func TestPlacementBackoffNilIsUsable(t *testing.T) {
	var backoff *placementBackoff
	claim := types.NamespacedName{Namespace: testNamespace, Name: "waiting"}
	assert.Equal(t, placementBackoffBase, backoff.Next(claim, 1))
	assert.NotPanics(t, func() { backoff.Forget(claim) })
}
