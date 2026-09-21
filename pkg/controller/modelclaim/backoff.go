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
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/types"
)

// maximumPlacementBackoff is how far apart the attempts of a claim nothing can
// hold are allowed to drift. A pool that gains a card should take the waiting
// model within a minute of it happening, and a model that has been waiting all
// morning should not be asking a hundred runtimes about it every ten seconds.
const maximumPlacementBackoff = time.Minute

// placementBackoff spaces out the attempts of a claim no card can take.
//
// It is deliberately controller-local. A restart clears it, which costs one
// eager round of placement and nothing else, and that is a better trade than a
// field in the API that every reader would have to understand.
type placementBackoff struct {
	mu       sync.Mutex
	now      func() time.Time
	attempts map[types.NamespacedName]placementAttempt
}

type placementAttempt struct {
	refusals int
	readyAt  time.Time
}

func newPlacementBackoff(now func() time.Time) *placementBackoff {
	if now == nil {
		now = time.Now
	}
	return &placementBackoff{
		now:      now,
		attempts: make(map[types.NamespacedName]placementAttempt),
	}
}

// due reports whether this claim may try to find a card again, and how long is
// left when it may not.
func (b *placementBackoff) due(claim types.NamespacedName) (bool, time.Duration) {
	b.mu.Lock()
	defer b.mu.Unlock()
	attempt, waiting := b.attempts[claim]
	if !waiting {
		return true, 0
	}
	left := attempt.readyAt.Sub(b.now())
	if left <= 0 {
		return true, 0
	}
	return false, left
}

// refused records that no card could take this claim, and returns how long to
// wait before asking again. The wait doubles with each refusal in a row.
func (b *placementBackoff) refused(claim types.NamespacedName) time.Duration {
	b.mu.Lock()
	defer b.mu.Unlock()
	attempt := b.attempts[claim]
	attempt.refusals++
	wait := DefaultRequeueDuration << min(attempt.refusals-1, 16)
	if wait > maximumPlacementBackoff || wait <= 0 {
		wait = maximumPlacementBackoff
	}
	attempt.readyAt = b.now().Add(wait)
	b.attempts[claim] = attempt
	return wait
}

// placed clears the wait, so a claim that lands on a card starts from scratch
// if it ever has to wait again.
func (b *placementBackoff) placed(claim types.NamespacedName) {
	b.mu.Lock()
	defer b.mu.Unlock()
	delete(b.attempts, claim)
}
