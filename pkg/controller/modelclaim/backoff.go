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

const (
	// placementBackoffBase is the first retry delay. It matches the ordinary
	// reconcile pace, so a claim's first retry is no slower than before.
	placementBackoffBase = 10 * time.Second
	// placementBackoffCap bounds the wait. A claim that cannot fit should cost
	// almost nothing to keep pending, but it must still notice a change that
	// reaches it through no event at all.
	placementBackoffCap = 5 * time.Minute
)

// placementBackoff slows down retries for claims that no warm pod will take.
// It is deliberately process-local: a controller restart resets every claim to
// the base delay, which costs a few extra rounds and needs no API surface to
// persist.
//
// It does not try to notice when a claim's chances change. A placement, a
// route coming up or going away, and a model leaving all change a warm pod,
// and every change to a warm pod reconciles every claim at once. The delay
// only decides how often a claim is looked at while nothing happens.
type placementBackoff struct {
	mu       sync.Mutex
	attempts map[types.NamespacedName]int
}

func newPlacementBackoff() *placementBackoff {
	return &placementBackoff{attempts: make(map[types.NamespacedName]int)}
}

// Next records one more round in which no warm pod would take the claim, and
// returns how long to wait before the next.
func (b *placementBackoff) Next(claim types.NamespacedName) time.Duration {
	if b == nil {
		return placementBackoffBase
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.attempts[claim]++
	return placementBackoffDuration(b.attempts[claim])
}

// Forget drops a claim's count once it is placed, is only waiting for room,
// or has gone away. Without it the map would grow with every claim ever
// refused.
func (b *placementBackoff) Forget(claim types.NamespacedName) {
	if b == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	delete(b.attempts, claim)
}

// placementBackoffDuration doubles the base delay per consecutive round and
// stops at the cap: 10s, 20s, 40s, 80s, 160s, then 5m for good.
func placementBackoffDuration(attempts int) time.Duration {
	if attempts < 1 {
		attempts = 1
	}
	delay := placementBackoffBase
	for i := 1; i < attempts; i++ {
		delay *= 2
		if delay >= placementBackoffCap {
			return placementBackoffCap
		}
	}
	return delay
}
