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
	// placementBackoffBase is the first retry delay, and matches the ordinary
	// reconcile pace so a claim's first few retries are no slower than before.
	placementBackoffBase = 10 * time.Second
	// placementBackoffCap bounds the wait. A claim that cannot fit should cost
	// almost nothing to keep pending, but it must still notice a card that
	// frees up without anything else waking it.
	placementBackoffCap = 5 * time.Minute
)

// placementBackoff slows down retries for claims that could not be placed,
// and speeds them back up the moment the situation they are waiting on
// changes. It is deliberately process-local: a controller restart resets every
// claim to the base delay, which costs a few extra rounds and needs no API
// surface to persist.
type placementBackoff struct {
	mu      sync.Mutex
	entries map[types.NamespacedName]placementBackoffEntry
}

type placementBackoffEntry struct {
	attempts    int
	fingerprint uint64
}

func newPlacementBackoff() *placementBackoff {
	return &placementBackoff{entries: make(map[types.NamespacedName]placementBackoffEntry)}
}

// Next records one failed placement round for a claim and returns how long to
// wait before the next. The fingerprint describes what the claim was refused
// against; when it differs from the previous round the situation has moved and
// the delay drops back to the base, so a claim is placed within one reconcile
// of room appearing rather than at the end of a long wait.
func (b *placementBackoff) Next(claim types.NamespacedName, fingerprint uint64) time.Duration {
	if b == nil {
		return placementBackoffBase
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	entry, found := b.entries[claim]
	if !found || entry.fingerprint != fingerprint {
		entry = placementBackoffEntry{fingerprint: fingerprint}
	}
	entry.attempts++
	b.entries[claim] = entry
	return placementBackoffDuration(entry.attempts)
}

// Forget drops a claim's backoff state, on a successful placement or when the
// claim goes away. Without it the map would grow with every claim ever seen.
func (b *placementBackoff) Forget(claim types.NamespacedName) {
	if b == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	delete(b.entries, claim)
}

// placementBackoffDuration doubles the base delay per consecutive failure and
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
