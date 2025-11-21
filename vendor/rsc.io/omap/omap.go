// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package omap implements in-memory ordered maps.
// [Map][K, V] is suitable for ordered types K,
// while [MapFunc][K, V] supports arbitrary keys and comparison functions.
package omap

// The implementation is a treap. See:
// https://en.wikipedia.org/wiki/Treap
// https://faculty.washington.edu/aragon/pubs/rst89.pdf

import (
	"cmp"
	"iter"
	"math/rand/v2"
)

// A Map is a map[K]V ordered according to K's standard Go ordering.
// The zero value of a Map is an empty Map ready to use.
type Map[K cmp.Ordered, V any] struct {
	_root *node[K, V]
}

// A MapFunc is a map[K]V ordered according to an arbitrary comparison function.
// The zero value of a MapFunc is not meaningful since it has no comparison function.
// Use [NewMapFunc] to create a [MapFunc].
// A nil *MapFunc, like a nil Go map, can be read but not written and contains no entries.
type MapFunc[K, V any] struct {
	_root *node[K, V]
	cmp   func(K, K) int
}

// A node is a node in the treap.
type node[K any, V any] struct {
	parent *node[K, V]
	left   *node[K, V]
	right  *node[K, V]
	key    K
	val    V
	pri    uint64
}

// NewMapFunc returns a new MapFunc[K, V] ordered according to cmp.
func NewMapFunc[K, V any](cmp func(K, K) int) *MapFunc[K, V] {
	return &MapFunc[K, V]{cmp: cmp}
}

// omap is the interface implemented by both Map[K, V] and MapFunc[K, V]
// that enables a common implementation of the map operations.
type omap[K, V any] interface {
	// root returns &m._root; the caller can read or write *m.root().
	root() **node[K, V]

	// find reports where a node with the key would be: at *pos.
	// If *pos != nil, then key is present in the tree;
	// otherwise *pos is where a new node with the key should be attached.
	//
	// If parent != nil, then pos is either &parent.left or &parent.right
	// depending on how parent.key compares with key.
	// If parent == nil, then pos is m.root().
	find(key K) (pos **node[K, V], parent *node[K, V])
}

func (m *Map[K, V]) root() **node[K, V]     { return &m._root }
func (m *MapFunc[K, V]) root() **node[K, V] { return &m._root }

// find looks up the key k in the map.
// It returns the parent of k as well as the position where k would be attached.
// *pos is non-nil if k is present, nil if k is missing.
// parent is nil if there are no nodes in the map, or if there is only one node and it's k.
func (m *Map[K, V]) find(k K) (pos **node[K, V], parent *node[K, V]) {
	pos = &m._root
	for x := *pos; x != nil; x = *pos {
		if x.key == k {
			break
		}
		parent = x
		if x.key > k {
			pos = &x.left
		} else {
			pos = &x.right
		}
	}
	return pos, parent
}

// find is the same as for Map[K, V] but using m.cmp.
func (m *MapFunc[K, V]) find(k K) (pos **node[K, V], parent *node[K, V]) {
	pos = &m._root
	for x := *pos; x != nil; x = *pos {
		cmp := m.cmp(x.key, k)
		if cmp == 0 {
			break
		}
		parent = x
		if cmp > 0 {
			pos = &x.left
		} else {
			pos = &x.right
		}
	}
	return pos, parent
}

// Get returns the value of m[key] and reports whether it exists.
func (m *Map[K, V]) Get(key K) (V, bool) {
	return get(m, key)
}

// Get returns the value of m[key] and reports whether it exists.
func (m *MapFunc[K, V]) Get(key K) (V, bool) {
	return get(m, key)
}

func get[K, V any](m omap[K, V], key K) (V, bool) {
	pos, _ := m.find(key)
	if x := *pos; x != nil {
		return x.val, true
	}
	var zero V
	return zero, false
}

// Set sets m[key] = val.
func (m *Map[K, V]) Set(key K, val V) {
	set(m, key, val)
}

// Set sets m[key] = val.
func (m *MapFunc[K, V]) Set(key K, val V) {
	set(m, key, val)
}

func set[K, V any](m omap[K, V], key K, val V) {
	pos, parent := m.find(key)
	if x := *pos; x != nil {
		x.val = val
		return
	}
	x := &node[K, V]{key: key, val: val, pri: rand.Uint64() | 1, parent: parent}
	*pos = x
	rotateUp(m, x)
}

// rotateUp rotates x upward in the tree to correct any priority inversions.
func rotateUp[K, V any](m omap[K, V], x *node[K, V]) {
	// Rotate up into tree according to priority.
	for x.parent != nil && x.parent.pri > x.pri {
		if x.parent.left == x {
			rotateRight(m, x.parent)
		} else {
			rotateLeft(m, x.parent)
		}
	}
}

// Delete deletes m[key].
func (m *Map[K, V]) Delete(key K) {
	_delete(m, key)
}

// Delete deletes m[key].
func (m *MapFunc[K, V]) Delete(key K) {
	_delete(m, key)
}

func _delete[K, V any](m omap[K, V], key K) {
	pos, _ := m.find(key)
	x := *pos
	if x == nil {
		return
	}

	// Rotate x down to be leaf of tree for removal, respecting priorities.
	for x.right != nil || x.left != nil {
		if x.right == nil || x.left != nil && x.left.pri < x.right.pri {
			rotateRight(m, x)
		} else {
			rotateLeft(m, x)
		}
	}

	// Remove x, now a leaf.
	switch p := x.parent; {
	case p == nil:
		*m.root() = nil
	case p.left == x:
		p.left = nil
	default:
		p.right = nil
	}
	x.pri = 0 // mark deleted
}

// DeleteRange deletes m[k] for all keys k satisfying lo ≤ k ≤ hi.
func (m *Map[K, V]) DeleteRange(lo, hi K) {
	if lo > hi {
		return
	}
	deleteRange(m, lo, hi)
}

// DeleteRange deletes m[k] for all keys k satisfying lo ≤ k ≤ hi.
func (m *MapFunc[K, V]) DeleteRange(lo, hi K) {
	if m.cmp(lo, hi) > 0 {
		return
	}
	deleteRange(m, lo, hi)
}

func deleteRange[K, V any](m omap[K, V], lo, hi K) {
	after := split(m, hi)
	middle := split(m, lo)
	markDeleted(middle)
	if after != nil {
		pos, parent := m.find(lo)
		*pos = after
		after.parent = parent
		rotateUp(m, after)
	}
}

func split[K, V any](m omap[K, V], key K) (before *node[K, V]) {
	pos, parent := m.find(key)
	if *pos == nil {
		*pos = &node[K, V]{parent: parent}
	}
	x := *pos
	x.pri = 0
	rotateUp(m, x)

	*m.root() = x.left
	if x.left != nil {
		x.left.parent = nil
	}
	return x.right
}

func markDeleted[K, V any](x *node[K, V]) {
	if x == nil {
		return
	}
	x.pri = 0
	markDeleted(x.left)
	markDeleted(x.right)
}

// All returns an iterator over the map m.
// If m is modified during the iteration, some keys may not be visited.
// No keys will be visited multiple times.
func (m *Map[K, V]) All() iter.Seq2[K, V] {
	return all(m)
}

// All returns an iterator over the map m.
// If m is modified during the iteration, some keys may not be visited.
// No keys will be visited multiple times.
func (m *MapFunc[K, V]) All() iter.Seq2[K, V] {
	return all(m)
}

// all returns an iterator over the map m, where *root is the root.
func all[K, V any](m omap[K, V]) iter.Seq2[K, V] {
	return func(yield func(K, V) bool) {
		x := *m.root()
		if x != nil {
			for x.left != nil {
				x = x.left
			}
		}
		for x != nil && yield(x.key, x.val) {
			x = x.next(m)
		}
	}
}

// Scan returns an iterator over the map m
// limited to keys k satisfying lo ≤ k ≤ hi.
//
// If m is modified during the iteration, some keys may not be visited.
// No keys will be visited multiple times.
func (m *Map[K, V]) Scan(lo, hi K) iter.Seq2[K, V] {
	return func(yield func(K, V) bool) {
		x, _ := findGE(m, lo)
		for x != nil && x.key <= hi && yield(x.key, x.val) {
			x = x.next(m)
		}
	}
}

// Scan returns an iterator over the map m
// limited to keys k satisfying lo ≤ k ≤ hi.
//
// If m is modified during the iteration, some keys may not be visited.
// No keys will be visited multiple times.
func (m *MapFunc[K, V]) Scan(lo, hi K) iter.Seq2[K, V] {
	return func(yield func(K, V) bool) {
		x, _ := findGE(m, lo)
		for x != nil && m.cmp(x.key, hi) <= 0 && yield(x.key, x.val) {
			x = x.next(m)
		}
	}
}

// next returns the successor node of x in the treap,
// even if x has been removed from the treap.
// x must not be nil.
func (x *node[K, V]) next(m omap[K, V]) *node[K, V] {
	if x.pri == 0 {
		// x has been deleted.
		// Find where x.key would be in the current tree.
		var eq bool
		x, eq = findGE(m, x.key)
		if !eq {
			// The new x is already greater than the old x.key.
			return x
		}
	}

	if x.right == nil {
		for x.parent != nil && x.parent.right == x {
			x = x.parent
		}
		return x.parent
	}
	x = x.right
	for x.left != nil {
		x = x.left
	}
	return x
}

// findGE finds the node x in m with the least key k such that k ≥ key.
func findGE[K, V any](m omap[K, V], key K) (x *node[K, V], eq bool) {
	pos, parent := m.find(key)
	if *pos != nil {
		return *pos, true
	}
	if parent == nil {
		return nil, false
	}
	if pos == &parent.left {
		return parent, false
	}
	return parent.next(m), false
}

// rotateLeft rotates the subtree rooted at node x.
// turning (x a (y b c)) into (y (x a b) c).
func rotateLeft[K, V any](m omap[K, V], x *node[K, V]) {
	// p -> (x a (y b c))
	p := x.parent
	y := x.right
	b := y.left

	y.left = x
	x.parent = y
	x.right = b
	if b != nil {
		b.parent = x
	}

	y.parent = p
	if p == nil {
		*m.root() = y
	} else if p.left == x {
		p.left = y
	} else if p.right == x {
		p.right = y
	} else {
		// unreachable
		panic("corrupt treap")
	}
}

// rotateRight rotates the subtree rooted at node y.
// turning (y (x a b) c) into (x a (y b c)).
func rotateRight[K, V any](m omap[K, V], y *node[K, V]) {
	// p -> (y (x a b) c)
	p := y.parent
	x := y.left
	b := x.right

	x.right = y
	y.parent = x
	y.left = b
	if b != nil {
		b.parent = y
	}

	x.parent = p
	if p == nil {
		*m.root() = x
	} else if p.left == y {
		p.left = x
	} else if p.right == y {
		p.right = x
	} else {
		// unreachable
		panic("corrupt treap")
	}
}
