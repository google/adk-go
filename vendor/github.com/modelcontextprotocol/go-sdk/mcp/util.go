// Copyright 2025 The Go MCP SDK Authors. All rights reserved.
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

package mcp

import (
	"crypto/rand"
)

func assert(cond bool, msg string) {
	if !cond {
		panic(msg)
	}
}

// Copied from crypto/rand.
// TODO: once 1.24 is assured, just use crypto/rand.
const base32alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"

func randText() string {
	// ⌈log₃₂ 2¹²⁸⌉ = 26 chars
	src := make([]byte, 26)
	rand.Read(src)
	for i := range src {
		src[i] = base32alphabet[src[i]%32]
	}
	return string(src)
}
