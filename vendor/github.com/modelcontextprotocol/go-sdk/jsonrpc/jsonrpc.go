// Copyright 2025 The Go MCP SDK Authors. All rights reserved.
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

// Package jsonrpc exposes part of a JSON-RPC v2 implementation
// for use by mcp transport authors.
package jsonrpc

import "github.com/modelcontextprotocol/go-sdk/internal/jsonrpc2"

type (
	// ID is a JSON-RPC request ID.
	ID = jsonrpc2.ID
	// Message is a JSON-RPC message.
	Message = jsonrpc2.Message
	// Request is a JSON-RPC request.
	Request = jsonrpc2.Request
	// Response is a JSON-RPC response.
	Response = jsonrpc2.Response
)

// MakeID coerces the given Go value to an ID. The value is assumed to be the
// default JSON marshaling of a Request identifier -- nil, float64, or string.
//
// Returns an error if the value type was not a valid Request ID type.
func MakeID(v any) (ID, error) {
	return jsonrpc2.MakeID(v)
}

// EncodeMessage serializes a JSON-RPC message to its wire format.
func EncodeMessage(msg Message) ([]byte, error) {
	return jsonrpc2.EncodeMessage(msg)
}

// DecodeMessage deserializes JSON-RPC wire format data into a Message.
// It returns either a Request or Response based on the message content.
func DecodeMessage(data []byte) (Message, error) {
	return jsonrpc2.DecodeMessage(data)
}
