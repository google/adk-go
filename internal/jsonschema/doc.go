// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package jsonschema is an implementation of the [JSON Schema specification],
a JSON-based format for describing the structure of JSON data.
The package can be used to read schemas for code generation, and to validate data using the
draft 2020-12 specification. Validation with other drafts or custom meta-schemas
is not supported.

Construct a [Schema] as you would any Go struct (for example, by writing a struct
literal), or unmarshal a JSON schema into a [Schema] in the usual way (with [encoding/json],
for instance). It can then be used for code generation or other purposes without
further processing.

# Validation

Before using a Schema to validate a JSON value, you must first resolve it by calling
[Schema.Resolve].
The call [Resolved.Validate] on the result to validate a JSON value.
The value must be a Go value that looks like the result of unmarshaling a JSON
value into an [any] or a struct. For example, the JSON value

	{"name": "Al", "scores": [90, 80, 100]}

could be represented as the Go value

	map[string]any{
		"name": "Al",
		"scores": []any{90, 80, 100},
	}

or as a value of this type:

	type Player struct {
		Name   string `json:"name"`
		Scores []int  `json:"scores"`
	}

# Inference

The [For] and [ForType] functions return a [Schema] describing the given Go type.
The type cannot contain any function or channel types, and any map types must have a string key.
For example, calling For on the above Player type results in this schema:

	{
	    "properties": {
	        "name": {
	            "type": "string"
	        },
	        "scores": {
	            "type": "array",
	            "items": {"type": "integer"}
	        }
	        "required": ["name", "scores"],
	        "additionalProperties": {"not": {}}
	    }
	}

# Deviations from the specification

Regular expressions are processed with Go's regexp package, which differs from ECMA 262,
most significantly in not supporting back-references.
See [this table of differences] for more.

The value of the "format" keyword is recorded in the Schema, but is ignored during validation.
It does not even produce [annotations].

[JSON Schema specification]: https://json-schema.org
[this table of differences] https://github.com/dlclark/regexp2?tab=readme-ov-file#compare-regexp-and-regexp2
[annotations]: https://json-schema.org/draft/2020-12/json-schema-core#name-annotations
*/
package jsonschema
