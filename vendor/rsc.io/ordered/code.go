// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package ordered defines an encoding of a sequence of basic typed values into []byte and back.

[Encode] converts a list of values to a []byte;
[Append] is like [Encode] but appends to an existing buffer.

[Decode] converts a []byte back to a list of typed values.
[DecodePrefix] is like [Decode] but decodes only a prefix of the encoded data.

Define the notation E([list]) to mean the encoding of a list of values
The encoding has a few important properties:

  - Typed: The types of the values are part of the encoding,
    so decoding can check that the expected types match the actual types,
    and decoding can proceed without knowing the types at all.

  - Unique: There is only one possible encoding for a given list of values.

  - Associative: The encoding of a list of values is the concatenation
    of the encodings of individual values. E([A₁, ..., Aₙ]) = E([A₁]) + ... + E([Aₙ]).

  - Prefix-Preserving: The associative property implies that any encoding
    of a list of values is ordered less than any longer list sharing the same prefix:
    E([A₁, ..., Aₘ]) < E([A₁, ..., Aₘ, ..., Aₙ]).

  - Ordered: for any two values A, B of the same type,
    bytes.Compare(E([A]), E([B])) == cmp.Compare(A, B).

    The associative property implies an extension of the ordered property:
    for any two lists of values [A₁, ..., Aₙ] and [B₁, ..., Bₙ],
    where Aᵢ and Bᵢ have the same type for all i,
    bytes.Compare(E([A₁, ..., Aₙ]), E([B₁, ..., Bₙ])) returns the same result
    as comparing the lists with cmp.Compare(A₁, B₁), cmp.Compare(A₂, B₂), ...,
    cmp.Compare(Aₙ, Bₙ), stopping at the first non-zero result.

These properties are particularly useful for preparing keys in an ordered key/value storage
such as Bigtable, Bolt, CockroachDB, LevelDB, Pebble, RocksDB, or Spanner.

The encoding is also useful when a basic, efficient, typed serialization is needed.

# Infinity

The value [Inf] of type [Infinity] has an encoding greater than any
non-infinite value's encoding.

The main use for infinity is to provide an upper bound for range scans
considering a certain prefix.
E([prefix]) < E([prefix, x]) < E([prefix, Inf]) for all x,
so E([prefix]) and E([prefix, Inf]) can be used as the scan bounds.

There is no explicit negative infinity value; E[] = "" can be used instead,
as was done implicitly in the previous example.
Alternately, the reversed infinity Rev(Inf) encodes as "\x00" and
can also be used as a negative infinity.

The type [StringOrInfinity] holds either a string or an infinity.

# Reverse Order

It can be necessary to order encodings in the reverse order of the values,
so that bytes.Compare orders values from greatest to least.
The type [Reverse][T] provides that ordering.
For x of type T, [Rev](x) returns a Reverse[T] holding the value x.

Given x and y of type T, bytes.Compare(E([Rev(x)]), E([Rev(y)])) = cmp.Compare(y, x).

# Raw Data

It can be useful and efficient to store a large []byte as itself at the end
of an encoding. The type [Raw] represents such a sequence.
It is encoded with a length prefix, so that shorter values are ordered
before longer ones, and values of the same length are ordered
to match bytes.Compare.
When decoding, the extracted values of type []byte are copies of the
original slice, not pointers into the original.
Raw is an exception: decoding a [Raw] returns a slice overlapping
the original encoded input.

# Encodings

This package only supports encoding a limited number of types.
The types and their encodings are:

  - string and []byte: The encoding is a 0x01 byte followed by the string data
    and then a 0x00 0x00 end-of-string marker.
    In the string data, any 0x00 byte is replaced with 0x00 0xFF,
    to avoid ambiguity with the marker.

    For example, the encoding of "hello\x00world" is "\x01hello\x00\xFFworld\x00\x00",
    and the encoding of "" is "\x01\x00\x00".

  - int, uint, int8, uint8, ..., int64, uint64, uintptr:
    The encoding of a non-negative integer x is a byte 0x30+(n-1) for n in 1..8
    where n is the number of significant low bytes of x.
    That byte is followed by the low n bytes of x, most significant
    to least significant.

    For example, the encoding of 0x12345 is "\x32\x01\x23\x45",
    the encoding of 1 is "\x30\x01", and the encoding of 0 is "\x31\x00".

    The encoding of a negative integer x is a byte 0x30-n for n in 1..8
    where n is the number of significant low bytes of ^x = -(x+1).
    That byte is followed by the low n bytes of x, most significant to
    least significant.

    For example, the encoding of -0x12345 is "\x2D\xFE\xDC\xBB",
    and the encoding of -1 is "\x2F\xFF".

    Because all integer types share a single encoding, it is valid to
    encode using one type and decode using a different type,
    provided the values encoded are in range for the decoding type,
    and the specific integer type does not play a role in the overall
    ordering of encodings.

  - float32: The encoding of a float32 f is a 0x02 byte followed by
    a big-endian uint32 u representing f.
    If f is a NaN, then u = 0.
    Otherwise, u is math.Float32bits(f) with the top (sign) bit always
    inverted and the remaining bits inverted when f < 0.

  - float64: The encoding of a float32 f is a 0x03 byte followed by
    a big-endian uint64 u representing f.
    If f is a NaN, then u = 0.
    Otherwise, u is math.Float64bits(f) with the top (sign) bit always
    inverted and the remaining bits inverted when f < 0.

    Although the float32 and float64 encodings are similar,
    they are not the same, so float32 and float64 values cannot be
    interchanged between encode and decode, and all float32
    values order before all float64 values.

  - [Infinity]: [Inf] encodes as the byte 0xFF.

  - [StringOrInfinity]: The encoding is either the encoding of the string
    or the encoding of infinity, depending on which the value is.

  - [Reverse][T]: The encoding of [Rev](x) is the encoding of x with all
    bits negated (equivalently, all bytes XOR'ed with 0xFF).

  - [Raw]: The encoding is a 0x04 byte followed by the raw bytes.
    The content is the remainder of the encoded data.

# Type Ordering

Code should in general not depend on the relative ordering of values of different types,
but the order is:
Reverse[Infinity] < string/[]byte < float32 < float64 < Raw < integer <
Reverse[integer] < Reverse[float64] < Reverse[float32] < Reverse[string/[]byte] < Infinity.

# Compatibility

Because the encodings are expected to be used as keys and values
in storage systems, existing encodings will not changed in future versions.
New types may be introduced.

Although this package is inspired by [github.com/google/orderedcode],
this package uses a different, typed (self-describing) encoding.
*/
package ordered

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/bits"
)

// opcode is the type byte for an encoded value.
type opcode uint8

const (
	opString  opcode = 0x01
	opFloat32 opcode = 0x02
	opFloat64 opcode = 0x03
	opRaw     opcode = 0x04
	opInt     opcode = 0x20
	opPosInt  opcode = 0x10 // bit indicating positive for opInt
	opInf     opcode = 0xFF
	rev              = 0xFF // reversed values XOR with rev
)

func (o opcode) String() string {
	if o == opInf {
		return "infinity"
	}
	if o == 0 {
		return "rev infinity"
	}
	rev := ""
	if o&0x80 != 0 {
		rev = "rev "
		o = ^o
	}
	if o&0xE0 == opInt {
		o = opInt
	}

	switch o {
	case opString:
		return rev + "string"
	case opFloat32:
		return rev + "float32"
	case opFloat64:
		return rev + "float64"
	case opRaw:
		return rev + "raw"
	case opInt:
		return rev + "int"
	}
	return fmt.Sprintf("opcode(%#x)", uint8(o))
}

// Infinity is the infinity type. [Inf] is its only value.
type Infinity struct{}

// Inf is the only value of type [Infinity].
var Inf Infinity

// A StringOrInfinity represents either a string or an infinity.
// If Inf is true, it represents infinity, and S is ignored.
// Otherwise, it represents the string S.
type StringOrInfinity struct {
	S   string
	Inf bool
}

// Reversible is the type constraint for types that can be used with [Reverse].
type Reversible interface {
	string | []byte |
		int8 | int16 | int32 | int64 | int |
		uint8 | uint16 | uint32 | uint64 | uint | uintptr |
		float32 | float64 |
		Infinity |
		StringOrInfinity
}

// Reverse[T] holds a single value of type T, accessible using the [Reverse.value] method.
// [Rev] and [RevAny] make a Reverse[T].
//
// For two values x < y, Encode(x) < Encode(y) but Encode(Rev(x)) > Encode(Rev(y)).
type Reverse[T Reversible] struct {
	v T
}

// Raw is a []byte that encodes as itself, with only a type prefix added.
// When decoding into a Raw, the Raw is a subslice of the original encoded input,
// not a copy.
type Raw []byte

// Value returns the value x for which r == Rev(x).
func (r Reverse[T]) Value() T {
	return r.v
}

// Rev returns the reversed value representing x.
func Rev[T Reversible](x T) Reverse[T] {
	return Reverse[T]{x}
}

// RevAny is the untyped version of [Rev].
// It panics if x is not a [Reversible] type.
func RevAny(x any) any {
	switch x := x.(type) {
	default:
		panic(fmt.Sprintf("ordered: invalid type %T", x))
	case string:
		return Rev(x)
	case []byte:
		return Rev(x)
	case Infinity:
		return Rev(x)
	case StringOrInfinity:
		return Rev(x)
	case uint64:
		return Rev(x)
	case uint32:
		return Rev(x)
	case uint16:
		return Rev(x)
	case uint8:
		return Rev(x)
	case uint:
		return Rev(x)
	case uintptr:
		return Rev(x)
	case int64:
		return Rev(x)
	case int32:
		return Rev(x)
	case int16:
		return Rev(x)
	case int8:
		return Rev(x)
	case int:
		return Rev(x)
	case float64:
		return Rev(x)
	case float32:
		return Rev(x)
	}
}

// CanEncode reports whether the values x in the list can be encoded.
// A value can be encoded if its type is []byte, string, Infinity, StringOrInfinity,
// int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, uintptr,
// float32, float64, or if its type is Reverse[T] for any type T in the
// preceding list.
func CanEncode(list ...any) bool {
	for _, x := range list {
		switch x.(type) {
		default:
			return false
		case []byte,
			string,
			Infinity,
			StringOrInfinity,
			int,
			int8,
			int16,
			int32,
			int64,
			uint,
			uint8,
			uint16,
			uint32,
			uint64,
			float32,
			float64,
			Reverse[[]byte],
			Reverse[string],
			Reverse[Infinity],
			Reverse[StringOrInfinity],
			Reverse[int],
			Reverse[int8],
			Reverse[int16],
			Reverse[int32],
			Reverse[int64],
			Reverse[uint],
			Reverse[uint8],
			Reverse[uint16],
			Reverse[uint32],
			Reverse[uint64],
			Reverse[float32],
			Reverse[float64],
			Raw:
			continue
		}
	}
	return true
}

// Encode returns the encoding of list.
// If CanEncode(list...) is not true, then Encode panics.
//
// (Note that CanEncode is a property only of the types of the
// values in list, not their values. At any call site passing a
// list of valid concrete types, Encode can be assumed not to panic.)
func Encode(list ...any) []byte {
	return Append(nil, list...)
}

// Append returns the result of appending the encodings of list
// to the existing encoding enc.
// If CanEncode(list...) is not true, then Append panics.
//
// (Note that CanEncode is a property only of the types of the
// values in list, not their values. At any call site passing a
// list of valid concrete types, Append can be assumed not to panic.)
func Append(enc []byte, list ...any) []byte {
	for _, x := range list {
		switch x := x.(type) {
		default:
			panic(fmt.Sprintf("ordered: invalid type %T", x))
		case string:
			enc = appendString(enc, x, 0)
		case Reverse[string]:
			enc = appendString(enc, x.Value(), rev)
		case []byte:
			enc = appendBytes(enc, x, 0)
		case Reverse[[]byte]:
			enc = appendBytes(enc, x.Value(), rev)
		case Infinity:
			enc = append(enc, byte(opInf))
		case Reverse[Infinity]:
			enc = append(enc, byte(opInf)^rev)
		case StringOrInfinity:
			if x.Inf {
				enc = append(enc, byte(opInf))
			} else {
				enc = appendString(enc, x.S, 0)
			}
		case Reverse[StringOrInfinity]:
			if x.Value().Inf {
				enc = append(enc, byte(opInf)^rev)
			} else {
				enc = appendString(enc, x.Value().S, rev)
			}
		case uintptr:
			enc = appendUint64(enc, uint64(x), 0)
		case uint64:
			enc = appendUint64(enc, x, 0)
		case uint32:
			enc = appendUint64(enc, uint64(x), 0)
		case uint16:
			enc = appendUint64(enc, uint64(x), 0)
		case uint8:
			enc = appendUint64(enc, uint64(x), 0)
		case uint:
			enc = appendUint64(enc, uint64(x), 0)
		case Reverse[uintptr]:
			enc = appendUint64(enc, uint64(x.Value()), rev)
		case Reverse[uint64]:
			enc = appendUint64(enc, uint64(x.Value()), rev)
		case Reverse[uint32]:
			enc = appendUint64(enc, uint64(x.Value()), rev)
		case Reverse[uint16]:
			enc = appendUint64(enc, uint64(x.Value()), rev)
		case Reverse[uint8]:
			enc = appendUint64(enc, uint64(x.Value()), rev)
		case Reverse[uint]:
			enc = appendUint64(enc, uint64(x.Value()), rev)
		case int64:
			enc = appendInt64(enc, x, 0)
		case int32:
			enc = appendInt64(enc, int64(x), 0)
		case int16:
			enc = appendInt64(enc, int64(x), 0)
		case int8:
			enc = appendInt64(enc, int64(x), 0)
		case int:
			enc = appendInt64(enc, int64(x), 0)
		case Reverse[int64]:
			enc = appendInt64(enc, int64(x.Value()), rev)
		case Reverse[int32]:
			enc = appendInt64(enc, int64(x.Value()), rev)
		case Reverse[int16]:
			enc = appendInt64(enc, int64(x.Value()), rev)
		case Reverse[int8]:
			enc = appendInt64(enc, int64(x.Value()), rev)
		case Reverse[int]:
			enc = appendInt64(enc, int64(x.Value()), rev)
		case float64:
			enc = appendFloat64(enc, x, 0)
		case float32:
			enc = appendFloat32(enc, x, 0)
		case Reverse[float64]:
			enc = appendFloat64(enc, x.Value(), rev)
		case Reverse[float32]:
			enc = appendFloat32(enc, x.Value(), rev)
		case Raw:
			enc = append(enc, byte(opRaw))
			enc = binary.AppendUvarint(enc, uint64(len(x)))
			enc = append(enc, x...)
		}
	}
	return enc
}

func appendString(out []byte, x string, rev byte) []byte {
	w := len(out)
	out = append(out, byte(opString))
	start := 0
	for i := 0; i < len(x); i++ {
		switch x[i] {
		case 0x00:
			out = append(out, x[start:i]...)
			out = append(out, 0x00, 0xFF)
			start = i + 1
		}
	}
	out = append(out, x[start:]...)
	out = append(out, 0x00, 0x00)
	if rev == 0xFF {
		for ; w < len(out); w++ {
			out[w] ^= 0xFF
		}
	}
	return out
}

func appendBytes(out []byte, x []byte, rev byte) []byte {
	w := len(out)
	out = append(out, byte(opString))
	start := 0
	for i := 0; i < len(x); i++ {
		switch x[i] {
		case 0x00:
			out = append(out, x[start:i]...)
			out = append(out, 0x00, 0xFF)
			start = i + 1
		}
	}
	out = append(out, x[start:]...)
	out = append(out, 0x00, 0x00)
	if rev == 0xFF {
		for ; w < len(out); w++ {
			out[w] ^= 0xFF
		}
	}
	return out
}

func appendUint64(out []byte, x uint64, rev byte) []byte {
	n := (bits.Len64(x) + 7) / 8
	if n == 0 {
		n = 1
	}
	var buf [9]byte
	for i := 0; i < n; i++ {
		buf[8-i] = byte(x) ^ rev
		x >>= 8
	}
	buf[8-n] = (byte(n-1) | byte(opInt|opPosInt)) ^ rev
	out = append(out, buf[8-n:]...)
	return out
}

func appendInt64(out []byte, x int64, rev byte) []byte {
	if x >= 0 {
		return appendUint64(out, uint64(x), rev)
	}
	u := uint64(^x)
	n := (bits.Len64(u) + 7) / 8
	if n == 0 {
		n = 1
	}
	var buf [9]byte
	for i := 0; i < n; i++ {
		buf[8-i] = byte(u) ^ 0xFF ^ rev
		u >>= 8
	}
	buf[8-n] = (byte(n-1) | byte(opInt)) ^ 0xF ^ rev
	out = append(out, buf[8-n:]...)
	return out
}

func appendFloat64(out []byte, f float64, rev byte) []byte {
	out = append(out, byte(opFloat64)^rev)
	u := math.Float64bits(f)
	if f != f {
		u = ^uint64(0) // NaN maps beyond negative infinity
	}
	u ^= 1<<63 | uint64(int64(u)>>63)
	var buf [8]byte
	binary.BigEndian.PutUint64(buf[:], u)
	if rev == 0xFF {
		for i := range buf {
			buf[i] ^= 0xFF
		}
	}
	out = append(out, buf[:]...)
	return out
}

func appendFloat32(out []byte, f float32, rev byte) []byte {
	out = append(out, byte(opFloat32)^rev)
	u := math.Float32bits(f)
	if f != f {
		u = ^uint32(0) // NaN maps beyond negative infinity
	}
	u ^= 1<<31 | uint32(int32(u)>>31)
	var buf [4]byte
	binary.BigEndian.PutUint32(buf[:], u)
	if rev == 0xFF {
		for i := range buf {
			buf[i] ^= 0xFF
		}
	}
	out = append(out, buf[:]...)
	return out
}

var (
	errExtra    = errors.New("ordered: extra encoded data")
	errCorrupt  = errors.New("ordered: invalid encoded data")
	errIntRange = errors.New("ordered int out of range")
)

// Decode decodes enc into a list of typed values.
//
// Each value in list must be a non-nil pointer to an encodable type
// (see [CanEncode]) or else be a non-nil *any
// or else be a nil any.
// When decoding into a *any, the existing value in the any
// is ignored and replaced with one of the types listed in
// the documentation for [DecodeAny].
// Decoding into a nil any discards the corresponding value.
//
// Decode checks that there is no encoded data remaining
// after decoding values into each element of list.
// To decode only a prefix of the encoding, use [DecodePrefix].
func Decode(enc []byte, list ...any) error {
	enc, err := DecodePrefix(enc, list...)
	if err != nil {
		return err
	}
	if len(enc) > 0 {
		return errExtra
	}
	return nil
}

// DecodePrefix decodes a prefix of enc into a list of typed values,
// returning the rest of the encoding and any error encountered.
//
// Each value in list must be a non-nil pointer to an encodable type
// (see [CanEncode]) or else be a non-nil *any
// or else be a nil any.
// When decoding into a *any, the existing value in the any
// is ignored and replaced with one of the types listed in
// the documentation for [DecodeAny].
// Decoding into a nil any discards the corresponding value.
func DecodePrefix(enc []byte, list ...any) (rest []byte, err error) {
	for _, x := range list {
		var err error
		enc, err = decode(enc, x)
		if err != nil {
			return nil, err
		}
	}
	return enc, nil
}

// DecodeAny decodes enc into a []any.
// Each element in the returned slice will have one of the following types:
//
//   - string, for both string and []byte values
//   - Infinity
//   - int64, for integer values representable by int64
//   - uint64, for other very large integer values
//   - float32
//   - float64
//   - Reverse[T] for a type earlier in this list
//   - Raw for a raw byte slice.
func DecodeAny(enc []byte) ([]any, error) {
	var out []any
	for len(enc) > 0 {
		var err error
		var x any
		enc, err = decode(enc, &x)
		if err != nil {
			return nil, err
		}
		out = append(out, x)
	}
	return out, nil
}

// decode decodes one value from enc into x.
func decode(enc []byte, x any) ([]byte, error) {
	op, b, u, iok, f, rest, err := decodeNext(enc)
	if err != nil {
		return nil, err
	}
	enc = rest
Outer:
	switch x := x.(type) {
	default:
		return nil, fmt.Errorf("ordered: invalid type %T", x)
	case nil:
		return enc, nil
	case *any:
		switch op {
		default:
			// unreachable missing case
			break Outer
		case opString:
			*x = string(b)
		case opInf:
			*x = Infinity{}
		case opInt:
			if iok <= 0 {
				*x = int64(u)
			} else {
				*x = u
			}
		case opFloat32:
			*x = float32(f)
		case opFloat64:
			*x = f
		case opString ^ rev:
			*x = Rev(string(b))
		case opInf ^ rev:
			*x = Rev(Infinity{})
		case opInt ^ rev:
			if iok <= 0 {
				*x = Rev(int64(u))
			} else {
				*x = Rev(u)
			}
		case opFloat32 ^ rev:
			*x = Rev(float32(f))
		case opFloat64 ^ rev:
			*x = Rev(f)
		case opRaw:
			*x = Raw(b)
		}
		return enc, nil
	case *string:
		if op == opString {
			*x = string(b)
			return enc, nil
		}
	case *Reverse[string]:
		if op == opString^rev {
			*x = Rev(string(b))
			return enc, nil
		}
	case *[]byte:
		if op == opString {
			*x = b
			return enc, nil
		}
	case *Raw:
		if op == opRaw {
			*x = b
			return enc, nil
		}
	case *Reverse[[]byte]:
		if op == opString^rev {
			*x = Rev(b)
			return enc, nil
		}
	case *Infinity:
		if op == opInf {
			return enc, nil
		}
	case *Reverse[Infinity]:
		if op == opInf^rev {
			return enc, nil
		}
	case *StringOrInfinity:
		if op == opString {
			*x = StringOrInfinity{S: string(b)}
			return enc, nil
		}
		if op == opInf {
			*x = StringOrInfinity{Inf: true}
			return enc, nil
		}
	case *Reverse[StringOrInfinity]:
		if op == opString^rev {
			*x = Rev(StringOrInfinity{S: string(b)})
			return enc, nil
		}
		if op == opInf^rev {
			*x = Rev(StringOrInfinity{Inf: true})
			return enc, nil
		}
	case *uint64:
		if op == opInt {
			if iok >= 0 {
				*x = u
				return enc, nil
			}
			return nil, errIntRange
		}
	case *uintptr:
		if op == opInt {
			if iok >= 0 && uint64(uintptr(u)) == u {
				*x = uintptr(u)
				return enc, nil
			}
			return nil, errIntRange
		}
	case *uint32:
		if op == opInt {
			if iok >= 0 && uint64(uint32(u)) == u {
				*x = uint32(u)
				return enc, nil
			}
			return nil, errIntRange
		}
	case *uint16:
		if op == opInt {
			if iok >= 0 && uint64(uint16(u)) == u {
				*x = uint16(u)
				return enc, nil
			}
			return nil, errIntRange
		}
	case *uint8:
		if op == opInt {
			if iok >= 0 && uint64(uint8(u)) == u {
				*x = uint8(u)
				return enc, nil
			}
			return nil, errIntRange
		}
	case *uint:
		if op == opInt {
			if iok >= 0 && uint64(uint(u)) == u {
				*x = uint(u)
				return enc, nil
			}
			return nil, errIntRange
		}
	case *Reverse[uint64]:
		if op == opInt^rev {
			if iok >= 0 {
				x.v = u
				return enc, nil
			}
			return nil, errIntRange
		}
	case *Reverse[uintptr]:
		if op == opInt^rev {
			if iok >= 0 && uint64(uintptr(u)) == u {
				x.v = uintptr(u)
				return enc, nil
			}
			return nil, errIntRange
		}
	case *Reverse[uint32]:
		if op == opInt^rev {
			if iok >= 0 && uint64(uint32(u)) == u {
				x.v = uint32(u)
				return enc, nil
			}
			return nil, errIntRange
		}
	case *Reverse[uint16]:
		if op == opInt^rev {
			if iok >= 0 && uint64(uint16(u)) == u {
				x.v = uint16(u)
				return enc, nil
			}
			return nil, errIntRange
		}
	case *Reverse[uint8]:
		if op == opInt^rev {
			if iok >= 0 && uint64(uint8(u)) == u {
				x.v = uint8(u)
				return enc, nil
			}
			return nil, errIntRange
		}
	case *Reverse[uint]:
		if op == opInt^rev {
			if iok >= 0 && uint64(uint(u)) == u {
				x.v = uint(u)
				return enc, nil
			}
			return nil, errIntRange
		}
	case *int64:
		if op == opInt {
			if iok <= 0 {
				*x = int64(u)
				return enc, nil
			}
			return nil, errIntRange
		}
	case *int32:
		if op == opInt {
			if iok <= 0 && int64(int32(u)) == int64(u) {
				*x = int32(u)
				return enc, nil
			}
			return nil, errIntRange
		}
	case *int16:
		if op == opInt {
			if iok <= 0 && int64(int16(u)) == int64(u) {
				*x = int16(u)
				return enc, nil
			}
			return nil, errIntRange
		}
	case *int8:
		if op == opInt {
			if iok <= 0 && int64(int8(u)) == int64(u) {
				*x = int8(u)
				return enc, nil
			}
			return nil, errIntRange
		}
	case *int:
		if op == opInt {
			if iok <= 0 && int64(int(u)) == int64(u) {
				*x = int(u)
				return enc, nil
			}
			return nil, errIntRange
		}
	case *Reverse[int64]:
		if op == opInt^rev {
			if iok <= 0 {
				x.v = int64(u)
				return enc, nil
			}
			return nil, errIntRange
		}
	case *Reverse[int32]:
		if op == opInt^rev {
			if iok <= 0 && int64(int32(u)) == int64(u) {
				x.v = int32(u)
				return enc, nil
			}
			return nil, errIntRange
		}
	case *Reverse[int16]:
		if op == opInt^rev {
			if iok <= 0 && int64(int16(u)) == int64(u) {
				x.v = int16(u)
				return enc, nil
			}
			return nil, errIntRange
		}
	case *Reverse[int8]:
		if op == opInt^rev {
			if iok <= 0 && int64(int8(u)) == int64(u) {
				x.v = int8(u)
				return enc, nil
			}
			return nil, errIntRange
		}
	case *Reverse[int]:
		if op == opInt^rev {
			if iok <= 0 && int64(int(u)) == int64(u) {
				x.v = int(u)
				return enc, nil
			}
			return nil, errIntRange
		}
	case *float64:
		if op == opFloat64 {
			*x = f
			return enc, nil
		}
	case *float32:
		if op == opFloat32 {
			*x = float32(f)
			return enc, nil
		}
	case *Reverse[float64]:
		if op == opFloat64^rev {
			*x = Rev(f)
			return enc, nil
		}
	case *Reverse[float32]:
		if op == opFloat32^rev {
			*x = Rev(float32(f))
			return enc, nil
		}
	}
	switch op {
	case
		opString,
		opFloat32,
		opFloat64,
		opInt + opPosInt - 8,
		opInt + opPosInt - 7,
		opInt + opPosInt - 6,
		opInt + opPosInt - 5,
		opInt + opPosInt - 4,
		opInt + opPosInt - 3,
		opInt + opPosInt - 2,
		opInt + opPosInt - 1,
		opInt + opPosInt + 0,
		opInt + opPosInt + 1,
		opInt + opPosInt + 2,
		opInt + opPosInt + 3,
		opInt + opPosInt + 4,
		opInt + opPosInt + 5,
		opInt + opPosInt + 6,
		opInt + opPosInt + 7,
		opInf,
		rev ^ opString,
		rev ^ opFloat32,
		rev ^ opFloat64,
		rev ^ (opInt + opPosInt - 8),
		rev ^ (opInt + opPosInt - 7),
		rev ^ (opInt + opPosInt - 6),
		rev ^ (opInt + opPosInt - 5),
		rev ^ (opInt + opPosInt - 4),
		rev ^ (opInt + opPosInt - 3),
		rev ^ (opInt + opPosInt - 2),
		rev ^ (opInt + opPosInt - 1),
		rev ^ (opInt + opPosInt + 0),
		rev ^ (opInt + opPosInt + 1),
		rev ^ (opInt + opPosInt + 2),
		rev ^ (opInt + opPosInt + 3),
		rev ^ (opInt + opPosInt + 4),
		rev ^ (opInt + opPosInt + 5),
		rev ^ (opInt + opPosInt + 6),
		rev ^ (opInt + opPosInt + 7),
		rev ^ opInf:
		// unreachable missing case: don't say errCorrupt
		return nil, fmt.Errorf("cannot parse %s into %T", op, x)
	}
	return nil, errCorrupt
}

var (
	endString    = []byte{0x00, 0x00}
	endStringRev = []byte{0x00 ^ rev, 0x00 ^ rev}
)

func decodeNext(enc []byte) (op opcode, b []byte, u uint64, iok int, f float64, rest []byte, err error) {
	if len(enc) == 0 {
		err = errCorrupt
		return
	}
	e0 := opcode(enc[0])
	op = e0
	if op&0xE0 == 0x20 { // opInt
		op &= 0xE0
	}
	if ^op&0xE0 == 0x20 { // opInt ^ rev
		op |= 0x1F
	}
	enc = enc[1:]
	switch op {
	default:
		err = errCorrupt
		return

	case opInf, opInf ^ rev:
		rest = enc

	case opString:
		i := bytes.Index(enc, endString)
		if i < 0 {
			err = errCorrupt
			return
		}
		enc, rest = enc[:i], enc[i+len(endString):]
		start := 0
		for i := 0; i < len(enc); i++ {
			if enc[i] == 0x00 {
				b = append(b, enc[start:i]...)
				// Note: i+1 >= len(enc) is impossible,
				// because we cut enc at the first 00 00,
				// so enc cannot end in 00.
				// But check bounds anyway in case that changes.
				if i+1 >= len(enc) || enc[i+1] != 0xFF {
					err = errCorrupt
					return
				}
				b = append(b, 0)
				start = i + 2
				i++
			}
		}
		b = append(b, enc[start:]...)

	case opString ^ rev:
		i := bytes.Index(enc, endStringRev)
		if i < 0 {
			err = errCorrupt
			return
		}
		enc, rest = enc[:i], enc[i+len(endString):]
		for i := 0; i < len(enc); i++ {
			if enc[i] == 0x00^0xFF {
				// Note: i+1 >= len(enc) is impossible,
				// because we cut enc at the first FF FF,
				// so enc cannot end in FF.
				// But check bounds anyway in case that changes.
				if i+1 >= len(enc) || enc[i+1] != 0xFF^0xFF {
					err = errCorrupt
					return
				}
				b = append(b, 0)
				i++
			} else {
				b = append(b, enc[i]^rev)
			}
		}

	case opRaw:
		n, w := binary.Uvarint(enc)
		if w <= 0 || n > uint64(len(enc)-w) {
			err = errCorrupt
			return
		}
		b, rest = enc[w:w+int(n)], enc[w+int(n):]

	case opInt, opInt ^ rev:
		d := byte(0)
		if op == opInt^rev {
			d = rev
			e0 = ^e0
		}
		n := int(e0 & 0x0F)
		pos := e0&opPosInt != 0
		if !pos {
			u = ^uint64(0)
			n ^= 0xF
		}
		n++
		if n > 8 {
			err = errCorrupt
			return
		}
		if len(enc) < n {
			err = errCorrupt
			return
		}
		enc, rest = enc[:n], enc[n:]
		if len(enc) > 1 && (pos && enc[0]^d == 0 || !pos && enc[0]^d == 0xff) {
			err = errCorrupt
			return
		}
		for _, x := range enc {
			u = u<<8 | uint64(x^d)
		}
		if !pos {
			u = ^u
		}
		if pos && u < 1<<63 {
			iok = 0
		} else if pos {
			iok = 1
		} else {
			if u >= 1<<63 {
				err = errCorrupt
				return
			}
			u = ^u
			iok = -1
		}

	case opFloat64, opFloat64 ^ rev:
		if len(enc) < 8 {
			err = errCorrupt
			return
		}
		rest = enc[8:]
		u := binary.BigEndian.Uint64(enc[:])
		if op == opFloat64^rev {
			u = ^u
		}
		u ^= 1 << 63
		u ^= uint64(int64(u)>>63) >> 1
		f = math.Float64frombits(u)

	case opFloat32, opFloat32 ^ rev:
		if len(enc) < 4 {
			err = errCorrupt
			return
		}
		rest = enc[4:]
		u := binary.BigEndian.Uint32(enc[:])
		if op == opFloat32^rev {
			u = ^u
		}
		u ^= 1 << 31
		u ^= uint32(int32(u)>>31) >> 1
		f = float64(math.Float32frombits(u))
	}
	return
}

// DecodeFmt decodes enc into a parenthesized, comma-separated list of textual values.
// Each value in the list will have one of the following formats:
//
//   - double-quoted string, for both string and []byte values
//   - "Inf", for Infinity
//   - decimal or negative decimal, for integer values.
//   - "float32(f)" for float32 values; f may be NaN or +Inf or -Inf
//   - "float64(f)" for float64 values; f may be NaN or +Inf or -Inf
//   - "Rev(x)" for a reverse of formatted value x earlier in this list
//   - "Raw(q)", where q is a double-quoted string, for a Raw value.
func DecodeFmt(enc []byte) (string, error) {
	list, err := DecodeAny(enc)
	if err != nil {
		return "", err
	}
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "(")
	for i, x := range list {
		if i > 0 {
			fmt.Fprintf(&buf, ", ")
		}
		var rx any
		switch x := x.(type) {
		case Reverse[string]:
			rx = x.Value()
		case Reverse[Infinity]:
			rx = x.Value()
		case Reverse[int64]:
			rx = x.Value()
		case Reverse[uint64]:
			rx = x.Value()
		case Reverse[float32]:
			rx = x.Value()
		case Reverse[float64]:
			rx = x.Value()
		}
		end := ""
		if rx != nil {
			fmt.Fprintf(&buf, "Rev(")
			end = ")"
			x = rx
		}
		switch x := x.(type) {
		case string:
			fmt.Fprintf(&buf, "%q", x)
		case Raw:
			fmt.Fprintf(&buf, "Raw(%q)", []byte(x))
		case Infinity:
			fmt.Fprintf(&buf, "Inf")
		case int64:
			fmt.Fprintf(&buf, "%d", x)
		case uint64:
			fmt.Fprintf(&buf, "%d", x)
		case float32:
			fmt.Fprintf(&buf, "float32(%v)", x)
		case float64:
			fmt.Fprintf(&buf, "float64(%v)", x)
		}
		fmt.Fprintf(&buf, "%s", end)
	}
	fmt.Fprintf(&buf, ")")
	return buf.String(), nil
}
