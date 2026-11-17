// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package toolcallintegrity

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"math/big"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/session"
)

// The stamp payload is a persisted format shared with adk-python
// (plugins/_tool_call_integrity_plugin.py, _canonical_payload): a stamp
// written by either language must verify in the other. It is the bytes of
// json.dumps(_normalize(value), sort_keys=True, separators=(",", ":"),
// ensure_ascii=False) over the JSON form of the call's args. encoding/json
// cannot write that form (it escapes U+2028 and U+2029, writes 1e-7 for
// Python's 1e-07, and refuses NaN), so the writer below is the plugin's own.

// stampPayload returns the bytes a stamp of fc in ev covers: the session's
// app name, user ID and ID, the event's invocation ID, branch and author, and
// the call's name, ID and args. It fails only when the args cannot be encoded.
func stampPayload(sess session.Session, ev *session.Event, fc *genai.FunctionCall) ([]byte, error) {
	var b bytes.Buffer
	b.WriteString(`{"app_name":`)
	writeString(&b, sess.AppName())
	b.WriteString(`,"args":`)
	if err := writeArgs(&b, fc.Args); err != nil {
		return nil, encodeError(fc.Name, fc.ID, err)
	}
	b.WriteString(`,"author":`)
	writeString(&b, ev.Author)
	b.WriteString(`,"branch":`)
	writeOptionalString(&b, ev.Branch)
	b.WriteString(`,"id":`)
	writeString(&b, fc.ID)
	b.WriteString(`,"invocation_id":`)
	writeString(&b, ev.InvocationID)
	b.WriteString(`,"name":`)
	writeOptionalString(&b, fc.Name)
	b.WriteString(`,"session_id":`)
	writeString(&b, sess.ID())
	b.WriteString(`,"user_id":`)
	writeString(&b, sess.UserID())
	b.WriteByte('}')
	return b.Bytes(), nil
}

// canonicalArgs returns the canonical JSON of args alone, as the payload
// holds them, for comparing the args a tool runs with against a stored call.
// The caller wraps its error with encodeError.
func canonicalArgs(args map[string]any) ([]byte, error) {
	var b bytes.Buffer
	if err := writeArgs(&b, args); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

// encodeError reports args that cannot be encoded. It does not match
// ErrIntegrity: nothing was tampered with, and adk-python raises pydantic's
// serialization error here, not ToolCallIntegrityError. It names the Go type
// of the failing value and never the value, which is tool input.
func encodeError(name, id string, err error) error {
	return fmt.Errorf("toolcallintegrity: cannot encode the args of %s call '%s': %s", name, id, err.Error())
}

// notJSON is the error of a value writeValue cannot encode.
func notJSON(v any) error {
	return fmt.Errorf("value of type %T is not JSON", v)
}

// writeArgs writes a call's args; missing args are {} (adk-python: args or {}).
func writeArgs(b *bytes.Buffer, args map[string]any) error {
	if len(args) == 0 {
		b.WriteString("{}")
		return nil
	}
	return writeValue(b, args)
}

// writeOptionalString writes s, or null when it is empty: genai and session
// store an empty name and branch with omitempty, so adk-python reads them as
// None.
func writeOptionalString(b *bytes.Buffer, s string) {
	if s == "" {
		b.WriteString("null")
		return
	}
	writeString(b, s)
}

// writeValue writes v by its dynamic type. A value that is not one of the
// types a JSON store returns is written in the form encoding/json stores it
// in, so the stamp still verifies after a store round trip.
func writeValue(b *bytes.Buffer, v any) error {
	switch v := v.(type) {
	case nil:
		b.WriteString("null")
	case bool:
		b.WriteString(strconv.FormatBool(v))
	case string:
		writeString(b, v)
	case float64:
		writeFloat(b, v)
	case json.Number:
		return writeNumber(b, v)
	case map[string]any:
		if v == nil {
			b.WriteString("null")
			return nil
		}
		b.WriteByte('{')
		for i, k := range slices.Sorted(maps.Keys(v)) {
			if i > 0 {
				b.WriteByte(',')
			}
			writeString(b, k)
			b.WriteByte(':')
			if err := writeValue(b, v[k]); err != nil {
				return err
			}
		}
		b.WriteByte('}')
	case []any:
		if v == nil {
			b.WriteString("null")
			return nil
		}
		b.WriteByte('[')
		for i, e := range v {
			if i > 0 {
				b.WriteByte(',')
			}
			if err := writeValue(b, e); err != nil {
				return err
			}
		}
		b.WriteByte(']')
	default:
		return writeStoredForm(b, v)
	}
	return nil
}

// writeStoredForm writes v as encoding/json stores it, decoded the way a
// store hands it back (numbers as json.Number).
func writeStoredForm(b *bytes.Buffer, v any) error {
	data, err := json.Marshal(v)
	if err != nil {
		return notJSON(v)
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	var decoded any
	if err := dec.Decode(&decoded); err != nil {
		return notJSON(v)
	}
	return writeValue(b, decoded)
}

// writeNumber writes a JSON number literal as adk-python writes the value
// json.loads gives for it: an integer literal in full, any other literal as
// the float it denotes. A string that is not a JSON number is an error.
func writeNumber(b *bytes.Buffer, n json.Number) error {
	s := string(n)
	isInt, ok := numberForm(s)
	if !ok {
		return notJSON(n)
	}
	if isInt {
		i, _ := new(big.Int).SetString(s, 10)
		b.WriteString(i.String())
		return nil
	}
	// Out of range, ParseFloat returns ±Inf or 0 with ErrRange, which is
	// what json.loads gives in Python, so the error is not one.
	f, _ := strconv.ParseFloat(s, 64)
	writeFloat(b, f)
	return nil
}

// numberForm reports whether s is a number by the grammar of RFC 8259,
// section 6, and whether it is an integer literal (no fraction, no exponent).
func numberForm(s string) (isInt, ok bool) {
	i := 0
	if i < len(s) && s[i] == '-' {
		i++
	}
	switch {
	case i < len(s) && s[i] == '0':
		i++
	case i < len(s) && s[i] >= '1' && s[i] <= '9':
		i = skipDigits(s, i)
	default:
		return false, false
	}
	isInt = true
	if i < len(s) && s[i] == '.' {
		isInt = false
		j := skipDigits(s, i+1)
		if j == i+1 {
			return false, false
		}
		i = j
	}
	if i < len(s) && (s[i] == 'e' || s[i] == 'E') {
		isInt = false
		i++
		if i < len(s) && (s[i] == '+' || s[i] == '-') {
			i++
		}
		j := skipDigits(s, i)
		if j == i {
			return false, false
		}
		i = j
	}
	return isInt, i == len(s)
}

func skipDigits(s string, i int) int {
	for i < len(s) && s[i] >= '0' && s[i] <= '9' {
		i++
	}
	return i
}

// writeFloat writes f as adk-python does: NaN and the infinities as null
// (pydantic's JSON mode), a whole number as the exact integer it equals, and
// any other value as Python's repr.
func writeFloat(b *bytes.Buffer, f float64) {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		b.WriteString("null")
		return
	}
	if f == math.Trunc(f) {
		i, _ := big.NewFloat(f).Int(nil)
		b.WriteString(i.String())
		return
	}
	// Python's repr uses the exponent form when the decimal exponent is below
	// -4 (a non-integer never reaches its upper switch point, 16). Go also
	// writes at least two exponent digits, as Python does.
	s := strconv.FormatFloat(f, 'e', -1, 64)
	exp, _ := strconv.Atoi(s[strings.IndexByte(s, 'e')+1:])
	if exp < -4 {
		b.WriteString(s)
		return
	}
	b.WriteString(strconv.FormatFloat(f, 'f', -1, 64))
}

// writeString writes s quoted as Python's json.dumps with ensure_ascii=False
// does: only the quote, the backslash and U+0000 to U+001F are escaped, and
// invalid UTF-8 becomes U+FFFD.
func writeString(b *bytes.Buffer, s string) {
	const hex = "0123456789abcdef"
	b.WriteByte('"')
	for i := 0; i < len(s); {
		r, size := utf8.DecodeRuneInString(s[i:])
		i += size
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		default:
			if r < 0x20 {
				b.WriteString(`\u00`)
				b.WriteByte(hex[r>>4])
				b.WriteByte(hex[r&0xf])
				continue
			}
			b.WriteRune(r)
		}
	}
	b.WriteByte('"')
}
