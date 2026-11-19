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

package agentanalytics_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"iter"
	"net/http"
	"path"
	"reflect"
	"strings"
	"testing"
	"time"
	"unsafe"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/adk/plugin/agentanalytics"
)

const agentAnalyticsPkgPath = "google.golang.org/adk/plugin/agentanalytics"

func TestStructLiterals_ExternalKeyedComparableAndJSON(t *testing.T) {
	retryA := agentanalytics.RetryConfig{
		MaxRetries:   3,
		InitialDelay: 100 * time.Millisecond,
		MaxDelay:     10 * time.Second,
		Multiplier:   2.0,
	}
	retryB := agentanalytics.RetryConfig{
		MaxRetries:   3,
		InitialDelay: 100 * time.Millisecond,
		MaxDelay:     10 * time.Second,
		Multiplier:   2.0,
	}
	retryC := agentanalytics.RetryConfig{
		MaxRetries:   4,
		InitialDelay: 100 * time.Millisecond,
		MaxDelay:     10 * time.Second,
		Multiplier:   2.0,
	}
	if retryA != retryB {
		t.Fatalf("equal RetryConfig values compared unequal: %#v != %#v", retryA, retryB)
	}
	if retryA == retryC {
		t.Fatalf("distinct RetryConfig values compared equal: %#v == %#v", retryA, retryC)
	}

	dropsA := agentanalytics.DropStats{
		QueueFullDrops:        1,
		AppendFailureDrops:    2,
		RowErrorDrops:         5,
		FormatterFailureDrops: 3,
		ShutdownDrops:         4,
		TotalDrops:            12,
	}
	dropsB := agentanalytics.DropStats{
		QueueFullDrops:        1,
		AppendFailureDrops:    2,
		RowErrorDrops:         5,
		FormatterFailureDrops: 3,
		ShutdownDrops:         4,
		TotalDrops:            12,
	}
	dropsC := agentanalytics.DropStats{
		QueueFullDrops:        1,
		AppendFailureDrops:    2,
		RowErrorDrops:         5,
		FormatterFailureDrops: 3,
		ShutdownDrops:         4,
		TotalDrops:            13,
	}
	if dropsA != dropsB {
		t.Fatalf("equal DropStats values compared unequal: %#v != %#v", dropsA, dropsB)
	}
	if dropsA == dropsC {
		t.Fatalf("distinct DropStats values compared equal: %#v == %#v", dropsA, dropsC)
	}

	gotJSON, err := json.Marshal(dropsA)
	if err != nil {
		t.Fatalf("json.Marshal(DropStats) failed: %v", err)
	}
	const wantJSON = `{"queue_full_drops":1,"append_failure_drops":2,"row_error_drops":5,"formatter_failure_drops":3,"shutdown_drops":4,"total_drops":12}`
	if string(gotJSON) != wantJSON {
		t.Fatalf("json.Marshal(DropStats) = %s, want %s", gotJSON, wantJSON)
	}
	var roundTripDrops agentanalytics.DropStats
	if err := json.Unmarshal(gotJSON, &roundTripDrops); err != nil {
		t.Fatalf("json.Unmarshal(DropStats) failed: %v", err)
	}
	if roundTripDrops != dropsA {
		t.Fatalf("json.Unmarshal(DropStats) = %#v, want %#v", roundTripDrops, dropsA)
	}

	uploadReqA := agentanalytics.GCSUploadRequest{
		Path:        "prefix/2026-11-07/obj.txt",
		Data:        []byte("payload"),
		ContentType: "text/plain; charset=utf-8",
	}
	uploadReqB := agentanalytics.GCSUploadRequest{
		Path:        "prefix/2026-11-07/obj.txt",
		Data:        []byte("payload"),
		ContentType: "text/plain; charset=utf-8",
	}
	uploadReqC := agentanalytics.GCSUploadRequest{
		Path:        "prefix/2026-11-07/obj.txt",
		Data:        []byte("other"),
		ContentType: "text/plain; charset=utf-8",
	}
	if uploadReqA.Path != "prefix/2026-11-07/obj.txt" || !bytes.Equal(uploadReqA.Data, []byte("payload")) || uploadReqA.ContentType != "text/plain; charset=utf-8" {
		t.Fatalf("unexpected GCSUploadRequest fields: %#v", uploadReqA)
	}

	viewA := agentanalytics.ViewDefinition{
		Name:      "v_llm_request",
		EventType: agentanalytics.EventLLMRequest,
		Query:     "SELECT 1",
	}
	viewB := agentanalytics.ViewDefinition{
		Name:      "v_llm_request",
		EventType: agentanalytics.EventLLMRequest,
		Query:     "SELECT 1",
	}
	viewC := agentanalytics.ViewDefinition{
		Name:      "v_llm_response",
		EventType: agentanalytics.EventLLMResponse,
		Query:     "SELECT 1",
	}
	if viewA != viewB {
		t.Fatalf("equal ViewDefinition values compared unequal: %#v != %#v", viewA, viewB)
	}
	if viewA == viewC {
		t.Fatalf("distinct ViewDefinition values compared equal: %#v == %#v", viewA, viewC)
	}

	// Pin cmp.Equal without options on the four exported value structs. Blank
	// `_ struct{}` guards are ignored by go-cmp, whereas adding a named
	// unexported field to any of these structs would make cmp.Equal panic for
	// external callers.
	t.Run("CmpEqual", func(t *testing.T) {
		if !cmp.Equal(retryA, retryB) {
			t.Errorf("cmp.Equal(retryA, retryB) = false, want true")
		}
		if cmp.Equal(retryA, retryC) {
			t.Errorf("cmp.Equal(retryA, retryC) = true, want false")
		}
		if !cmp.Equal(dropsA, dropsB) {
			t.Errorf("cmp.Equal(dropsA, dropsB) = false, want true")
		}
		if cmp.Equal(dropsA, dropsC) {
			t.Errorf("cmp.Equal(dropsA, dropsC) = true, want false")
		}
		if !cmp.Equal(viewA, viewB) {
			t.Errorf("cmp.Equal(viewA, viewB) = false, want true")
		}
		if cmp.Equal(viewA, viewC) {
			t.Errorf("cmp.Equal(viewA, viewC) = true, want false")
		}
		if !cmp.Equal(uploadReqA, uploadReqB) {
			t.Errorf("cmp.Equal(uploadReqA, uploadReqB) = false, want true")
		}
		if cmp.Equal(uploadReqA, uploadReqC) {
			t.Errorf("cmp.Equal(uploadReqA, uploadReqC) = true, want false")
		}
	})

	cfg := agentanalytics.Config{
		Enabled:        false,
		ProjectID:      "test-project",
		DatasetID:      "test_dataset",
		TableName:      "agent_events",
		RetryConfig:    retryA,
		ViewPrefix:     "v",
		CustomTags:     map[string]any{"env": "test"},
		EventAllowlist: []string{agentanalytics.EventLLMRequest},
	}
	defs, err := cfg.ViewDefinitions()
	if err != nil {
		t.Fatalf("Config.ViewDefinitions() failed: %v", err)
	}
	if len(defs) == 0 {
		t.Fatal("Config.ViewDefinitions() returned no views")
	}

	if agentanalytics.SchemaVersion == "" {
		t.Fatal("SchemaVersion must not be empty")
	}
	if len(agentanalytics.EventsSchema()) == 0 {
		t.Fatal("EventsSchema() returned empty schema")
	}
	defaultCfg := agentanalytics.DefaultConfig()
	if !defaultCfg.Enabled {
		t.Fatal("DefaultConfig().Enabled = false, want true")
	}

	p, err := agentanalytics.NewBigQueryAgentAnalyticsPluginWithConfig(context.Background(), cfg)
	if err != nil {
		t.Fatalf("NewBigQueryAgentAnalyticsPluginWithConfig(disabled) failed: %v", err)
	}
	bqp, ok := agentanalytics.AsBigQueryAgentAnalyticsPlugin(p)
	if !ok || bqp == nil {
		t.Fatal("AsBigQueryAgentAnalyticsPlugin returned false for plugin constructed by package")
	}
	if got := bqp.DropStats(); got != (agentanalytics.DropStats{}) {
		t.Fatalf("disabled plugin DropStats() = %#v, want zero", got)
	}
	if err := p.Close(); err != nil {
		t.Fatalf("disabled plugin Close() failed: %v", err)
	}
}

// guardedStruct is a struct whose first field, `_ struct{}`, keeps code in
// other packages from writing positional (unkeyed) literals of it, so that
// adding a field cannot break such code. name is the exported identifier in
// package agentanalytics (recorded explicitly rather than via rt.Name() so an
// exported alias over an unexported struct still probes the exported name).
type guardedStruct struct {
	name string
	typ  reflect.Type
}

// guardedStructs returns the package's guarded structs with their explicit
// exported identifiers.
func guardedStructs() []guardedStruct {
	return []guardedStruct{
		{name: "RetryConfig", typ: reflect.TypeFor[agentanalytics.RetryConfig]()},
		{name: "DropStats", typ: reflect.TypeFor[agentanalytics.DropStats]()},
		{name: "GCSUploadRequest", typ: reflect.TypeFor[agentanalytics.GCSUploadRequest]()},
		{name: "ViewDefinition", typ: reflect.TypeFor[agentanalytics.ViewDefinition]()},
		{name: "Config", typ: reflect.TypeFor[agentanalytics.Config]()},
	}
}

// checkGuardLayout returns an error unless field 0 of gs is the unexported
// blank `_ struct{}` guard of gs's own package at byte offset 0 tagged with
// `json:"-"` and gs has at least one exported field after it. Additional
// unexported fields after field 0 are permitted here for non-value structs such
// as Config (which already carries function and interface fields); the plain
// value structs (RetryConfig, DropStats, GCSUploadRequest, and ViewDefinition)
// are pinned separately with cmp.Equal so named unexported fields cannot break
// caller comparisons.
func checkGuardLayout(gs guardedStruct) error {
	rt := gs.typ
	if rt.NumField() == 0 {
		return fmt.Errorf("%s has no fields, want `_ struct{} `json:\"-\"`` guard at field 0", gs.name)
	}
	guard := rt.Field(0)
	if guard.Name != "_" || guard.Anonymous || guard.IsExported() || guard.PkgPath != rt.PkgPath() || guard.Type != reflect.TypeFor[struct{}]() || guard.Offset != 0 || guard.Tag.Get("json") != "-" {
		return fmt.Errorf("%s field 0 = {Name:%q PkgPath:%q Type:%v Offset:%d Anonymous:%v JSONTag:%q}, want unexported `_ struct{}` of %q at offset 0 with `json:\"-\"`",
			gs.name, guard.Name, guard.PkgPath, guard.Type, guard.Offset, guard.Anonymous, guard.Tag.Get("json"), rt.PkgPath())
	}
	for i := 1; i < rt.NumField(); i++ {
		if rt.Field(i).IsExported() {
			return nil
		}
	}
	return fmt.Errorf("%s has no exported fields after the guard", gs.name)
}

func checkGuardStructName(gs guardedStruct) error {
	if gs.typ.PkgPath() != agentAnalyticsPkgPath || gs.typ.Name() != gs.name {
		return fmt.Errorf("guardedStructs entry %q has reflect type %q.%q, want %q.%q",
			gs.name, gs.typ.PkgPath(), gs.typ.Name(), agentAnalyticsPkgPath, gs.name)
	}
	return nil
}

func TestStructLiterals_ExternalPositionalRejected(t *testing.T) {
	structs := guardedStructs()
	for _, gs := range structs {
		t.Run(gs.name+"_GuardLayout", func(t *testing.T) {
			if err := checkGuardStructName(gs); err != nil {
				t.Fatal(err)
			}
			if err := checkGuardLayout(gs); err != nil {
				t.Error(err)
			}
		})
	}
	t.Run("TypoNameRejectedInProcess", func(t *testing.T) {
		typo := guardedStruct{name: "DropStatz", typ: reflect.TypeFor[agentanalytics.DropStats]()}
		if err := checkGuardStructName(typo); err == nil {
			t.Fatal("checkGuardStructName(DropStatz) = nil, want mismatch error")
		}
	})

	// Type-check against the struct types compiled into this test binary via
	// reflect so in-memory edits (including `go test -overlay`) are checked
	// hermetic in-process without shelling out to `go list`.
	probePkg, err := probePackage(structs)
	if err != nil {
		t.Fatalf("probePackage() failed: %v", err)
	}
	for _, gs := range structs {
		t.Run(gs.name+"_ExternalLiterals", func(t *testing.T) {
			for _, problem := range checkExternalPositional(probePkg, gs) {
				t.Error(problem)
			}
		})
	}
}

func TestRetryConfig_ValidationErrorIncludesAllExportedFields(t *testing.T) {
	cfg := agentanalytics.Config{
		Enabled:   true,
		ProjectID: "test-project",
		RetryConfig: agentanalytics.RetryConfig{
			MaxRetries:   -1,
			InitialDelay: 100 * time.Millisecond,
			MaxDelay:     10 * time.Second,
			Multiplier:   2.0,
		},
	}
	_, err := agentanalytics.NewBigQueryAgentAnalyticsPluginWithConfig(context.Background(), cfg)
	if err == nil {
		t.Fatal("NewBigQueryAgentAnalyticsPluginWithConfig with negative MaxRetries = nil, want error")
	}
	msg := err.Error()
	if strings.Contains(msg, "_:") || strings.Contains(msg, "struct {}") {
		t.Fatalf("RetryConfig validation error leaked guard field: %q", msg)
	}
	for sf := range reflect.TypeFor[agentanalytics.RetryConfig]().Fields() {
		if !sf.IsExported() {
			continue
		}
		if !strings.Contains(msg, sf.Name+":") {
			t.Errorf("RetryConfig validation error %q does not include exported field %q", msg, sf.Name)
		}
	}
}

func TestGuardedStructs_JSONTagAndReflectionEncoderOmitsGuard(t *testing.T) {
	// Mirror reflection encoders (such as Agent Engine streaming's ConvertSnake)
	// that iterate all struct fields including unexported ones and only skip
	// fields carrying `json:"-"`.
	reflectEncodeKeys := func(v any) ([]string, error) {
		rt := reflect.TypeOf(v)
		var keys []string
		for sf := range rt.Fields() {
			if sf.Tag.Get("json") == "-" {
				continue
			}
			if !sf.IsExported() {
				return nil, fmt.Errorf("%s includes unexported field %q without `json:\"-\"`", rt.Name(), sf.Name)
			}
			keys = append(keys, sf.Name)
		}
		return keys, nil
	}

	for _, gs := range guardedStructs() {
		t.Run(gs.name+"_ReflectEncoderSkipsGuard", func(t *testing.T) {
			if got := gs.typ.Field(0).Tag.Get("json"); got != "-" {
				t.Fatalf("%s field 0 json tag = %q, want %q", gs.name, got, "-")
			}
			v := reflect.Zero(gs.typ).Interface()
			keys, err := reflectEncodeKeys(v)
			if err != nil {
				t.Fatal(err)
			}
			for _, k := range keys {
				if k == "_" {
					t.Fatalf("%s reflection encoder emitted %q key", gs.name, k)
				}
			}
		})
	}

	// Verify encoding/json marshaling produces no "_" key and decodes cleanly
	// with DisallowUnknownFields into structs matching the exported fields.
	retry := agentanalytics.RetryConfig{
		MaxRetries:   3,
		InitialDelay: 100 * time.Millisecond,
		MaxDelay:     10 * time.Second,
		Multiplier:   2.0,
	}
	drops := agentanalytics.DropStats{
		QueueFullDrops:        1,
		AppendFailureDrops:    2,
		RowErrorDrops:         3,
		FormatterFailureDrops: 4,
		ShutdownDrops:         5,
		TotalDrops:            11,
	}
	upload := agentanalytics.GCSUploadRequest{
		Path:        "prefix/obj.txt",
		Data:        []byte("payload"),
		ContentType: "text/plain",
	}
	view := agentanalytics.ViewDefinition{
		Name:      "v_llm_request",
		EventType: agentanalytics.EventLLMRequest,
		Query:     "SELECT 1",
	}

	for _, tc := range []struct {
		name string
		val  any
		dst  any
	}{
		{name: "RetryConfig", val: retry, dst: &agentanalytics.RetryConfig{}},
		{name: "DropStats", val: drops, dst: &agentanalytics.DropStats{}},
		{name: "GCSUploadRequest", val: upload, dst: &agentanalytics.GCSUploadRequest{}},
		{name: "ViewDefinition", val: view, dst: &agentanalytics.ViewDefinition{}},
	} {
		t.Run(tc.name+"_StrictJSON", func(t *testing.T) {
			raw, err := json.Marshal(tc.val)
			if err != nil {
				t.Fatalf("json.Marshal(%s) failed: %v", tc.name, err)
			}
			var m map[string]any
			if err := json.Unmarshal(raw, &m); err != nil {
				t.Fatalf("json.Unmarshal(%s to map) failed: %v", tc.name, err)
			}
			if _, hasGuard := m["_"]; hasGuard {
				t.Fatalf("json.Marshal(%s) emitted %q key in %s", tc.name, "_", raw)
			}
			dec := json.NewDecoder(bytes.NewReader(raw))
			dec.DisallowUnknownFields()
			if err := dec.Decode(tc.dst); err != nil {
				t.Fatalf("strict json.Decode(%s) failed: %v (payload %s)", tc.name, err, raw)
			}
		})
	}
}

// basicKinds maps the kinds of the predeclared boolean, numeric and string
// types to their go/types counterparts.
var basicKinds = map[reflect.Kind]types.BasicKind{
	reflect.Bool:       types.Bool,
	reflect.Int:        types.Int,
	reflect.Int8:       types.Int8,
	reflect.Int16:      types.Int16,
	reflect.Int32:      types.Int32,
	reflect.Int64:      types.Int64,
	reflect.Uint:       types.Uint,
	reflect.Uint8:      types.Uint8,
	reflect.Uint16:     types.Uint16,
	reflect.Uint32:     types.Uint32,
	reflect.Uint64:     types.Uint64,
	reflect.Uintptr:    types.Uintptr,
	reflect.Float32:    types.Float32,
	reflect.Float64:    types.Float64,
	reflect.Complex64:  types.Complex64,
	reflect.Complex128: types.Complex128,
	reflect.String:     types.String,
}

// probeTypes converts the reflected types of guarded structs to go/types for
// the literal probes. The structs themselves are converted field by field.
// Every other named type becomes an opaque stand-in: the probes only pass its
// values around through field selectors, which needs nothing but its
// identity, so whatever it holds inside never has to be converted.
type probeTypes struct {
	pkg *types.Package // stands in for agentanalytics
	// local holds the import paths of the guarded structs, whose unexported
	// fields belong to pkg.
	local map[string]bool
	pkgs  map[string]*types.Package
	named map[reflect.Type]*types.Named
}

// probePackage returns a go/types package with the import path of
// agentanalytics that declares each struct under its name.
func probePackage(structs []guardedStruct) (*types.Package, error) {
	c := probeTypes{
		pkg:   types.NewPackage(agentAnalyticsPkgPath, "agentanalytics"),
		local: make(map[string]bool),
		pkgs:  make(map[string]*types.Package),
		named: make(map[reflect.Type]*types.Named),
	}
	// Declare every struct before converting any, so that a struct used as a
	// field of another, as RetryConfig is in Config, is one type in both.
	for _, gs := range structs {
		c.local[gs.typ.PkgPath()] = true
		tn := types.NewTypeName(token.NoPos, c.pkg, gs.name, nil)
		c.named[gs.typ] = types.NewNamed(tn, nil, nil)
		c.pkg.Scope().Insert(tn)
	}
	for _, gs := range structs {
		under, err := c.underlying(gs.typ)
		if err != nil {
			return nil, fmt.Errorf("converting %s: %w", gs.name, err)
		}
		c.named[gs.typ].SetUnderlying(under)
	}
	c.pkg.MarkComplete()
	return c.pkg, nil
}

// pkgFor returns the package that declares the names of the package with the
// given import path.
func (c *probeTypes) pkgFor(pkgPath string) *types.Package {
	if pkgPath == "" || c.local[pkgPath] {
		return c.pkg
	}
	if p, ok := c.pkgs[pkgPath]; ok {
		return p
	}
	p := types.NewPackage(pkgPath, path.Base(pkgPath))
	c.pkgs[pkgPath] = p
	return p
}

func (c *probeTypes) typ(rt reflect.Type) (types.Type, error) {
	if n, ok := c.named[rt]; ok {
		return n, nil
	}
	switch {
	case rt == reflect.TypeFor[error]():
		return types.Universe.Lookup("error").Type(), nil
	case rt == reflect.TypeFor[unsafe.Pointer]():
		return types.Typ[types.UnsafePointer], nil
	case rt.Name() != "" && rt.PkgPath() != "":
		tn := types.NewTypeName(token.NoPos, c.pkgFor(rt.PkgPath()), rt.Name(), nil)
		n := types.NewNamed(tn, types.NewStruct(nil, nil), nil)
		c.named[rt] = n
		return n, nil
	}
	return c.underlying(rt)
}

// underlying converts a predeclared or unnamed type, or the fields of a
// guarded struct.
func (c *probeTypes) underlying(rt reflect.Type) (types.Type, error) {
	if kind, ok := basicKinds[rt.Kind()]; ok {
		return types.Typ[kind], nil
	}
	switch rt.Kind() {
	case reflect.Pointer:
		elem, err := c.typ(rt.Elem())
		if err != nil {
			return nil, err
		}
		return types.NewPointer(elem), nil
	case reflect.Slice:
		elem, err := c.typ(rt.Elem())
		if err != nil {
			return nil, err
		}
		return types.NewSlice(elem), nil
	case reflect.Array:
		elem, err := c.typ(rt.Elem())
		if err != nil {
			return nil, err
		}
		return types.NewArray(elem, int64(rt.Len())), nil
	case reflect.Chan:
		elem, err := c.typ(rt.Elem())
		if err != nil {
			return nil, err
		}
		dir := types.SendRecv
		switch rt.ChanDir() {
		case reflect.SendDir:
			dir = types.SendOnly
		case reflect.RecvDir:
			dir = types.RecvOnly
		}
		return types.NewChan(dir, elem), nil
	case reflect.Map:
		key, err := c.typ(rt.Key())
		if err != nil {
			return nil, err
		}
		elem, err := c.typ(rt.Elem())
		if err != nil {
			return nil, err
		}
		return types.NewMap(key, elem), nil
	case reflect.Func:
		return c.signature(rt)
	case reflect.Interface:
		var methods []*types.Func
		for m := range rt.Methods() {
			sig, err := c.signature(m.Type)
			if err != nil {
				return nil, err
			}
			methods = append(methods, types.NewFunc(token.NoPos, c.pkgFor(m.PkgPath), m.Name, sig))
		}
		return types.NewInterfaceType(methods, nil).Complete(), nil
	case reflect.Struct:
		fields := make([]*types.Var, 0, rt.NumField())
		for sf := range rt.Fields() {
			ft, err := c.typ(sf.Type)
			if err != nil {
				return nil, fmt.Errorf("field %s: %w", sf.Name, err)
			}
			fields = append(fields, types.NewField(token.NoPos, c.pkgFor(sf.PkgPath), sf.Name, ft, sf.Anonymous))
		}
		return types.NewStruct(fields, nil), nil
	}
	return nil, fmt.Errorf("unsupported kind %v of %v", rt.Kind(), rt)
}

func (c *probeTypes) signature(rt reflect.Type) (*types.Signature, error) {
	params, err := c.tuple(rt.Ins())
	if err != nil {
		return nil, err
	}
	results, err := c.tuple(rt.Outs())
	if err != nil {
		return nil, err
	}
	return types.NewSignatureType(nil, nil, nil, params, results, rt.IsVariadic()), nil
}

func (c *probeTypes) tuple(rts iter.Seq[reflect.Type]) (*types.Tuple, error) {
	var vars []*types.Var
	for rt := range rts {
		t, err := c.typ(rt)
		if err != nil {
			return nil, err
		}
		vars = append(vars, types.NewParam(token.NoPos, c.pkg, "", t))
	}
	return types.NewTuple(vars...), nil
}

// checkExternalPositional type-checks literals of gs written in another
// package against pkg, and returns an error for each that does not come out
// the way the guard intends:
//   - a keyed literal of all the exported fields compiles with no type errors;
//   - a positional literal of the exported fields fails type-checking;
//   - the same positional literal with struct{}{} in the guard's place also
//     fails type-checking.
//
// Any go/types error on the positional literals counts as a rejection so the
// check does not depend on toolchain-specific diagnostic wording. The values
// are selectors on a zero value of gs, so each has the exact type of its
// field, whatever that type is.
func checkExternalPositional(pkg *types.Package, gs guardedStruct) []error {
	var keyed, positional []string
	for sf := range gs.typ.Fields() {
		if sf.IsExported() {
			keyed = append(keyed, sf.Name+": z."+sf.Name)
			positional = append(positional, "z."+sf.Name)
		}
	}
	var problems []error
	if errs := typecheckLiteral(pkg, gs.name, keyed); len(errs) != 0 {
		problems = append(problems, fmt.Errorf("keyed literal of %s: got errors %v, want none", gs.name, errs))
	}
	if errs := typecheckLiteral(pkg, gs.name, positional); len(errs) == 0 {
		problems = append(problems, fmt.Errorf("positional literal of %s compiled without error, want rejection", gs.name))
	}
	padded := append([]string{"struct{}{}"}, positional...)
	if errs := typecheckLiteral(pkg, gs.name, padded); len(errs) == 0 {
		problems = append(problems, fmt.Errorf("positional literal of %s with struct{}{} for the guard compiled without error, want rejection", gs.name))
	}
	return problems
}

// typecheckLiteral type-checks a file of another package that declares z, a
// zero value of the struct called name in pkg, and a literal of that struct
// with the given elements, and returns the errors.
func typecheckLiteral(pkg *types.Package, name string, elems []string) []error {
	src := fmt.Sprintf(`package external

import %q

var z agentanalytics.%s

var _ = agentanalytics.%s{%s}
`,
		pkg.Path(), name, name, strings.Join(elems, ", "))
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "external.go", src, 0)
	if err != nil {
		return []error{fmt.Errorf("parsing %q: %w", src, err)}
	}
	var errs []error
	conf := types.Config{
		Importer: probeImporter{pkg},
		Error:    func(err error) { errs = append(errs, err) },
	}
	_, _ = conf.Check("external", fset, []*ast.File{file}, nil)
	return errs
}

// probeImporter resolves the import of agentanalytics to the probe package.
type probeImporter struct{ pkg *types.Package }

func (i probeImporter) Import(importPath string) (*types.Package, error) {
	if importPath != i.pkg.Path() {
		return nil, fmt.Errorf("unexpected import %q", importPath)
	}
	return i.pkg, nil
}

// unguardedFields returns a struct type with the exported fields of rt, in
// order, and no other field. None of them is embedded: reflect.StructOf cannot
// build every embedded field, and embedding does not change the layout.
func unguardedFields(rt reflect.Type) reflect.Type {
	var fields []reflect.StructField
	for sf := range rt.Fields() {
		if sf.IsExported() {
			sf.Anonymous = false
			fields = append(fields, sf)
		}
	}
	return reflect.StructOf(fields)
}

// syntheticGuardedStruct has the guard and fields of kinds that none of the
// guarded structs has yet. It goes through the same layout check and literal
// probes as they do, which shows that adding fields like these to one of them
// leaves those checks working.
type syntheticGuardedStruct struct {
	_ struct{} `json:"-"`
	*bytes.Buffer
	Client   *http.Client
	Done     chan struct{}
	Ready    <-chan struct{}
	Errs     chan<- error
	Since    time.Time
	Backoffs [3]time.Duration
	Dims     [2]int
	OnEvent  func(...string)
	Closer   interface{ Close() error }
	Ptr      unsafe.Pointer
}

// syntheticGuardedWithPrivateCache verifies that adding unexported fields
// after the field-0 `_ struct{}` guard on a Config-like struct keeps keyed
// construction working and positional literals rejected (note that named
// unexported fields would make cmp.Equal panic on plain value structs).
type syntheticGuardedWithPrivateCache struct {
	_     struct{} `json:"-"`
	Name  string
	cache map[string]int
	Count int
	note  *http.Client
}

type syntheticUnexportedGuarded struct {
	_     struct{} `json:"-"`
	Value string
}

// SyntheticGuardedAlias exports the exact identifier imported by the external
// literal probe over unexported syntheticUnexportedGuarded.
type SyntheticGuardedAlias = syntheticUnexportedGuarded

type syntheticNamedPrivateFirst struct {
	cache int
	Name  string
}

type syntheticMissingJSONTag struct {
	_    struct{}
	Name string
}

func TestStructLiterals_SyntheticGuardedStruct(t *testing.T) {
	var zeroPrivate syntheticGuardedWithPrivateCache
	var zeroNamedFirst syntheticNamedPrivateFirst
	if zeroPrivate.cache != nil || zeroPrivate.note != nil || zeroNamedFirst.cache != 0 {
		t.Fatalf("unexpected non-zero synthetic private fields: %#v / %#v", zeroPrivate, zeroNamedFirst)
	}

	gs := guardedStruct{name: "SyntheticGuarded", typ: reflect.TypeFor[syntheticGuardedStruct]()}
	if err := checkGuardLayout(gs); err != nil {
		t.Error(err)
	}
	pkg, err := probePackage([]guardedStruct{gs})
	if err != nil {
		t.Fatalf("probePackage(%s) failed: %v", gs.name, err)
	}
	for _, problem := range checkExternalPositional(pkg, gs) {
		t.Error(problem)
	}

	// Verify structurally (without depending on types.TypeString formatting)
	// that the conversion keeps the name, embedding, and type of each field,
	// including array lengths, channel directions, and variadic parameters.
	converted := pkg.Scope().Lookup(gs.name).Type()
	st, ok := converted.Underlying().(*types.Struct)
	if !ok {
		t.Fatalf("%s converted to %v, want a struct", gs.name, converted)
	}
	if st.NumFields() != gs.typ.NumField() {
		t.Fatalf("converted struct has %d fields, want %d", st.NumFields(), gs.typ.NumField())
	}
	for i := range st.NumFields() {
		f, sf := st.Field(i), gs.typ.Field(i)
		if f.Name() != sf.Name || f.Embedded() != sf.Anonymous {
			t.Errorf("field %d = {Name:%q Embedded:%v}, want {Name:%q Embedded:%v}", i, f.Name(), f.Embedded(), sf.Name, sf.Anonymous)
		}
	}
	isNamed := func(t types.Type, pkgPath, name string) bool {
		n, ok := t.(*types.Named)
		return ok && n.Obj().Pkg() != nil && n.Obj().Pkg().Path() == pkgPath && n.Obj().Name() == name
	}
	isEmptyStruct := func(t types.Type) bool {
		s, ok := t.(*types.Struct)
		return ok && s.NumFields() == 0
	}
	errorType := types.Universe.Lookup("error").Type()
	fieldChecks := []struct {
		name string
		ok   func(types.Type) bool
	}{
		{"_", isEmptyStruct},
		{"Buffer", func(t types.Type) bool {
			p, ok := t.(*types.Pointer)
			return ok && isNamed(p.Elem(), "bytes", "Buffer")
		}},
		{"Client", func(t types.Type) bool {
			p, ok := t.(*types.Pointer)
			return ok && isNamed(p.Elem(), "net/http", "Client")
		}},
		{"Done", func(t types.Type) bool {
			ch, ok := t.(*types.Chan)
			return ok && ch.Dir() == types.SendRecv && isEmptyStruct(ch.Elem())
		}},
		{"Ready", func(t types.Type) bool {
			ch, ok := t.(*types.Chan)
			return ok && ch.Dir() == types.RecvOnly && isEmptyStruct(ch.Elem())
		}},
		{"Errs", func(t types.Type) bool {
			ch, ok := t.(*types.Chan)
			return ok && ch.Dir() == types.SendOnly && ch.Elem() == errorType
		}},
		{"Since", func(t types.Type) bool { return isNamed(t, "time", "Time") }},
		{"Backoffs", func(t types.Type) bool {
			a, ok := t.(*types.Array)
			return ok && a.Len() == 3 && isNamed(a.Elem(), "time", "Duration")
		}},
		{"Dims", func(t types.Type) bool {
			a, ok := t.(*types.Array)
			return ok && a.Len() == 2 && a.Elem() == types.Typ[types.Int]
		}},
		{"OnEvent", func(t types.Type) bool {
			sig, ok := t.(*types.Signature)
			if !ok || !sig.Variadic() || sig.Params().Len() != 1 || sig.Results().Len() != 0 {
				return false
			}
			sl, ok := sig.Params().At(0).Type().(*types.Slice)
			return ok && sl.Elem() == types.Typ[types.String]
		}},
		{"Closer", func(t types.Type) bool {
			iface, ok := t.(*types.Interface)
			if !ok || iface.NumMethods() != 1 || iface.Method(0).Name() != "Close" {
				return false
			}
			sig, ok := iface.Method(0).Type().(*types.Signature)
			return ok && sig.Params().Len() == 0 && sig.Results().Len() == 1 && sig.Results().At(0).Type() == errorType
		}},
		{"Ptr", func(t types.Type) bool { return t == types.Typ[types.UnsafePointer] }},
	}
	if len(fieldChecks) != st.NumFields() {
		t.Fatalf("fieldChecks has %d entries, want %d", len(fieldChecks), st.NumFields())
	}
	for i, want := range fieldChecks {
		f := st.Field(i)
		if f.Name() != want.name || !want.ok(f.Type()) {
			t.Errorf("field %d (%s) converted to %T (%v), did not match structural check for %s", i, f.Name(), f.Type(), f.Type(), want.name)
		}
	}

	// An exported alias over an unexported struct type works both when
	// guardedStruct records the explicit exported identifier rather than
	// rt.Name() and when go/types imports a real source-checked type alias.
	aliasGS := guardedStruct{name: "SyntheticGuardedAlias", typ: reflect.TypeFor[SyntheticGuardedAlias]()}
	testPkgPath := reflect.TypeFor[syntheticGuardedStruct]().PkgPath()
	if aliasGS.typ.PkgPath() != testPkgPath || aliasGS.typ.Name() != "syntheticUnexportedGuarded" {
		t.Errorf("reflect.TypeFor[SyntheticGuardedAlias] = %q.%q, want %q.%q",
			aliasGS.typ.PkgPath(), aliasGS.typ.Name(), testPkgPath, "syntheticUnexportedGuarded")
	}
	if err := checkGuardStructName(aliasGS); err == nil {
		t.Errorf("checkGuardStructName(%s) = nil, want mismatch error for alias over unexported target", aliasGS.name)
	}
	if err := checkGuardLayout(aliasGS); err != nil {
		t.Errorf("checkGuardLayout(%s) = %v, want nil", aliasGS.name, err)
	}
	aliasPkg, err := probePackage([]guardedStruct{aliasGS})
	if err != nil {
		t.Fatalf("probePackage(%s) failed: %v", aliasGS.name, err)
	}
	if problems := checkExternalPositional(aliasPkg, aliasGS); len(problems) != 0 {
		t.Errorf("checkExternalPositional(%s) reported problems: %v", aliasGS.name, problems)
	}
	aliasFset := token.NewFileSet()
	aliasFile, err := parser.ParseFile(aliasFset, "alias_fixture.go", `package agentanalytics

type syntheticUnexportedGuarded struct {
	_     struct{} `+"`json:\"-\"`"+`
	Value string
}

type SyntheticGuardedAlias = syntheticUnexportedGuarded
`, 0)
	if err != nil {
		t.Fatalf("parsing alias_fixture.go: %v", err)
	}
	realAliasPkg, err := (&types.Config{}).Check(agentAnalyticsPkgPath, aliasFset, []*ast.File{aliasFile}, nil)
	if err != nil {
		t.Fatalf("type-checking alias_fixture.go: %v", err)
	}
	if problems := checkExternalPositional(realAliasPkg, aliasGS); len(problems) != 0 {
		t.Errorf("checkExternalPositional(%s against source-checked alias package) reported problems: %v", aliasGS.name, problems)
	}

	// On a Config-like struct that is not compared with plain cmp.Equal,
	// unexported fields after field 0 remain compatible with both the layout
	// check and the external literal probes.
	withPrivate := guardedStruct{name: "SyntheticGuardedWithPrivateCache", typ: reflect.TypeFor[syntheticGuardedWithPrivateCache]()}
	if err := checkGuardLayout(withPrivate); err != nil {
		t.Errorf("checkGuardLayout(%s) = %v, want nil", withPrivate.name, err)
	}
	withPrivatePkg, err := probePackage([]guardedStruct{withPrivate})
	if err != nil {
		t.Fatalf("probePackage(%s) failed: %v", withPrivate.name, err)
	}
	if problems := checkExternalPositional(withPrivatePkg, withPrivate); len(problems) != 0 {
		t.Errorf("checkExternalPositional(%s) reported problems: %v", withPrivate.name, problems)
	}

	// An unguarded struct fails both checkGuardLayout and checkExternalPositional
	// (its unpadded positional literal compiles without error).
	unguarded := guardedStruct{name: "SyntheticUnguarded", typ: unguardedFields(gs.typ)}
	if err := checkGuardLayout(unguarded); err == nil {
		t.Errorf("checkGuardLayout(%s) = nil, want an error", unguarded.name)
	}
	unguardedPkg, err := probePackage([]guardedStruct{unguarded})
	if err != nil {
		t.Fatalf("probePackage(%s) failed: %v", unguarded.name, err)
	}
	if problems := checkExternalPositional(unguardedPkg, unguarded); len(problems) != 1 {
		t.Errorf("checkExternalPositional(%s) reported %d problems, want 1 (unpadded positional literal): %v", unguarded.name, len(problems), problems)
	}

	// A struct whose first field is a named unexported field rather than `_`
	// fails checkGuardLayout.
	namedFirst := guardedStruct{name: "SyntheticNamedPrivateFirst", typ: reflect.TypeFor[syntheticNamedPrivateFirst]()}
	if err := checkGuardLayout(namedFirst); err == nil {
		t.Errorf("checkGuardLayout(%s) = nil, want an error", namedFirst.name)
	}

	// A guard field missing the `json:"-"` tag fails checkGuardLayout.
	missingTag := guardedStruct{name: "SyntheticMissingJSONTag", typ: reflect.TypeFor[syntheticMissingJSONTag]()}
	if err := checkGuardLayout(missingTag); err == nil {
		t.Errorf("checkGuardLayout(%s) = nil, want an error", missingTag.name)
	}
}
