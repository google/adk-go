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
	"encoding/asn1"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"go/types"
	"iter"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"
	"unsafe"

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
		FormatterFailureDrops: 3,
		ShutdownDrops:         4,
		TotalDrops:            7,
	}
	dropsB := agentanalytics.DropStats{
		QueueFullDrops:        1,
		AppendFailureDrops:    2,
		FormatterFailureDrops: 3,
		ShutdownDrops:         4,
		TotalDrops:            7,
	}
	dropsC := agentanalytics.DropStats{
		QueueFullDrops:        1,
		AppendFailureDrops:    2,
		FormatterFailureDrops: 3,
		ShutdownDrops:         4,
		TotalDrops:            8,
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
	const wantJSON = `{"queue_full_drops":1,"append_failure_drops":2,"formatter_failure_drops":3,"shutdown_drops":4,"total_drops":7}`
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

	uploadReq := agentanalytics.GCSUploadRequest{
		Path:        "prefix/2026-11-07/obj.txt",
		Data:        []byte("payload"),
		ContentType: "text/plain; charset=utf-8",
	}
	if uploadReq.Path != "prefix/2026-11-07/obj.txt" || !bytes.Equal(uploadReq.Data, []byte("payload")) || uploadReq.ContentType != "text/plain; charset=utf-8" {
		t.Fatalf("unexpected GCSUploadRequest fields: %#v", uploadReq)
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
// adding a field cannot break such code. name is what the literal probes call
// the struct.
type guardedStruct struct {
	name string
	typ  reflect.Type
}

// guardedStructs returns the package's guarded structs.
func guardedStructs() []guardedStruct {
	var structs []guardedStruct
	for _, rt := range []reflect.Type{
		reflect.TypeFor[agentanalytics.RetryConfig](),
		reflect.TypeFor[agentanalytics.DropStats](),
		reflect.TypeFor[agentanalytics.GCSUploadRequest](),
		reflect.TypeFor[agentanalytics.ViewDefinition](),
		reflect.TypeFor[agentanalytics.Config](),
	} {
		structs = append(structs, guardedStruct{name: rt.Name(), typ: rt})
	}
	return structs
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

// checkGuardLayout returns an error unless the first field of gs is the
// guard, the blank `_ struct{}` of the struct's own package, every other field
// is exported, and the guard leaves the size and alignment of the struct as
// they would be without it.
func checkGuardLayout(gs guardedStruct) error {
	rt := gs.typ
	if rt.NumField() < 2 {
		return fmt.Errorf("%s has %d fields, want the guard and at least one exported field", gs.name, rt.NumField())
	}
	if guard := rt.Field(0); guard.Name != "_" || guard.PkgPath != rt.PkgPath() || guard.Type != reflect.TypeFor[struct{}]() {
		return fmt.Errorf("%s field 0 = {Name:%q PkgPath:%q Type:%v}, want `_ struct{}` of %q", gs.name, guard.Name, guard.PkgPath, guard.Type, rt.PkgPath())
	}
	for i := 1; i < rt.NumField(); i++ {
		if sf := rt.Field(i); !sf.IsExported() {
			return fmt.Errorf("%s field %d (%s) is unexported; the literal probes expect the guard to be the only unexported field", gs.name, i, sf.Name)
		}
	}
	if u := unguardedFields(rt); rt.Size() != u.Size() || rt.Align() != u.Align() {
		return fmt.Errorf("%s Size=%d Align=%d, want %d and %d as without the guard", gs.name, rt.Size(), rt.Align(), u.Size(), u.Align())
	}
	return nil
}

func TestStructLiterals_ExternalPositionalRejected(t *testing.T) {
	structs := guardedStructs()
	for _, gs := range structs {
		t.Run(gs.name+"_GuardLayout", func(t *testing.T) {
			if err := checkGuardLayout(gs); err != nil {
				t.Error(err)
			}
		})
	}

	// The probes type-check against the struct types compiled into this
	// binary, read through reflect, so they also see a change supplied with
	// `go test -overlay`.
	pkg, err := probePackage(structs)
	if err != nil {
		t.Fatalf("probePackage() failed: %v", err)
	}
	for _, gs := range structs {
		t.Run(gs.name+"_ExternalLiterals", func(t *testing.T) {
			for _, problem := range checkExternalPositional(pkg, gs) {
				t.Error(problem)
			}
		})
	}

	t.Run("SourceRemovedExports", checkUnexportedSymbolsByParser)
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
	named map[reflect.Type]*types.Named
}

// probePackage returns a go/types package with the import path of
// agentanalytics that declares each struct under its name.
func probePackage(structs []guardedStruct) (*types.Package, error) {
	c := probeTypes{
		pkg:   types.NewPackage(agentAnalyticsPkgPath, "agentanalytics"),
		local: make(map[string]bool),
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
	return types.NewPackage(pkgPath, path.Base(pkgPath))
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

// guardRejection is how go/types rejects a positional literal, written in
// another package, that assigns a value to the guard.
const guardRejection = "implicit assignment to unexported field _ in struct literal"

func isGuardRejection(err error) bool {
	return strings.Contains(err.Error(), guardRejection)
}

// checkExternalPositional type-checks literals of gs written in another
// package against pkg, and returns an error for each that does not come out
// the way the guard intends:
//   - a keyed literal of all the exported fields compiles;
//   - a positional literal of the exported fields, as code could write before
//     the guard was added, is rejected for assigning to the guard;
//   - the same literal with struct{}{} in the guard's place is rejected for
//     that alone.
//
// The values are selectors on a zero value of gs, so each has exactly the type
// of its field, whatever that type is.
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
	if errs := typecheckLiteral(pkg, gs.name, positional); !slices.ContainsFunc(errs, isGuardRejection) {
		problems = append(problems, fmt.Errorf("positional literal of %s: got errors %v, want %q", gs.name, errs, guardRejection))
	}
	padded := append([]string{"struct{}{}"}, positional...)
	if errs := typecheckLiteral(pkg, gs.name, padded); len(errs) != 1 || !isGuardRejection(errs[0]) {
		problems = append(problems, fmt.Errorf("positional literal of %s with struct{}{} for the guard: got errors %v, want only %q", gs.name, errs, guardRejection))
	}
	return problems
}

// typecheckLiteral type-checks a file of another package that declares z, a
// zero value of the struct called name in pkg, and a literal of that struct
// with the given elements, and returns the errors.
func typecheckLiteral(pkg *types.Package, name string, elems []string) []error {
	src := fmt.Sprintf("package external\n\nimport %q\n\nvar z agentanalytics.%s\n\nvar _ = agentanalytics.%s{%s}\n",
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

// removedNames are the names removed from the package's exported API.
var removedNames = []string{
	"ArrowSchema",
	"SerializedArrowSchema",
	"BatchProcessor",
	"NewBatchProcessor",
	"FormatContentParts",
	"SmartTruncate",
}

// checkUnexportedSymbolsByParser fails if one of the package's non-test
// source files that a build would compile declares one of removedNames at
// package level. removed_exports_test.go already makes the test build fail
// on any such declaration that it compiles, including one supplied with
// `go test -overlay`; this check reads the files on disk, so it still holds
// without that file. It skips when it finds no source files, either next to
// the path this file was compiled from or in the working directory.
func checkUnexportedSymbolsByParser(t *testing.T) {
	var dirs []string
	if _, file, _, ok := runtime.Caller(0); ok {
		dirs = append(dirs, filepath.Dir(file))
	}
	if wd, err := os.Getwd(); err == nil {
		dirs = append(dirs, wd)
	}
	for _, dir := range dirs {
		declared, files := packageLevelNames(dir)
		if files == 0 {
			continue
		}
		if _, ok := declared["Config"]; !ok {
			t.Fatalf("found no Config declaration in the %d agentanalytics files in %s", files, dir)
		}
		for _, name := range removedNames {
			if file, ok := declared[name]; ok {
				t.Errorf("%s declares %s, which must stay out of the exported API", filepath.Join(dir, file), name)
			}
		}
		return
	}
	t.Skipf("found no agentanalytics source files in %q", dirs)
}

// packageLevelNames returns the file that declares each package-level name in
// the non-test agentanalytics files in dir that a build would compile, and
// how many such files there are.
func packageLevelNames(dir string) (map[string]string, int) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, 0
	}
	declared := make(map[string]string)
	files := 0
	fset := token.NewFileSet()
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		// Leave out the files a build would, such as one under
		// //go:build ignore.
		if match, err := build.Default.MatchFile(dir, name); err != nil || !match {
			continue
		}
		f, err := parser.ParseFile(fset, filepath.Join(dir, name), nil, parser.SkipObjectResolution)
		if err != nil || f.Name.Name != "agentanalytics" {
			continue
		}
		files++
		for _, decl := range f.Decls {
			switch d := decl.(type) {
			case *ast.FuncDecl:
				if d.Recv == nil {
					declared[d.Name.Name] = name
				}
			case *ast.GenDecl:
				for _, spec := range d.Specs {
					switch s := spec.(type) {
					case *ast.TypeSpec:
						declared[s.Name.Name] = name
					case *ast.ValueSpec:
						for _, id := range s.Names {
							declared[id.Name] = name
						}
					}
				}
			}
		}
	}
	return declared, files
}

func TestStructLiterals_ASN1MarshalRejectsGuardedStructs(t *testing.T) {
	// encoding/asn1 encoded each of these before the guard was added.
	// RetryConfig and Config already had fields it cannot encode.
	cases := []struct {
		name  string
		value any
	}{
		{
			name:  "DropStats",
			value: agentanalytics.DropStats{},
		},
		{
			name: "GCSUploadRequest",
			value: agentanalytics.GCSUploadRequest{
				Path:        "p",
				Data:        []byte("x"),
				ContentType: "text/plain",
			},
		},
		{
			name: "ViewDefinition",
			value: agentanalytics.ViewDefinition{
				Name:  "v",
				Query: "SELECT 1",
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := asn1.Marshal(tc.value)
			if err == nil {
				t.Fatalf("asn1.Marshal(%s) succeeded; want unexported-field error", tc.name)
			}
			if !strings.Contains(err.Error(), "struct contains unexported fields") {
				t.Fatalf("asn1.Marshal(%s) error = %q; want substring %q", tc.name, err.Error(), "struct contains unexported fields")
			}
		})
	}
}

// syntheticGuardedStruct has the guard and fields of kinds that none of the
// guarded structs has yet. It goes through the same layout check and literal
// probes as they do, which shows that adding fields like these to one of them
// leaves those checks working.
type syntheticGuardedStruct struct {
	_ struct{}
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

func TestStructLiterals_SyntheticGuardedStruct(t *testing.T) {
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

	// The conversion keeps the name, embedding and type of each field,
	// including array lengths, channel directions and variadic parameters.
	converted := pkg.Scope().Lookup(gs.name).Type()
	st, ok := converted.Underlying().(*types.Struct)
	if !ok {
		t.Fatalf("%s converted to %v, want a struct", gs.name, converted)
	}
	wantTypes := []string{
		"struct{}",
		"*bytes.Buffer",
		"*http.Client",
		"chan struct{}",
		"<-chan struct{}",
		"chan<- error",
		"time.Time",
		"[3]time.Duration",
		"[2]int",
		"func(...string)",
		"interface{Close() error}",
		"unsafe.Pointer",
	}
	var gotTypes []string
	for i := range st.NumFields() {
		f, sf := st.Field(i), gs.typ.Field(i)
		if f.Name() != sf.Name || f.Embedded() != sf.Anonymous {
			t.Errorf("field %d = {Name:%q Embedded:%v}, want {Name:%q Embedded:%v}", i, f.Name(), f.Embedded(), sf.Name, sf.Anonymous)
		}
		gotTypes = append(gotTypes, types.TypeString(f.Type(), (*types.Package).Name))
		// An opaque stand-in would print the same, so compare the type itself.
		if sf.Type == reflect.TypeFor[unsafe.Pointer]() && f.Type() != types.Typ[types.UnsafePointer] {
			t.Errorf("field %s converted to %v (%T), want the predeclared unsafe.Pointer", sf.Name, f.Type(), f.Type())
		}
	}
	if !slices.Equal(gotTypes, wantTypes) {
		t.Errorf("converted field types = %q, want %q", gotTypes, wantTypes)
	}

	// The same checks fail for a struct with these fields and no guard.
	control := guardedStruct{name: "SyntheticUnguarded", typ: unguardedFields(gs.typ)}
	if err := checkGuardLayout(control); err == nil {
		t.Errorf("checkGuardLayout(%s) = nil, want an error", control.name)
	}
	controlPkg, err := probePackage([]guardedStruct{control})
	if err != nil {
		t.Fatalf("probePackage(%s) failed: %v", control.name, err)
	}
	if problems := checkExternalPositional(controlPkg, control); len(problems) != 2 {
		t.Errorf("checkExternalPositional(%s) reported %d problems, want 2, one for each positional literal: %v", control.name, len(problems), problems)
	}
}
