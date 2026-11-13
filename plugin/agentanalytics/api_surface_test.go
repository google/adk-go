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
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"
	"unsafe"

	"google.golang.org/adk/plugin/agentanalytics"
	"google.golang.org/api/option"
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

type unguardedRetryConfig struct {
	MaxRetries   int
	InitialDelay time.Duration
	MaxDelay     time.Duration
	Multiplier   float64
}

type unguardedDropStats struct {
	QueueFullDrops        int64
	AppendFailureDrops    int64
	FormatterFailureDrops int64
	ShutdownDrops         int64
	TotalDrops            int64
}

type unguardedGCSUploadRequest struct {
	Path        string
	Data        []byte
	ContentType string
}

type unguardedViewDefinition struct {
	Name      string
	EventType string
	Query     string
}

type unguardedConfig struct {
	Enabled                      bool
	ProjectID                    string
	DatasetID                    string
	TableName                    string
	MaxContentLen                int
	BatchSize                    int
	BatchFlushIntv               time.Duration
	QueueMaxSize                 int
	ClientOptions                []option.ClientOption
	ClusteringFields             []string
	CustomTags                   map[string]any
	LogMultiModalContent         bool
	ShutdownTimeout              time.Duration
	Logger                       agentanalytics.Logger
	RetryConfig                  agentanalytics.RetryConfig
	DisableMultiModalContent     bool
	DisableSessionMetadata       bool
	CaptureFullMessageContent    bool
	DisableSensitiveKeyRedaction bool
	ContentFormatter             agentanalytics.ContentFormatter
	DebugContentFormatterErrors  bool
	EventAllowlist               []string
	EventDenylist                []string
	PayloadColumnDenylist        []string
	CustomMetadataAllowlist      []string
	FinalResponseToolNames       []string
	DisableFlushOnRunEnd         bool
	DisableSchemaUpgrade         bool
	EnableOTelCorrelation        bool
	GCSBucketName                string
	GCSClientOptions             []option.ClientOption
	GCSPathPrefix                string
	ConnectionID                 string
	MaxInlineTextSize            int
	GCSMaxConcurrentUploads      int
	GCSUploadTimeout             time.Duration
	GCSMaxPendingUploadBytes     int64
	GCSOffloader                 agentanalytics.GCSContentOffloader
	DisableCreateViews           bool
	ViewPrefix                   string
	ExactlyOnceDelivery          bool
}

type guardedStructSpec struct {
	name          string
	typ           reflect.Type
	unguardedSize uintptr
	unguardedAlgn uintptr
	exportedArgs  []string
}

func guardedStructSpecs() []guardedStructSpec {
	return []guardedStructSpec{
		{
			name:          "RetryConfig",
			typ:           reflect.TypeFor[agentanalytics.RetryConfig](),
			unguardedSize: unsafe.Sizeof(unguardedRetryConfig{}),
			unguardedAlgn: unsafe.Alignof(unguardedRetryConfig{}),
			exportedArgs:  []string{"1", "0", "0", "1"},
		},
		{
			name:          "DropStats",
			typ:           reflect.TypeFor[agentanalytics.DropStats](),
			unguardedSize: unsafe.Sizeof(unguardedDropStats{}),
			unguardedAlgn: unsafe.Alignof(unguardedDropStats{}),
			exportedArgs:  []string{"1", "2", "3", "4", "7"},
		},
		{
			name:          "GCSUploadRequest",
			typ:           reflect.TypeFor[agentanalytics.GCSUploadRequest](),
			unguardedSize: unsafe.Sizeof(unguardedGCSUploadRequest{}),
			unguardedAlgn: unsafe.Alignof(unguardedGCSUploadRequest{}),
			exportedArgs:  []string{`"p"`, "nil", `"text/plain"`},
		},
		{
			name:          "ViewDefinition",
			typ:           reflect.TypeFor[agentanalytics.ViewDefinition](),
			unguardedSize: unsafe.Sizeof(unguardedViewDefinition{}),
			unguardedAlgn: unsafe.Alignof(unguardedViewDefinition{}),
			exportedArgs:  []string{`"v_llm_request"`, `"LLM_REQUEST"`, `"SELECT 1"`},
		},
		{
			name:          "Config",
			typ:           reflect.TypeFor[agentanalytics.Config](),
			unguardedSize: unsafe.Sizeof(unguardedConfig{}),
			unguardedAlgn: unsafe.Alignof(unguardedConfig{}),
			exportedArgs: []string{
				"true", `"p"`, `"d"`, `"t"`, "0", "0", "0", "0", "nil",
				"nil", "nil", "false", "0", "nil", "agentanalytics.RetryConfig{}",
				"false", "false", "false", "false", "nil", "false",
				"nil", "nil", "nil", "nil", "nil", "false", "false", "false",
				`""`, "nil", `""`, `""`, "0", "0", "0", "0", "nil",
				"false", `""`, "false",
			},
		},
	}
}

func TestStructLiterals_ExternalPositionalRejected(t *testing.T) {
	specs := guardedStructSpecs()

	for _, spec := range specs {
		t.Run(spec.name+"_LayoutAndBlankField", func(t *testing.T) {
			if spec.typ.NumField() == 0 {
				t.Fatalf("%s has 0 fields", spec.name)
			}
			var exportedCount int
			for i := 0; i < spec.typ.NumField(); i++ {
				if spec.typ.Field(i).IsExported() {
					exportedCount++
				}
			}
			if exportedCount != len(spec.exportedArgs) {
				t.Fatalf("%s has %d exported fields, but test fixture has %d positional args", spec.name, exportedCount, len(spec.exportedArgs))
			}

			first := spec.typ.Field(0)
			if first.Name != "_" || first.IsExported() || first.PkgPath != agentAnalyticsPkgPath || first.Type != reflect.TypeFor[struct{}]() {
				t.Fatalf("%s field 0 = {Name:%q Exported:%v PkgPath:%q Type:%v}, want unexported `_ struct{}` in %q",
					spec.name, first.Name, first.IsExported(), first.PkgPath, first.Type, agentAnalyticsPkgPath)
			}
			if first.Offset != 0 || first.Type.Size() != 0 {
				t.Fatalf("%s field 0 Offset=%d Size=%d, want 0 and 0", spec.name, first.Offset, first.Type.Size())
			}
			if spec.typ.Size() != spec.unguardedSize {
				t.Fatalf("%s Size=%d, want %d (blank struct{} guard must not add padding)", spec.name, spec.typ.Size(), spec.unguardedSize)
			}
			if spec.typ.Align() != int(spec.unguardedAlgn) {
				t.Fatalf("%s Align=%d, want %d", spec.name, spec.typ.Align(), spec.unguardedAlgn)
			}
		})
	}

	// 1. In-process go/types check built from live reflected struct types.
	// Because reflect.TypeFor inspects the compiled binary directly, this check
	// also catches mutations supplied through `go test -overlay=...`.
	typesPkg := buildReflectedAgentAnalyticsTypesPackage(specs)
	checkExternalTypecheck(t, typesPkg, specs)

	// 2. Real `go build` compiler check across virtual external packages using
	// `-overlay` so all 5 guarded structs (unpadded and padded with `struct{}{}`)
	// and the 6 unexported helper symbols are verified by the Go toolchain.
	checkExternalGoBuild(t, specs)
}

func buildReflectedAgentAnalyticsTypesPackage(specs []guardedStructSpec) *types.Package {
	pkg := types.NewPackage(agentAnalyticsPkgPath, "agentanalytics")
	pkgs := map[string]*types.Package{
		agentAnalyticsPkgPath: pkg,
	}
	named := make(map[reflect.Type]*types.Named)
	for _, spec := range specs {
		_ = reflectTypeToGoTypes(spec.typ, pkg, pkgs, named)
	}
	pkg.MarkComplete()
	return pkg
}

func reflectTypeToGoTypes(rt reflect.Type, rootPkg *types.Package, pkgs map[string]*types.Package, named map[reflect.Type]*types.Named) types.Type {
	if rt.Name() != "" && rt.PkgPath() != "" {
		if existing, ok := named[rt]; ok {
			return existing
		}
		ownerPkg := pkgs[rt.PkgPath()]
		if ownerPkg == nil {
			pkgName := filepath.Base(rt.PkgPath())
			ownerPkg = types.NewPackage(rt.PkgPath(), pkgName)
			ownerPkg.MarkComplete()
			pkgs[rt.PkgPath()] = ownerPkg
		}
		tn := types.NewTypeName(token.NoPos, ownerPkg, rt.Name(), nil)
		n := types.NewNamed(tn, nil, nil)
		named[rt] = n
		under := reflectUnderlyingToGoTypes(rt, rootPkg, pkgs, named)
		n.SetUnderlying(under)
		if ownerPkg == rootPkg && token.IsExported(rt.Name()) && rootPkg.Scope().Lookup(rt.Name()) == nil {
			rootPkg.Scope().Insert(tn)
		}
		return n
	}
	return reflectUnderlyingToGoTypes(rt, rootPkg, pkgs, named)
}

func reflectUnderlyingToGoTypes(rt reflect.Type, rootPkg *types.Package, pkgs map[string]*types.Package, named map[reflect.Type]*types.Named) types.Type {
	switch rt.Kind() {
	case reflect.Bool:
		return types.Typ[types.Bool]
	case reflect.Int:
		return types.Typ[types.Int]
	case reflect.Int64:
		return types.Typ[types.Int64]
	case reflect.Uint8:
		return types.Typ[types.Byte]
	case reflect.Float64:
		return types.Typ[types.Float64]
	case reflect.String:
		return types.Typ[types.String]
	case reflect.Slice:
		return types.NewSlice(reflectTypeToGoTypes(rt.Elem(), rootPkg, pkgs, named))
	case reflect.Map:
		return types.NewMap(
			reflectTypeToGoTypes(rt.Key(), rootPkg, pkgs, named),
			reflectTypeToGoTypes(rt.Elem(), rootPkg, pkgs, named),
		)
	case reflect.Func:
		return types.NewSignatureType(nil, nil, nil, nil, nil, rt.IsVariadic())
	case reflect.Interface:
		if rt.NumMethod() == 0 {
			return types.NewInterfaceType(nil, nil).Complete()
		}
		m := rt.Method(0)
		sig := types.NewSignatureType(nil, nil, nil, nil, nil, false)
		fn := types.NewFunc(token.NoPos, rootPkg, m.Name, sig)
		return types.NewInterfaceType([]*types.Func{fn}, nil).Complete()
	case reflect.Struct:
		fields := make([]*types.Var, 0, rt.NumField())
		tags := make([]string, 0, rt.NumField())
		for i := 0; i < rt.NumField(); i++ {
			sf := rt.Field(i)
			ft := reflectTypeToGoTypes(sf.Type, rootPkg, pkgs, named)
			fieldPkg := rootPkg
			if sf.PkgPath != "" && sf.PkgPath != rootPkg.Path() {
				fieldPkg = pkgs[sf.PkgPath]
				if fieldPkg == nil {
					fieldPkg = types.NewPackage(sf.PkgPath, filepath.Base(sf.PkgPath))
					fieldPkg.MarkComplete()
					pkgs[sf.PkgPath] = fieldPkg
				}
			}
			fields = append(fields, types.NewField(token.NoPos, fieldPkg, sf.Name, ft, sf.Anonymous))
			tags = append(tags, string(sf.Tag))
		}
		return types.NewStruct(fields, tags)
	default:
		panic(fmt.Sprintf("unsupported reflect.Kind %v for %v", rt.Kind(), rt))
	}
}

type fixedPackageImporter map[string]*types.Package

func (m fixedPackageImporter) Import(path string) (*types.Package, error) {
	if pkg, ok := m[path]; ok {
		return pkg, nil
	}
	return nil, fmt.Errorf("unexpected import %q", path)
}

func typecheckExternalSource(t *testing.T, typesPkg *types.Package, src string) []error {
	t.Helper()
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "external_probe.go", src, 0)
	if err != nil {
		t.Fatalf("parser.ParseFile failed: %v\nsource:\n%s", err, src)
	}
	var errs []error
	conf := types.Config{
		Importer: fixedPackageImporter{agentAnalyticsPkgPath: typesPkg},
		Error: func(err error) {
			errs = append(errs, err)
		},
	}
	_, _ = conf.Check("external", fset, []*ast.File{file}, nil)
	return errs
}

func checkExternalTypecheck(t *testing.T, typesPkg *types.Package, specs []guardedStructSpec) {
	t.Helper()

	const keyedSrc = `package external

import "google.golang.org/adk/plugin/agentanalytics"

var (
	_ = agentanalytics.RetryConfig{MaxRetries: 3, Multiplier: 2.0}
	_ = agentanalytics.DropStats{QueueFullDrops: 1, TotalDrops: 1}
	_ = agentanalytics.GCSUploadRequest{Path: "p", ContentType: "text/plain"}
	_ = agentanalytics.ViewDefinition{Name: "v_llm_request", EventType: "LLM_REQUEST", Query: "SELECT 1"}
	_ = agentanalytics.Config{Enabled: true, ProjectID: "p", DatasetID: "d", TableName: "t"}
)
`
	if errs := typecheckExternalSource(t, typesPkg, keyedSrc); len(errs) != 0 {
		t.Fatalf("keyed external literals failed go/types check: %v", errs)
	}

	for _, spec := range specs {
		unpaddedExpr := fmt.Sprintf("agentanalytics.%s{%s}", spec.name, strings.Join(spec.exportedArgs, ", "))
		paddedArgs := append([]string{"struct{}{}"}, spec.exportedArgs...)
		paddedExpr := fmt.Sprintf("agentanalytics.%s{%s}", spec.name, strings.Join(paddedArgs, ", "))

		t.Run(spec.name+"_TypecheckPositionalUnpadded", func(t *testing.T) {
			src := fmt.Sprintf("package external\nimport %q\nvar _ = %s\n", agentAnalyticsPkgPath, unpaddedExpr)
			errs := typecheckExternalSource(t, typesPkg, src)
			joined := joinErrors(errs)
			if !strings.Contains(joined, "implicit assignment to unexported field _ in struct literal") {
				t.Fatalf("expected unpadded positional literal %s to fail with implicit assignment to unexported field _, got errors: %v", unpaddedExpr, errs)
			}
			if !strings.Contains(joined, "too few values in struct literal") {
				t.Fatalf("expected unpadded positional literal %s to report too few values in struct literal, got errors: %v", unpaddedExpr, errs)
			}
		})

		t.Run(spec.name+"_TypecheckPositionalPaddedWithBlankStruct", func(t *testing.T) {
			src := fmt.Sprintf("package external\nimport %q\nvar _ = %s\n", agentAnalyticsPkgPath, paddedExpr)
			errs := typecheckExternalSource(t, typesPkg, src)
			joined := joinErrors(errs)
			if len(errs) != 1 || !strings.Contains(joined, "implicit assignment to unexported field _ in struct literal") {
				t.Fatalf("expected padded positional literal %s to fail solely with implicit assignment to unexported field _, got %d error(s): %v", paddedExpr, len(errs), errs)
			}
		})
	}
}

func checkExternalGoBuild(t *testing.T, specs []guardedStructSpec) {
	t.Helper()

	goBin := filepath.Join(runtime.GOROOT(), "bin", "go")
	if _, err := os.Stat(goBin); err != nil {
		var lookErr error
		goBin, lookErr = exec.LookPath("go")
		if lookErr != nil {
			t.Skipf("go binary unavailable: %v", lookErr)
		}
	}

	tmpDir := t.TempDir()
	pkgDir, err := os.Getwd()
	if err != nil {
		t.Fatalf("os.Getwd failed: %v", err)
	}

	replace := make(map[string]string)

	// Positive external package check: keyed literals, equality, and JSON.
	const validKeyedPkg = "keyed_valid"
	const validKeyedSrc = `package keyed_valid

import "google.golang.org/adk/plugin/agentanalytics"

var (
	RetryA = agentanalytics.RetryConfig{MaxRetries: 3, Multiplier: 2.0}
	RetryB = agentanalytics.RetryConfig{MaxRetries: 3, Multiplier: 2.0}
	RetryEqual = RetryA == RetryB

	DropsA = agentanalytics.DropStats{QueueFullDrops: 1, TotalDrops: 1}
	DropsB = agentanalytics.DropStats{QueueFullDrops: 1, TotalDrops: 1}
	DropsEqual = DropsA == DropsB

	Upload = agentanalytics.GCSUploadRequest{Path: "p", Data: []byte("x"), ContentType: "text/plain"}

	ViewA = agentanalytics.ViewDefinition{Name: "v_llm_request", EventType: "LLM_REQUEST", Query: "SELECT 1"}
	ViewB = agentanalytics.ViewDefinition{Name: "v_llm_request", EventType: "LLM_REQUEST", Query: "SELECT 1"}
	ViewEqual = ViewA == ViewB

	Cfg = agentanalytics.Config{Enabled: true, ProjectID: "p", DatasetID: "d", TableName: "t", RetryConfig: RetryA}
)
`
	validRealFile := filepath.Join(tmpDir, validKeyedPkg+".go")
	if err := os.WriteFile(validRealFile, []byte(validKeyedSrc), 0o600); err != nil {
		t.Fatalf("os.WriteFile(%s) failed: %v", validRealFile, err)
	}
	replace[filepath.Join(pkgDir, "internal", "extprobe", validKeyedPkg, "probe.go")] = validRealFile

	type negativeCase struct {
		pkgSuffix  string
		structName string
		expr       string
		padded     bool
	}
	var negCases []negativeCase
	var negPkgArgs []string

	for _, spec := range specs {
		lower := strings.ToLower(spec.name)
		unpaddedPkg := lower + "_unpadded"
		unpaddedExpr := fmt.Sprintf("agentanalytics.%s{%s}", spec.name, strings.Join(spec.exportedArgs, ", "))
		negCases = append(negCases, negativeCase{
			pkgSuffix:  unpaddedPkg,
			structName: spec.name,
			expr:       unpaddedExpr,
			padded:     false,
		})

		paddedPkg := lower + "_padded"
		paddedArgs := append([]string{"struct{}{}"}, spec.exportedArgs...)
		paddedExpr := fmt.Sprintf("agentanalytics.%s{%s}", spec.name, strings.Join(paddedArgs, ", "))
		negCases = append(negCases, negativeCase{
			pkgSuffix:  paddedPkg,
			structName: spec.name,
			expr:       paddedExpr,
			padded:     true,
		})
	}

	for _, tc := range negCases {
		src := fmt.Sprintf("package %s\n\nimport %q\n\nvar _ = %s\n", tc.pkgSuffix, agentAnalyticsPkgPath, tc.expr)
		realFile := filepath.Join(tmpDir, tc.pkgSuffix+".go")
		if err := os.WriteFile(realFile, []byte(src), 0o600); err != nil {
			t.Fatalf("os.WriteFile(%s) failed: %v", realFile, err)
		}
		replace[filepath.Join(pkgDir, "internal", "extprobe", tc.pkgSuffix, "probe.go")] = realFile
		negPkgArgs = append(negPkgArgs, "./internal/extprobe/"+tc.pkgSuffix)
	}

	removedSymbols := []string{
		"ArrowSchema",
		"SerializedArrowSchema",
		"BatchProcessor",
		"NewBatchProcessor",
		"FormatContentParts",
		"SmartTruncate",
	}
	const removedPkg = "removed_symbols"
	var removedLines strings.Builder
	removedLines.WriteString("package removed_symbols\n\nimport \"google.golang.org/adk/plugin/agentanalytics\"\n\nvar (\n")
	for _, sym := range removedSymbols {
		fmt.Fprintf(&removedLines, "\t_ = agentanalytics.%s\n", sym)
	}
	removedLines.WriteString(")\n")
	removedRealFile := filepath.Join(tmpDir, removedPkg+".go")
	if err := os.WriteFile(removedRealFile, []byte(removedLines.String()), 0o600); err != nil {
		t.Fatalf("os.WriteFile(%s) failed: %v", removedRealFile, err)
	}
	replace[filepath.Join(pkgDir, "internal", "extprobe", removedPkg, "probe.go")] = removedRealFile
	negPkgArgs = append(negPkgArgs, "./internal/extprobe/"+removedPkg)

	overlayBytes, err := json.Marshal(struct {
		Replace map[string]string `json:"Replace"`
	}{Replace: replace})
	if err != nil {
		t.Fatalf("json.Marshal(overlay) failed: %v", err)
	}
	overlayPath := filepath.Join(tmpDir, "overlay.json")
	if err := os.WriteFile(overlayPath, overlayBytes, 0o600); err != nil {
		t.Fatalf("os.WriteFile(overlay.json) failed: %v", err)
	}

	env := buildGoProbeEnv(pkgDir)

	// 2a. Verify the keyed external package compiles cleanly.
	validCmd := exec.CommandContext(context.Background(), goBin, "build", "-mod=readonly", "-overlay="+overlayPath, "./internal/extprobe/"+validKeyedPkg)
	validCmd.Dir = pkgDir
	validCmd.Env = env
	if out, err := validCmd.CombinedOutput(); err != nil {
		t.Fatalf("go build on keyed external package failed (%v):\n%s", err, out)
	}

	// 2b. Verify all 10 positional literal packages and the unexported symbols
	// package fail compilation with the expected compiler diagnostics.
	args := append([]string{"build", "-mod=readonly", "-overlay=" + overlayPath}, negPkgArgs...)
	negCmd := exec.CommandContext(context.Background(), goBin, args...)
	negCmd.Dir = pkgDir
	negCmd.Env = env
	outBytes, err := negCmd.CombinedOutput()
	out := string(outBytes)
	if err == nil {
		t.Fatalf("go build on negative external packages unexpectedly succeeded; output:\n%s", out)
	}

	sections := parseGoBuildPackageSections(out)
	for _, tc := range negCases {
		t.Run(tc.pkgSuffix+"_GoBuild", func(t *testing.T) {
			pkgPath := agentAnalyticsPkgPath + "/internal/extprobe/" + tc.pkgSuffix
			sec, ok := sections[pkgPath]
			if !ok {
				t.Fatalf("external positional literal package %s compiled without error (expected failure for %s);\nfull go build output:\n%s",
					pkgPath, tc.expr, out)
			}
			wantUnexported := fmt.Sprintf("implicit assignment to unexported field _ in struct literal of type agentanalytics.%s", tc.structName)
			if !strings.Contains(sec, wantUnexported) {
				t.Fatalf("package %s compiler output missing %q:\n%s", pkgPath, wantUnexported, sec)
			}
			if !tc.padded && tc.structName != "Config" {
				wantTooFew := fmt.Sprintf("too few values in struct literal of type agentanalytics.%s", tc.structName)
				if !strings.Contains(sec, wantTooFew) {
					t.Fatalf("package %s compiler output missing %q:\n%s", pkgPath, wantTooFew, sec)
				}
			}
		})
	}

	t.Run("RemovedHelpersUnexported_GoBuild", func(t *testing.T) {
		pkgPath := agentAnalyticsPkgPath + "/internal/extprobe/" + removedPkg
		sec, ok := sections[pkgPath]
		if !ok {
			t.Fatalf("removed-symbols package %s compiled without error;\nfull go build output:\n%s", pkgPath, out)
		}
		for _, sym := range removedSymbols {
			want := "undefined: agentanalytics." + sym
			if !strings.Contains(sec, want) {
				t.Fatalf("package %s compiler output missing %q:\n%s", pkgPath, want, sec)
			}
		}
	})
}

func buildGoProbeEnv(pkgDir string) []string {
	repoRoot := filepath.Clean(filepath.Join(pkgDir, "..", ".."))
	rawEnv := os.Environ()
	out := make([]string, 0, len(rawEnv)+1)
	for _, kv := range rawEnv {
		k, v, ok := strings.Cut(kv, "=")
		if !ok {
			out = append(out, kv)
			continue
		}
		switch k {
		case "GOPROXY":
			continue
		case "GOWORK", "GOCACHE", "GOMODCACHE", "GOTMPDIR":
			if v != "" && v != "off" && !filepath.IsAbs(v) {
				if _, err := os.Stat(filepath.Join(pkgDir, v)); err != nil {
					candidate := filepath.Clean(filepath.Join(repoRoot, v))
					if _, rootErr := os.Stat(candidate); rootErr == nil {
						v = candidate
					}
				} else {
					v = filepath.Clean(filepath.Join(pkgDir, v))
				}
			}
			out = append(out, k+"="+v)
		default:
			out = append(out, kv)
		}
	}
	out = append(out, "GOPROXY=off")
	return out
}

func parseGoBuildPackageSections(output string) map[string]string {
	sections := make(map[string]string)
	var currentPkg string
	var buf strings.Builder
	flush := func() {
		if currentPkg != "" {
			sections[currentPkg] = buf.String()
			buf.Reset()
		}
	}
	for _, line := range strings.Split(output, "\n") {
		if strings.HasPrefix(line, "# ") {
			flush()
			hdr := strings.TrimSpace(strings.TrimPrefix(line, "# "))
			if idx := strings.IndexByte(hdr, ' '); idx >= 0 {
				hdr = hdr[:idx]
			}
			currentPkg = hdr
			continue
		}
		if currentPkg != "" {
			buf.WriteString(line)
			buf.WriteByte('\n')
		}
	}
	flush()
	return sections
}

func joinErrors(errs []error) string {
	parts := make([]string, 0, len(errs))
	for _, err := range errs {
		parts = append(parts, err.Error())
	}
	return strings.Join(parts, "\n")
}
