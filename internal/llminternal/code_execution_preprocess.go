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

package llminternal

import (
	"fmt"
	"iter"
	"slices"
	"strings"
	"unicode"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/codeexecutor"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// csvMIMEType is the only MIME type of inline data that the pre-processor
// turns into an input file (adk-python _DATA_FILE_UTIL_MAP).
const csvMIMEType = "text/csv"

// nonBuiltInExecutorInstruction is adk-python's
// _NON_BUILTIN_EXECUTOR_INSTRUCTION (flows/llm_flows/extensions/
// _code_execution.py), byte for byte, em dash included.
const nonBuiltInExecutorInstruction = `# CRITICAL: Code execution format

You have access to an external Python sandbox managed by the host
application. To run Python code, output it inside a fenced markdown
block exactly like this:

{code_fence_start}print("hello")
{code_fence_end}

DO NOT emit native executable_code parts.
DO NOT attempt to call a code_execution tool — no such tool is
registered for this request and the API will reject the response with
UNEXPECTED_TOOL_CALL or MALFORMED_FUNCTION_CALL.

Always wrap Python code in the markdown fence shown above.
`

// unfencedExecutorInstruction is the instruction for an executor without code
// block delimiters. It keeps the heading and the warning against native code
// parts of nonBuiltInExecutorInstruction and names no fence, because the
// post-processor searches none.
const unfencedExecutorInstruction = `# CRITICAL: Code execution format

DO NOT emit native executable_code parts.
DO NOT attempt to call a code_execution tool — no such tool is
registered for this request and the API will reject the response with
UNEXPECTED_TOOL_CALL or MALFORMED_FUNCTION_CALL.
`

// dataFileHelperLib is adk-python's _DATA_FILE_HELPER_LIB, byte for byte: the
// pandas helpers crop and explore_df that the exploration code of a data file
// calls.
const dataFileHelperLib = `
import pandas as pd

def crop(s: str, max_chars: int = 64) -> str:
  """Crops a string to max_chars characters."""
  if len(s) <= max_chars:
    return s
  if max_chars >= 3:
    return s[:max_chars - 3] + '...'
  return s[:max_chars]

def explore_df(df: pd.DataFrame) -> None:
  """Prints some information about a pandas DataFrame."""

  with pd.option_context(
      'display.max_columns', None, 'display.expand_frame_repr', False
  ):
    # Print the column names to never encounter KeyError when selecting one.
    df_dtypes = df.dtypes

    # Obtain information about data types and missing values.
    df_nulls = (len(df) - df.isnull().sum()).apply(
        lambda x: f'{x} / {df.shape[0]} non-null'
    )

    # Explore unique total values in columns using ` + "`" + `.unique()` + "`" + `.
    df_unique_count = df.apply(lambda x: len(x.unique()))

    # Explore unique values in columns using ` + "`" + `.unique()` + "`" + `.
    df_unique = df.apply(lambda x: crop(str(list(x.unique()))))

    df_info = pd.concat(
        (
            df_dtypes.rename('Dtype'),
            df_nulls.rename('Non-Null Count'),
            df_unique_count.rename('Unique Values Count'),
            df_unique.rename('Unique Values'),
        ),
        axis=1,
    )
    df_info.index.name = 'Columns'
    print(f"""Total rows: {df.shape[0]}
Total columns: {df.shape[1]}

{df_info}""")
`

// codeExecutionPreProcess prepares req for exec, an executor that is not a
// codeexecutor.BuiltIn (adk-python _run_pre_processor). ctx must have a
// session.
//
// With OptimizeDataFile false in exec's settings it leaves req unchanged.
// Otherwise it appends the fenced-code instruction to the system instruction
// and, unless the invocation's error count has reached ErrorRetryAttempts,
// replaces the text/csv inline data of the user contents of req with
// placeholders, stores each file in the session state and explores every
// stored file whose name is not yet listed as processed: it yields a code
// event, runs the exploration code with that one file, marks the file
// processed and yields the result event, appending both contents to req. A
// stored file with a MIME type that has no loader ends the exploration.
//
// On a failure it yields the error once and returns; events already yielded
// and contents already appended to req stay. A file whose execution failed is
// not marked processed.
func codeExecutionPreProcess(ctx agent.InvocationContext, req *model.LLMRequest, exec codeexecutor.CodeExecutor) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		settings := exec.Settings().WithDefaults()
		if !settings.OptimizeDataFile {
			return
		}
		appendFencedCodeInstruction(req, settings)
		ec, err := newCodeExecutorContext(ctx.Session().State())
		if err != nil {
			yield(nil, fmt.Errorf("prepare code execution: %w", err))
			return
		}
		errorCount, err := ec.errorCount(ctx.InvocationID())
		if err != nil {
			yield(nil, fmt.Errorf("prepare code execution: %w", err))
			return
		}
		if errorCount >= *settings.ErrorRetryAttempts {
			return
		}
		files, err := extractAndReplaceInlineFiles(ec, req)
		if err != nil {
			yield(nil, err)
			return
		}
		processed, err := ec.processedFileNames()
		if err != nil {
			yield(nil, fmt.Errorf("prepare code execution: %w", err))
			return
		}
		for _, f := range files {
			if slices.Contains(processed, f.Name) {
				continue
			}
			if !exploreDataFile(ctx, req, exec, settings, ec, f, yield) {
				return
			}
		}
	}
}

// exploreDataFile runs the exploration code of f, a stored input file, and
// yields its code and result events; see codeExecutionPreProcess. It reports
// whether the caller should go on with the next file.
func exploreDataFile(ctx agent.InvocationContext, req *model.LLMRequest, exec codeexecutor.CodeExecutor, settings codeexecutor.Settings, ec *codeExecutorContext, f *codeexecutor.File, yield func(*session.Event, error) bool) bool {
	code, ok := dataFilePreprocessingCode(f)
	if !ok {
		return false
	}
	codeContent := &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
		{Text: "Processing input file: `" + f.Name + "`"},
		genai.NewPartFromExecutableCode(code, genai.LanguagePython),
	}}
	req.Contents = append(req.Contents, clone(codeContent))
	if !yield(newExecutionEvent(ctx, codeContent), nil) {
		return false
	}
	res, err := runCodeWithFiles(ctx, exec, settings, ec, code, []*codeexecutor.File{f})
	if err != nil {
		yield(nil, err)
		return false
	}
	if err := ec.addProcessedFileName(f.Name); err != nil {
		yield(nil, fmt.Errorf("record processed input file: %w", err))
		return false
	}
	resultEvent, err := postProcessExecutionResult(ctx, ec, res)
	if err != nil {
		yield(nil, err)
		return false
	}
	if !yield(resultEvent, nil) {
		return false
	}
	req.Contents = append(req.Contents, clone(resultEvent.Content))
	return true
}

// appendFencedCodeInstruction appends to req's system instruction the
// instruction that tells the model to write code in the first code block
// delimiters of settings, which are the executor's with defaults applied.
// Without delimiters it appends unfencedExecutorInstruction instead.
func appendFencedCodeInstruction(req *model.LLMRequest, settings codeexecutor.Settings) {
	if len(settings.CodeBlockDelimiters) == 0 {
		utils.AppendInstructions(req, unfencedExecutorInstruction)
		return
	}
	pair := settings.CodeBlockDelimiters[0]
	utils.AppendInstructions(req, strings.NewReplacer("{code_fence_start}", pair[0], "{code_fence_end}", pair[1]).Replace(nonBuiltInExecutorInstruction))
}

// extractAndReplaceInlineFiles replaces each text/csv inline data part with
// data of every user content of req with the placeholder text of its file,
// data_<content>_<part>.csv counted from 1, and stores the file, with a copy
// of the bytes, in ec. It returns every stored input file, in storage order.
//
// A content it changes is replaced in req.Contents by a new Content; no
// Content or Part is edited in place. On a failure the contents replaced so
// far stay replaced.
func extractAndReplaceInlineFiles(ec *codeExecutorContext, req *model.LLMRequest) ([]*codeexecutor.File, error) {
	for i, content := range req.Contents {
		if content == nil || content.Role != genai.RoleUser || len(content.Parts) == 0 {
			continue
		}
		var parts []*genai.Part
		for j, part := range content.Parts {
			if part == nil || part.InlineData == nil || part.InlineData.MIMEType != csvMIMEType || part.InlineData.Data == nil {
				continue
			}
			name := fmt.Sprintf("data_%d_%d.csv", i+1, j+1)
			if parts == nil {
				parts = slices.Clone(content.Parts)
				req.Contents[i] = &genai.Content{Role: content.Role, Parts: parts}
			}
			parts[j] = &genai.Part{Text: "\nAvailable file: `" + name + "`\n"}
			file := &codeexecutor.File{Name: name, Content: slices.Clone(part.InlineData.Data), MimeType: csvMIMEType}
			if err := ec.storeInputFile(file); err != nil {
				return nil, fmt.Errorf("store input file: %w", err)
			}
		}
	}
	files, err := ec.inputFiles()
	if err != nil {
		return nil, fmt.Errorf("read input files: %w", err)
	}
	return files, nil
}

// dataFilePreprocessingCode returns the Python code that loads f into a
// pandas DataFrame and prints a summary of it with explore_df (adk-python
// _get_data_file_preprocessing_code), and false when f's MIME type has no
// loader. The file name reaches the code only as a Python string literal.
func dataFilePreprocessingCode(f *codeexecutor.File) (string, bool) {
	if f.MimeType != csvMIMEType {
		return "", false
	}
	name := dataFileVariableName(f.Name)
	return "\n" + dataFileHelperLib + "\n\n# Load the dataframe.\n" +
		name + " = pd.read_csv(" + pythonRepr(f.Name) + ")\n\n" +
		"# Use `explore_df` to guide my analysis.\nexplore_df(" + name + ")\n", true
}

// dataFileVariableName returns the Python variable name of the file name:
// the name without its extension, with every character outside [a-zA-Z0-9_]
// replaced by "_", "_data" when that is empty, and "_" in front when it starts
// with a digit.
func dataFileVariableName(fileName string) string {
	var b strings.Builder
	for _, r := range pathStem(fileName) {
		if r == '_' || 'a' <= r && r <= 'z' || 'A' <= r && r <= 'Z' || '0' <= r && r <= '9' {
			b.WriteRune(r)
		} else {
			b.WriteByte('_')
		}
	}
	v := b.String()
	if v == "" {
		return "_data"
	}
	if '0' <= v[0] && v[0] <= '9' {
		return "_" + v
	}
	return v
}

// pathStem returns p without its extension, as the first result of Python's
// os.path.splitext on POSIX: the extension starts at the last "." of the last
// path element, unless only dots precede it there.
func pathStem(p string) string {
	sep := strings.LastIndexByte(p, '/')
	dot := strings.LastIndexByte(p, '.')
	if dot <= sep {
		return p
	}
	for i := sep + 1; i < dot; i++ {
		if p[i] != '.' {
			return p[:dot]
		}
	}
	return p
}

// pythonRepr returns s as Python's repr writes a str: in single quotes, or in
// double quotes when s holds a single quote and no double quote; the
// backslash and the chosen quote escaped; tab, newline and carriage return as
// \t, \n and \r; other non-printable characters as \xNN below 0x100, \uNNNN
// below 0x10000 and \UNNNNNNNN above. Invalid UTF-8 is read as U+FFFD.
func pythonRepr(s string) string {
	quote := '\''
	if strings.ContainsRune(s, '\'') && !strings.ContainsRune(s, '"') {
		quote = '"'
	}
	var b strings.Builder
	b.WriteRune(quote)
	for _, r := range s {
		switch {
		case r == quote || r == '\\':
			b.WriteByte('\\')
			b.WriteRune(r)
		case r == '\t':
			b.WriteString(`\t`)
		case r == '\n':
			b.WriteString(`\n`)
		case r == '\r':
			b.WriteString(`\r`)
		case !unicode.IsPrint(r) && r < 0x100:
			fmt.Fprintf(&b, `\x%02x`, r)
		case !unicode.IsPrint(r) && r < 0x10000:
			fmt.Fprintf(&b, `\u%04x`, r)
		case !unicode.IsPrint(r):
			fmt.Fprintf(&b, `\U%08x`, r)
		default:
			b.WriteRune(r)
		}
	}
	b.WriteRune(quote)
	return b.String()
}
