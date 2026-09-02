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

package mcptoolset

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/modelcontextprotocol/go-sdk/mcp"

	"google.golang.org/adk/v2/agent"
)

type fixedResultMCPClient struct {
	result *mcp.CallToolResult
}

type unsupportedContent struct {
	mcp.Content
}

func (c *fixedResultMCPClient) CallTool(context.Context, *mcp.CallToolParams) (*mcp.CallToolResult, error) {
	return c.result, nil
}

func (*fixedResultMCPClient) ListTools(context.Context) ([]*mcp.Tool, error) {
	return nil, nil
}

func TestMCPToolRunContent(t *testing.T) {
	resourceSize := int64(1 << 20)
	tests := []struct {
		name    string
		result  *mcp.CallToolResult
		want    map[string]any
		wantErr string
	}{
		{
			name: "text only remains compatible",
			result: &mcp.CallToolResult{Content: []mcp.Content{
				&mcp.TextContent{Text: "first"},
				&mcp.TextContent{Text: "second"},
			}},
			want: map[string]any{"output": "firstsecond"},
		},
		{
			name: "GitHub file response includes embedded text",
			result: &mcp.CallToolResult{Content: []mcp.Content{
				&mcp.TextContent{Text: "successfully downloaded text file"},
				&mcp.EmbeddedResource{Resource: &mcp.ResourceContents{
					URI:      "repo://owner/project/main/file.go",
					MIMEType: "text/plain",
					Text:     "package example\n",
				}},
			}},
			want: map[string]any{"output": "successfully downloaded text file\n" +
				"[MCP embedded resource: uri=\"repo://owner/project/main/file.go\", mimeType=\"text/plain\"]\n" +
				"package example\n"},
		},
		{
			name: "text MIME blob is decoded",
			result: &mcp.CallToolResult{Content: []mcp.Content{
				&mcp.EmbeddedResource{Resource: &mcp.ResourceContents{
					URI:      "file:///response.json",
					MIMEType: "application/problem+json; charset=utf-8",
					Blob:     []byte(`{"status":"ok"}`),
				}},
			}},
			want: map[string]any{"output": "[MCP embedded resource: uri=\"file:///response.json\", " +
				"mimeType=\"application/problem+json; charset=utf-8\"]\n{\"status\":\"ok\"}"},
		},
		{
			name: "binary resource is represented by metadata",
			result: &mcp.CallToolResult{Content: []mcp.Content{
				&mcp.EmbeddedResource{Resource: &mcp.ResourceContents{
					URI:      "file:///report.pdf",
					MIMEType: "application/pdf",
					Blob:     []byte{1, 2, 3, 4},
				}},
			}},
			want: map[string]any{"output": "[MCP embedded resource: uri=\"file:///report.pdf\", " +
				"mimeType=\"application/pdf\", size=4 bytes]"},
		},
		{
			name: "unsupported text charset is represented by metadata",
			result: &mcp.CallToolResult{Content: []mcp.Content{
				&mcp.EmbeddedResource{Resource: &mcp.ResourceContents{
					URI:      "file:///utf16.txt",
					MIMEType: "text/plain; charset=utf-16",
					Blob:     []byte("hi"),
				}},
			}},
			want: map[string]any{"output": "[MCP embedded resource: uri=\"file:///utf16.txt\", " +
				"mimeType=\"text/plain; charset=utf-16\", size=2 bytes]"},
		},
		{
			name: "US-ASCII text blob is decoded",
			result: &mcp.CallToolResult{Content: []mcp.Content{
				&mcp.EmbeddedResource{Resource: &mcp.ResourceContents{
					URI:      "file:///ascii.txt",
					MIMEType: "text/plain; charset=us-ascii",
					Blob:     []byte("hello"),
				}},
			}},
			want: map[string]any{"output": "[MCP embedded resource: uri=\"file:///ascii.txt\", " +
				"mimeType=\"text/plain; charset=us-ascii\"]\nhello"},
		},
		{
			name: "non-ASCII byte is represented by metadata",
			result: &mcp.CallToolResult{Content: []mcp.Content{
				&mcp.EmbeddedResource{Resource: &mcp.ResourceContents{
					URI:      "file:///non-ascii.txt",
					MIMEType: "text/plain; charset=us-ascii",
					Blob:     []byte{0x80},
				}},
			}},
			want: map[string]any{"output": "[MCP embedded resource: uri=\"file:///non-ascii.txt\", " +
				"mimeType=\"text/plain; charset=us-ascii\", size=1 bytes]"},
		},
		{
			name: "malformed charset is represented by metadata",
			result: &mcp.CallToolResult{Content: []mcp.Content{
				&mcp.EmbeddedResource{Resource: &mcp.ResourceContents{
					URI:      "file:///utf16.txt",
					MIMEType: `text/plain; charset="utf-16`,
					Blob:     []byte{0x68, 0x00, 0x69, 0x00},
				}},
			}},
			want: map[string]any{"output": "[MCP embedded resource: uri=\"file:///utf16.txt\", " +
				"mimeType=\"text/plain; charset=\\\"utf-16\", size=4 bytes]"},
		},
		{
			name: "malformed media type is represented by metadata",
			result: &mcp.CallToolResult{Content: []mcp.Content{
				&mcp.EmbeddedResource{Resource: &mcp.ResourceContents{
					URI:      "file:///invalid.txt",
					MIMEType: "text/plain extra",
					Blob:     []byte("hello"),
				}},
			}},
			want: map[string]any{"output": "[MCP embedded resource: uri=\"file:///invalid.txt\", " +
				"mimeType=\"text/plain extra\", size=5 bytes]"},
		},
		{
			name: "invalid UTF-8 text blob is represented by metadata",
			result: &mcp.CallToolResult{Content: []mcp.Content{
				&mcp.EmbeddedResource{Resource: &mcp.ResourceContents{
					URI:      "file:///invalid.txt",
					MIMEType: "text/plain",
					Blob:     []byte{0xc3, 0x28},
				}},
			}},
			want: map[string]any{"output": "[MCP embedded resource: uri=\"file:///invalid.txt\", " +
				"mimeType=\"text/plain\", size=2 bytes]"},
		},
		{
			name: "resource link includes available metadata",
			result: &mcp.CallToolResult{Content: []mcp.Content{
				&mcp.ResourceLink{
					URI:         "https://example.com/archive.txt",
					Name:        "archive.txt",
					Title:       "Archive",
					Description: "Large text file",
					MIMEType:    "text/plain",
					Size:        &resourceSize,
				},
			}},
			want: map[string]any{"output": "[MCP resource link: uri=\"https://example.com/archive.txt\", " +
				"mimeType=\"text/plain\", name=\"archive.txt\", title=\"Archive\", " +
				"description=\"Large text file\", size=1048576 bytes]"},
		},
		{
			name: "image and audio are represented in order",
			result: &mcp.CallToolResult{Content: []mcp.Content{
				&mcp.TextContent{Text: "media:"},
				&mcp.ImageContent{MIMEType: "image/png", Data: []byte{1, 2, 3}},
				&mcp.AudioContent{MIMEType: "audio/wav", Data: []byte{4, 5}},
				&mcp.TextContent{},
				&mcp.TextContent{Text: "done"},
			}},
			want: map[string]any{"output": "media:\n[MCP image: mimeType=\"image/png\", size=3 bytes]\n" +
				"[MCP audio: mimeType=\"audio/wav\", size=2 bytes]\ndone"},
		},
		{
			name: "empty leading block does not add a separator",
			result: &mcp.CallToolResult{Content: []mcp.Content{
				&mcp.TextContent{},
				&mcp.ImageContent{MIMEType: "image/png", Data: []byte{1}},
			}},
			want: map[string]any{"output": "[MCP image: mimeType=\"image/png\", size=1 bytes]"},
		},
		{
			name: "unsupported content is represented in order",
			result: &mcp.CallToolResult{Content: []mcp.Content{
				&mcp.TextContent{Text: "before"},
				&unsupportedContent{},
				&mcp.TextContent{Text: "after"},
			}},
			want: map[string]any{"output": "before\n[MCP content: unsupported]\nafter"},
		},
		{
			name: "existing newline precedes non-text content",
			result: &mcp.CallToolResult{Content: []mcp.Content{
				&mcp.TextContent{Text: "downloaded\n"},
				&mcp.ResourceLink{URI: "https://example.com/file"},
			}},
			want: map[string]any{"output": "downloaded\n" +
				"[MCP resource link: uri=\"https://example.com/file\"]"},
		},
		{
			name: "structured output with mixed content remains compatible",
			result: &mcp.CallToolResult{
				StructuredContent: map[string]any{"sha": "abc123"},
				Content: []mcp.Content{
					&mcp.TextContent{Text: "downloaded"},
					&mcp.EmbeddedResource{Resource: &mcp.ResourceContents{
						URI:      "repo://owner/project/file.txt",
						MIMEType: "text/plain",
						Text:     "file contents",
					}},
				},
			},
			want: map[string]any{"output": map[string]any{"sha": "abc123"}},
		},
		{
			name: "structured output with text only remains compatible",
			result: &mcp.CallToolResult{
				StructuredContent: map[string]any{"status": "ok"},
				Content:           []mcp.Content{&mcp.TextContent{Text: `{"status":"ok"}`}},
			},
			want: map[string]any{"output": map[string]any{"status": "ok"}},
		},
		{
			name: "error response includes non-text details",
			result: &mcp.CallToolResult{
				IsError: true,
				Content: []mcp.Content{
					&mcp.TextContent{Text: "download failed"},
					&mcp.ResourceLink{URI: "https://example.com/error", MIMEType: "text/plain"},
				},
			},
			wantErr: "Tool execution failed. Details: download failed\n" +
				"[MCP resource link: uri=\"https://example.com/error\", mimeType=\"text/plain\"]",
		},
		{
			name:    "nil content returns an error",
			result:  &mcp.CallToolResult{},
			wantErr: "no text content in tool response",
		},
		{
			name: "empty content returns an error",
			result: &mcp.CallToolResult{
				Content: []mcp.Content{},
			},
			wantErr: "no text content in tool response",
		},
		{
			name: "all-empty text content returns an error",
			result: &mcp.CallToolResult{Content: []mcp.Content{
				&mcp.TextContent{},
				&mcp.TextContent{},
			}},
			wantErr: "no text content in tool response",
		},
		{
			name: "typed nil content does not panic",
			result: &mcp.CallToolResult{Content: []mcp.Content{
				(*mcp.TextContent)(nil),
				(*mcp.EmbeddedResource)(nil),
				(*mcp.ResourceLink)(nil),
				(*mcp.ImageContent)(nil),
				(*mcp.AudioContent)(nil),
			}},
			want: map[string]any{"output": "[MCP text content: unavailable]\n" +
				"[MCP embedded resource: unavailable]\n" +
				"[MCP resource link: unavailable]\n" +
				"[MCP image: unavailable]\n" +
				"[MCP audio: unavailable]"},
		},
		{
			name: "nil resource does not panic",
			result: &mcp.CallToolResult{Content: []mcp.Content{
				&mcp.EmbeddedResource{},
			}},
			want: map[string]any{"output": "[MCP embedded resource: unavailable]"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tool := &mcpTool{
				name:      "test_tool",
				mcpClient: &fixedResultMCPClient{result: test.result},
			}
			got, err := tool.Run(&agent.ContextMock{}, map[string]any{})
			if test.wantErr != "" {
				if err == nil || err.Error() != test.wantErr {
					t.Fatalf("Run() error = %v, want %q", err, test.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("Run() error = %v", err)
			}
			if diff := cmp.Diff(test.want, got); diff != "" {
				t.Errorf("Run() result mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
