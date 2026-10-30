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

package configurable

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// The tests below drive the containment check on workflow edge references
// through the public FromConfig. The referenced node is a JoinNode: it is built
// entirely inside resolveNodeFromYAML, so the second check inside
// ResolveAgentReference, which an agent-class node goes through, cannot refuse
// the reference on behalf of the workflow check under test.

const joinNodeYAML = "name: join_node\nagent_class: JoinNode\n"

// writeConfig writes content to dir/name, creating dir, and returns the path.
func writeConfig(t *testing.T, dir, name, content string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("MkdirAll(%q) failed: %v", filepath.Dir(path), err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("WriteFile(%q) failed: %v", path, err)
	}
	return path
}

// writeWorkflow writes a workflow whose only edge leads from START to ref.
func writeWorkflow(t *testing.T, dir, name, ref string) string {
	t.Helper()
	return writeConfig(t, dir, name, fmt.Sprintf("name: wf\nagent_class: Workflow\nedges:\n  - - START\n    - %q\n", ref))
}

// newWorkflowLayout returns a temporary base with agent/ for the workflow under
// test and outside/join.yaml, a valid node the workflow must not reach.
func newWorkflowLayout(t *testing.T) (base, agentDir, outsideNode string) {
	t.Helper()
	base = t.TempDir()
	if resolved, err := filepath.EvalSymlinks(base); err == nil {
		base = resolved
	}
	agentDir = filepath.Join(base, "agent")
	outsideNode = writeConfig(t, filepath.Join(base, "outside"), "join.yaml", joinNodeYAML)
	if err := os.MkdirAll(agentDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(%q) failed: %v", agentDir, err)
	}
	return base, agentDir, outsideNode
}

func TestWorkflowNodeReferenceRejectsEscapingPath(t *testing.T) {
	_, agentDir, outsideNode := newWorkflowLayout(t)

	for _, tc := range []struct{ name, ref string }{
		{"parent traversal", "../outside/join.yaml"},
		{"absolute path", outsideNode},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// A workflow file of its own per case, so the node registry cannot hand
			// back a node cached by an earlier case.
			wf := writeWorkflow(t, agentDir, "wf_"+filepath.Base(tc.name)+".yaml", tc.ref)
			if _, err := FromConfig(t.Context(), wf); !isContainmentRejection(err) {
				t.Errorf("FromConfig(%q) with node ref %q = %v, want a containment rejection", wf, tc.ref, err)
			}
		})
	}
}

func TestWorkflowNodeReferenceAllowsPathsInsideWorkflowDir(t *testing.T) {
	_, agentDir, _ := newWorkflowLayout(t)
	writeConfig(t, agentDir, "nodes/join.yaml", joinNodeYAML)
	wf := writeWorkflow(t, agentDir, "wf_inside.yaml", "nodes/join.yaml")

	if _, err := FromConfig(t.Context(), wf); err != nil {
		t.Errorf("FromConfig(%q) with a node ref inside the workflow directory = %v, want success", wf, err)
	}
}

// TestWorkflowNodeReferenceCacheDoesNotBypassCheck pins the check ahead of the
// nodeRegistry lookup, an ordering that is easy to invert by accident: the node
// outside the workflow directory is first loaded legitimately, which caches it
// under its absolute path, and the escaping reference must still be refused.
func TestWorkflowNodeReferenceCacheDoesNotBypassCheck(t *testing.T) {
	base, agentDir, outsideNode := newWorkflowLayout(t)

	legit := writeWorkflow(t, filepath.Dir(outsideNode), "wf_legit.yaml", "join.yaml")
	if _, err := FromConfig(t.Context(), legit); err != nil {
		t.Fatalf("FromConfig(%q) = %v, want success: the node is inside its own workflow directory", legit, err)
	}
	registryMu.RLock()
	_, cached := nodeRegistry[outsideNode]
	registryMu.RUnlock()
	if !cached {
		t.Fatalf("nodeRegistry has no entry for %q after loading it, the test would not exercise the cache", outsideNode)
	}

	escaping := writeWorkflow(t, agentDir, "wf_cached.yaml", "../outside/join.yaml")
	if _, err := FromConfig(t.Context(), escaping); !isContainmentRejection(err) {
		t.Errorf("FromConfig(%q) from under %q with a ref to the cached node = %v, want a containment rejection", escaping, base, err)
	}
}
