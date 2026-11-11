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
	"strings"
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

	// Literal spellings, not filepath.Join, which would clean the interior ".."
	// away before it reached the check.
	for i, ref := range []string{"nodes/join.yaml", "./nodes/join.yaml", "nodes/../nodes/join.yaml"} {
		t.Run(ref, func(t *testing.T) {
			wf := writeWorkflow(t, agentDir, fmt.Sprintf("wf_inside_%d.yaml", i), ref)
			if _, err := FromConfig(t.Context(), wf); err != nil {
				t.Errorf("FromConfig(%q) with node ref %q inside the workflow directory = %v, want success", wf, ref, err)
			}
		})
	}
}

// writeRoutedWorkflow writes a workflow whose edge routes from router to target
// through a route map, the other path by which parseEdges resolves a node.
func writeRoutedWorkflow(t *testing.T, dir, name, router, target string) string {
	t.Helper()
	return writeConfig(t, dir, name, fmt.Sprintf("name: wf\nagent_class: Workflow\nedges:\n  - - START\n    - %q\n    - default: %q\n", router, target))
}

// TestWorkflowRouteMapReferenceIsChecked pins that a route-map target goes
// through the same containment check as a linear edge.
func TestWorkflowRouteMapReferenceIsChecked(t *testing.T) {
	_, agentDir, outsideNode := newWorkflowLayout(t)
	writeConfig(t, agentDir, "nodes/router.yaml", "name: router_node\nagent_class: JoinNode\n")
	writeConfig(t, agentDir, "nodes/join.yaml", joinNodeYAML)

	inside := writeRoutedWorkflow(t, agentDir, "wf_route_inside.yaml", "nodes/router.yaml", "nodes/join.yaml")
	if _, err := FromConfig(t.Context(), inside); err != nil {
		t.Fatalf("FromConfig(%q) with a route-map target inside the workflow directory = %v, want success", inside, err)
	}

	for _, tc := range []struct{ name, ref string }{
		{"parent traversal", "../outside/join.yaml"},
		{"absolute path", outsideNode},
	} {
		t.Run(tc.name, func(t *testing.T) {
			wf := writeRoutedWorkflow(t, agentDir, "wf_route_"+filepath.Base(tc.name)+".yaml", "nodes/router.yaml", tc.ref)
			_, err := FromConfig(t.Context(), wf)
			if !isContainmentRejection(err) {
				t.Fatalf("FromConfig(%q) with route-map target %q = %v, want a containment rejection", wf, tc.ref, err)
			}
			if !strings.Contains(err.Error(), fmt.Sprintf("%q", tc.ref)) {
				t.Errorf("FromConfig(%q) = %q, want the error to name the reference %q", wf, err, tc.ref)
			}
		})
	}
}

// TestWorkflowNodeRegistryDoesNotLeakAcrossDirectories pins that a name
// reference never reaches a node loaded from YAML. nodeRegistry is keyed by the
// real path of the file, which need not end in .yaml: here n.yaml links to
// join.node, so the node is cached under a key without the suffix. (A link to
// n.YAML would do on a case-sensitive filesystem, but n.yaml and n.YAML are one
// file on macOS and Windows.) A workflow in another directory naming that real
// path spells a name, not a YAML reference, and a name lookup in nodeRegistry
// would hand the node over without the containment check.
func TestWorkflowNodeRegistryDoesNotLeakAcrossDirectories(t *testing.T) {
	base, agentDir, _ := newWorkflowLayout(t)
	real := writeConfig(t, agentDir, "join.node", joinNodeYAML)
	if err := os.Symlink("join.node", filepath.Join(agentDir, "n.yaml")); err != nil {
		t.Skipf("symlinks are not supported in this environment: %v", err)
	}

	legit := writeWorkflow(t, agentDir, "wf_link.yaml", "n.yaml")
	if _, err := FromConfig(t.Context(), legit); err != nil {
		t.Fatalf("FromConfig(%q) = %v, want success: the link stays inside the workflow directory", legit, err)
	}
	registryMu.RLock()
	_, cached := nodeRegistry[configCacheKey{path: real}]
	registryMu.RUnlock()
	if !cached {
		t.Fatalf("nodeRegistry has no entry for the real path %q, the test would not exercise the leak", real)
	}

	other := writeWorkflow(t, filepath.Join(base, "other"), "wf_other.yaml", real)
	_, err := FromConfig(t.Context(), other)
	if err == nil || !strings.Contains(err.Error(), "not found in registries") {
		t.Errorf("FromConfig(%q) with edge %q = %v, want the name lookup to fail as on main", other, real, err)
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
	_, cached := nodeRegistry[configCacheKey{path: outsideNode}]
	registryMu.RUnlock()
	if !cached {
		t.Fatalf("nodeRegistry has no entry for %q after loading it, the test would not exercise the cache", outsideNode)
	}

	escaping := writeWorkflow(t, agentDir, "wf_cached.yaml", "../outside/join.yaml")
	if _, err := FromConfig(t.Context(), escaping); !isContainmentRejection(err) {
		t.Errorf("FromConfig(%q) from under %q with a ref to the cached node = %v, want a containment rejection", escaping, base, err)
	}
}
