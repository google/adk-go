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

// incident_responder demonstrates dynamic skill loading via the
// skillregistry. The on-call agent ships with only list_skills and
// load_skill; runbooks activate on demand.
package main

import (
	"fmt"
	"log"

	"google.golang.org/adk/skill"
	"google.golang.org/adk/skillregistry"
)

func main() {
	reg := skillregistry.New()

	must(reg.RegisterSkill(&skill.Skill{
		Frontmatter: skill.Frontmatter{
			Name:        "database-runbook",
			Description: "General database incident triage. Use for connection pool, slow query, or storage-pressure alerts.",
			Metadata:    map[string]any{"tags": []string{"database", "infra"}},
		},
		Instructions: `1. Confirm scope: is this a single database or all replicas?
2. Run the standard health probe: SELECT now(), SELECT count(*) FROM pg_stat_activity.
3. If saturation, check for runaway queries via pg_stat_statements (top 10 by total_time).
4. If storage, look at WAL growth and pending vacuum.
5. Escalate to the database SME if root cause unclear within 15 minutes.`,
	}))
	must(reg.RegisterSkill(&skill.Skill{
		Frontmatter: skill.Frontmatter{
			Name:        "replication-runbook",
			Description: "Replication-lag triage. Use when the standby is lagging the primary.",
			Metadata:    map[string]any{"tags": []string{"database", "replication"}},
		},
		Instructions: `1. Compute the lag: pg_last_wal_receive_lsn vs pg_last_wal_replay_lsn on the standby.
2. Check the network: standby → primary RTT, replication slot status.
3. If WAL retention pressure on the primary, expand replication_slots and add storage if needed.
4. If replay-bound (CPU on standby pegged), inspect pg_stat_replication for long transactions.
5. If lag exceeds the SLO for >10 minutes, fail the standby out of read traffic.`,
	}))
	must(reg.RegisterSkill(&skill.Skill{
		Frontmatter: skill.Frontmatter{
			Name:        "k8s-runbook",
			Description: "Kubernetes incident triage. Use for pod crashloop or scheduling alerts.",
			Metadata:    map[string]any{"tags": []string{"k8s", "infra"}},
		},
		Instructions: `1. kubectl get pods -A --field-selector=status.phase!=Running.
2. kubectl describe pod for any in CrashLoopBackOff; capture last 50 lines of logs.
3. Check node pressure: kubectl describe node | grep -A 5 "Conditions".
4. If recently deployed, compare the last-known-good image; consider rollback.
5. Escalate to platform team for scheduling or networking issues unresolved in 20 minutes.`,
	}))
	must(reg.RegisterSkill(&skill.Skill{
		Frontmatter: skill.Frontmatter{
			Name:        "network-runbook",
			Description: "Network connectivity triage. Use for DNS / VPC peering / firewall alerts.",
			Metadata:    map[string]any{"tags": []string{"network", "infra"}},
		},
		Instructions: `1. dig +short the target hostname from at least two regions.
2. Trace the path: mtr or traceroute from a control-plane host.
3. Verify firewall and ACL changes in the last 6 hours via the audit log.
4. Probe the VPC peering connection status if cross-account.
5. Page the network on-call if DNS/VPC issues persist past 15 minutes.`,
	}))

	// In a real agent, register the toolset + plugin on the LlmAgent:
	//
	//   list, _ := skillregistry.NewListSkillsTool(reg)
	//   load, _ := skillregistry.NewLoadSkillTool(reg)
	//   plug, _ := skillregistry.NewSkillsInstructionPlugin(
	//       skillregistry.SkillsInstructionPluginConfig{Registry: reg})
	//   agent, _ := llmagent.New(llmagent.Config{
	//       Model: gemini, Tools: []tool.Tool{list, load}, ...})
	//   runner.New(runner.Config{App: app.App{Plugins: []*plugin.Plugin{plug}, ...}})
	//
	// Here we exercise the registry directly so the demo runs offline.

	alert := struct {
		ID      string
		Type    string
		Message string
	}{
		ID:      "PD-90234",
		Type:    "db_replication_lag",
		Message: "Standby lag 412s, threshold 60s",
	}

	fmt.Printf("=== incoming alert ===\n%s [%s]: %s\n\n", alert.ID, alert.Type, alert.Message)

	// Step 1: agent calls list_skills(query="database").
	fmt.Println("=== agent calls list_skills(query=\"database\") ===")
	candidates := reg.List(skillregistry.Filter{Query: "database"})
	for _, fm := range candidates {
		fmt.Printf("  - %s — %s\n", fm.Name, fm.Description)
	}

	// Step 2: agent picks replication-runbook (alert type matches).
	chosen := "replication-runbook"
	fmt.Printf("\n=== agent calls load_skill(%q) ===\n", chosen)
	s, err := reg.Get(chosen)
	if err != nil {
		log.Fatalf("Get: %v", err)
	}
	fmt.Println("loaded skill instructions:")
	fmt.Println(indent(s.Instructions, "  | "))

	// Step 3: at this point the SkillsInstructionPlugin would inject
	// the loaded skill's body into the LLM system instruction on the
	// next turn. The agent then triages the alert following the
	// runbook. We simulate the triage decision deterministically:
	fmt.Println("\n=== agent triage output ===")
	fmt.Println("• Computed lag: 412s (well over 60s SLO).")
	fmt.Println("• Checked network: RTT 18ms, slot active. Not network-bound.")
	fmt.Println("• pg_stat_replication shows replay-bound; CPU 92% on standby.")
	fmt.Println("• Decision: fail standby out of read traffic and page database SME.")
	fmt.Println("• Skills NOT loaded for this incident: k8s-runbook, network-runbook")
	fmt.Println("  (kept dormant so the LLM context didn't include them).")
}

func must(err error) {
	if err != nil {
		log.Fatalf("%v", err)
	}
}

func indent(s, prefix string) string {
	out := prefix
	for _, c := range s {
		out += string(c)
		if c == '\n' {
			out += prefix
		}
	}
	return out
}
