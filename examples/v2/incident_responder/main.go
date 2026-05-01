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

// On-call incident-responder agent. Demonstrates skillregistry: the
// agent ships with only list_skills and load_skill; runbooks activate
// on demand and their instructions land in the system prompt via
// SkillsInstructionPlugin.
package main

import (
	"context"
	"log"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/skill"
	"google.golang.org/adk/skillregistry"
	"google.golang.org/adk/tool"
)

const instruction = `You are an on-call incident responder.

A page comes in describing a production alert. Your workflow:

  1. Call list_skills(query="<keywords from the alert>") to discover
     relevant runbooks. Filter by tags too if helpful (e.g. "database",
     "k8s", "network").
  2. Call load_skill(name="<runbook>") for the most relevant runbook.
     Its instructions become part of your context starting now.
  3. Follow the loaded runbook step by step. Report findings,
     decisions, and escalations clearly.

Do NOT guess at procedures. If no runbook matches, say so and ask the
user for the right runbook name or escalation contact.`

func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, modelName(), &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	reg := buildSkillRegistry()
	listSkills, err := skillregistry.NewListSkillsTool(reg)
	if err != nil {
		log.Fatal(err)
	}
	loadSkill, err := skillregistry.NewLoadSkillTool(reg)
	if err != nil {
		log.Fatal(err)
	}
	skillPlugin, err := skillregistry.NewSkillsInstructionPlugin(skillregistry.SkillsInstructionPluginConfig{
		Registry: reg,
	})
	if err != nil {
		log.Fatal(err)
	}

	rootAgent, err := llmagent.New(llmagent.Config{
		Name:        "incident_responder",
		Model:       model,
		Description: "An on-call agent that runs the right runbook for the incident.",
		Instruction: instruction,
		Tools:       []tool.Tool{listSkills, loadSkill},
	})
	if err != nil {
		log.Fatal(err)
	}

	cfg := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(rootAgent),
		PluginConfig: runner.PluginConfig{
			Plugins: []*plugin.Plugin{skillPlugin},
		},
	}
	l := full.NewLauncher()
	if err = l.Execute(ctx, cfg, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}

func modelName() string {
	if v := os.Getenv("GOOGLE_GENAI_MODEL"); v != "" {
		return v
	}
	return "gemini-2.5-flash"
}

func buildSkillRegistry() *skillregistry.Registry {
	reg := skillregistry.New()
	must(reg.RegisterSkill(&skill.Skill{
		Frontmatter: skill.Frontmatter{
			Name:        "database-runbook",
			Description: "General database incident triage. Use for connection-pool, slow-query, or storage-pressure alerts.",
			Metadata:    map[string]any{"tags": []string{"database", "infra"}},
		},
		Instructions: `Database triage:
1. Confirm scope: single database or all replicas?
2. Run health probe: SELECT now(); SELECT count(*) FROM pg_stat_activity.
3. If saturation, inspect pg_stat_statements (top 10 by total_time).
4. If storage, check WAL growth and pending vacuum.
5. Escalate to the database SME if root cause unclear within 15 minutes.`,
	}))
	must(reg.RegisterSkill(&skill.Skill{
		Frontmatter: skill.Frontmatter{
			Name:        "replication-runbook",
			Description: "Replication-lag triage. Use when the standby lags the primary or you see db_replication_lag alerts.",
			Metadata:    map[string]any{"tags": []string{"database", "replication"}},
		},
		Instructions: `Replication-lag triage:
1. Compute the lag: pg_last_wal_receive_lsn vs pg_last_wal_replay_lsn on the standby.
2. Check the network: standby->primary RTT, replication slot status.
3. If WAL retention pressure on the primary, expand replication_slots and add storage.
4. If replay-bound (CPU pegged on standby), inspect pg_stat_replication for long transactions.
5. If lag exceeds the SLO for >10 minutes, fail the standby out of read traffic.`,
	}))
	must(reg.RegisterSkill(&skill.Skill{
		Frontmatter: skill.Frontmatter{
			Name:        "k8s-runbook",
			Description: "Kubernetes incident triage. Use for pod-crashloop or scheduling alerts.",
			Metadata:    map[string]any{"tags": []string{"k8s", "infra"}},
		},
		Instructions: `Kubernetes triage:
1. kubectl get pods -A --field-selector=status.phase!=Running.
2. kubectl describe pod for any in CrashLoopBackOff; capture last 50 lines of logs.
3. Check node pressure: kubectl describe node | grep -A 5 "Conditions".
4. If recently deployed, compare last-known-good image; consider rollback.
5. Escalate to platform team for scheduling or networking issues unresolved in 20 minutes.`,
	}))
	must(reg.RegisterSkill(&skill.Skill{
		Frontmatter: skill.Frontmatter{
			Name:        "network-runbook",
			Description: "Network connectivity triage. Use for DNS/VPC-peering/firewall alerts.",
			Metadata:    map[string]any{"tags": []string{"network", "infra"}},
		},
		Instructions: `Network triage:
1. dig +short the target hostname from at least two regions.
2. Trace the path: mtr or traceroute from a control-plane host.
3. Verify firewall and ACL changes in the last 6 hours via the audit log.
4. Probe the VPC peering connection status if cross-account.
5. Page the network on-call if DNS/VPC issues persist past 15 minutes.`,
	}))
	return reg
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
