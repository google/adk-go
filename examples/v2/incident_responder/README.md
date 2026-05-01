# incident_responder

On-call agent that holds a library of runbook **skills** but only loads
the ones relevant to the current incident, keeping the LLM context lean.

Scenario: a paging system fires `db_replication_lag` alert. The agent:

1. Calls `list_skills(query="database")` — the registry returns the
   `database-runbook` and `replication-runbook` frontmatters.
2. Calls `load_skill("replication-runbook")` — the skill's instructions
   become part of the agent's system prompt on the next turn (via the
   `SkillsInstructionPlugin`), and the skill's body is also returned in
   the `load_skill` response so the model has immediate access.
3. Follows the runbook to triage the alert.

Compare to declaring all 4 runbooks in the agent's instructions
upfront — that would balloon the system prompt and blur the focus on
every turn, even when the incident has nothing to do with databases.

The skills here are inline strings; in production you'd load them
from a directory of SKILL.md files (see Python's
`load_skill_from_dir`) or from object storage.
