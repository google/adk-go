# Configurable agents (internal)

This config loader is used by the repository's conformance CLI in
`cmd/internal/adkcli`. It is not a supported API for applications outside this
repository, and the public `cmd/adkgo` command does not expose it.

## Local MCP servers

YAML configs using `McpToolset` with stdio transport require an operator-provided
JSON policy. Without one, local MCP servers are denied. Each entry approves an
executable and its complete argument list:

```json
{
  "allowed_servers": [
    {
      "command": "/opt/mcp/bin/filesystem-server",
      "args": ["/srv/shared"]
    }
  ]
}
```

The command must be an absolute path to an installed executable. Policy and agent
commands are resolved through symlinks; agent commands may also be found through
`PATH`. Matching compares the resolved paths and every argument in order. No
prefixes or wildcards are supported. Use `args: []` for no arguments; `[""]`
passes one empty string. Missing or null `args`, null array elements, and unknown
fields are rejected. An empty `allowed_servers` array denies all local servers.
Field matching is case-insensitive; duplicate keys are accepted.

Pass the policy flag **before** any launcher arguments:

```sh
go build -o /tmp/adkcli ./cmd/internal/adkcli
cd /path/to/agent-configs
/tmp/adkcli --mcp-policy /trusted/mcp-policy.json console
```

`--mcp-policy=/trusted/mcp-policy.json` is also accepted. A missing or invalid
policy file stops startup. Agent configs rejected by the policy are logged and
skipped. Keep the policy under operator control, separate from agent configs;
the CLI only reads the file supplied through this flag.

The process starts when tools are first listed, using the first matching policy
entry's path. This preserves that entry's virtual environment or program mode.
Entries with identical resolved paths and arguments always select the first
entry, even when the config names a later entry's symlink.

Targets are checked again before startup; changed targets are rejected. Loading
under another policy rechecks approval without revoking existing agents or toolsets.

The policy does not sandbox approved programs. Their executables, dependencies,
environment, and referenced files must remain trusted. Subprocesses inherit the
directory the CLI scans; `python -m`, for example, can load modules from it, so
this directory must also be trusted. Startup checks cannot prevent filesystem
changes between checking and executing, or detect replacement of a binary at
the same resolved path.

Direct construction of an MCP toolset in Go code is unchanged.
