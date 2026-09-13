# Configurable agents (internal)

This config loader is used by the repository's conformance CLI in
`cmd/internal/adkcli`. It is not a supported API for applications outside this
repository, and the public `cmd/adkgo` command does not expose it.

## Local MCP servers

Agent YAML is not permission to execute a local program. Configuring a
`McpToolset` with stdio transport is denied by default. The operator must provide
an independent JSON policy approving each executable and its **complete argument
list**:

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
prefixes or wildcards are supported. `args: []` permits only zero arguments;
missing or null `args`, or a null element in that array, is an error. An explicit
empty string argument (`[""]`) is valid and differs from zero arguments. An empty `allowed_servers` array denies all
local servers. Unknown policy fields and invalid policy files are rejected.

Pass the policy flag **before** any launcher arguments:

```sh
go build -o /tmp/adkcli ./cmd/internal/adkcli
cd /path/to/agent-configs
/tmp/adkcli --mcp-policy /trusted/mcp-policy.json console
```

`--mcp-policy=/trusted/mcp-policy.json` is also accepted. A missing or invalid
policy file stops startup. Agent configs rejected by the policy are skipped with
the CLI's existing config-loading error message. The policy path is never taken
from agent YAML or automatically discovered in the scanned directory. Keep the
policy under operator control, separate from untrusted agent configs.

Authorization is scoped to a loading context and inherited by nested agent
loads. Cached agents are separated by policy identity. A toolset's command is
approved when its configuration is resolved, but the process starts lazily when
tools are first listed. Startup uses the absolute invocation path from the
matching **policy entry**, preserving virtual environments and program modes
selected by the executable name. A config alias cannot choose a different
invocation path. If multiple entries match the same target and arguments, the
first entry supplies that path. Before each startup (including reconnection),
the path is resolved again; a changed target is rejected.
Changing a policy does not revoke previously constructed agents or toolsets.
Tool filters and tool-call confirmation do not authorize server startup.

This is execution authorization, not a sandbox. The approved executable, scripts,
packages, inherited environment, working directory, and referenced files must
remain trusted. The startup check does not lock the filesystem: it cannot prevent
changes between checking and executing, or detect replacement of a binary at the
same resolved path. Pin launcher packages/images and all arguments to the intended
resources; approval of a program does not make that program safe.

Direct construction of an MCP toolset in Go code is unchanged. ADK Python also
rejects config-defined stdio servers by default, but its
`ADK_ALLOW_CONFIG_STDIO_MCP_SERVERS` switch trusts configs wholesale. This loader
requires exact command/argument approvals instead; it does not use that switch.

## Testing

The policy tests run offline. Factory tests exercise `ResolveToolReference`,
including default denial, exact arguments, executable identity, and independent
loading contexts. A child copy of the Go test binary writes a temporary marker to
verify lazy startup and the approved invocation path without launching an
installed MCP server or contacting a model. Retargeting a policy symlink before
startup is rejected, and null policy arguments fail decoding. CLI tests verify that the policy flag supplies authorization to the factory.
