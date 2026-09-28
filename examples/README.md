# ADK Go samples

This directory contains minimal examples that demonstrate one or a few ADK
features.

These examples differ from the
[google/adk-samples](https://github.com/google/adk-samples) repository, which
contains more complete end-to-end samples for customers to use or modify.

## Launcher

Many examples use the full launcher:

```go
l := full.NewLauncher()
if err := l.Execute(ctx, config, os.Args[1:]); err != nil {
    log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
}
```

The first argument selects a launcher. `console` is the default when no launcher
is specified. `web` starts an HTTP server with one or more sublaunchers:

| Command | Description |
|---|---|
| `console` | Run the agent interactively in a terminal |
| `web api` | Serve the ADK REST API |
| `web a2a` | Serve the agent over A2A |
| `web webui` | Serve the ADK web interface assets; combine it with `api` for a working UI |
| `web pubsub` | Serve the Pub/Sub trigger endpoint |
| `web eventarc` | Serve the Eventarc trigger endpoint |

Run examples from the repository root. For example:

```bash
go run ./examples/quickstart console
go run ./examples/quickstart web api
```

The `webui` sublauncher serves the frontend assets only. Combine it with `api`
to provide the backend routes used by the UI:

```bash
go run ./examples/quickstart web api webui
```

For deployments that need fewer server modes, `prod.NewLauncher()` exposes only
`web api` and `web a2a`.
