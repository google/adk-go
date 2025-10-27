# ADK GO samples
This folder hosts examples to test different features. The examples are usually minimal and simplistic to test one or a few scenarios.


**Note**: This is different from the [google/adk-samples](https://github.com/google/adk-samples) repo, which hosts more complex e2e samples for customers to use or modify directly.


# Launcher
In many examples you can see such lines:
```go
l := full.NewLaucher("weather_time_agent")
err = l.ParseAndRun(ctx, config, os.Args[1:], universal.ErrorOnUnparsedArgs)
if err != nil {
    log.Fatalf("run failed: %v\n\n%s", err, l.FormatSyntax())
}
```

it allows to to decide, which launching options are supported in the run-time. 
`full.NewLaucher("weather_time_agent")`
means
`universal.NewLauncher(console.NewLauncher(rootAgentName), web.NewLauncher(api.NewLauncher(), a2a.NewLauncher(rootAgentName), webui.NewLauncher()))`
 - in that case supported options are either console or web. For web you can enable independently api (ADK REST API), a2a and webui (ADK Web UI).

Run `go run ./example/quickstart/main.go help` for details


As an alternative, you may want to use `prod`
`prod.NewLaucher("weather_time_agent")`
meaning
`universal.NewLauncher(web.NewLauncher(api.NewLauncher(), a2a.NewLauncher(rootAgentName)))`
- the only supported options is web. For web you can enable independently api (ADK REST API) or a2a.


