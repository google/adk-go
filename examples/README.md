# ADK GO samples
This folder hosts examples to test different features. The examples are usually minimal and simplistic to test one or a few scenarios.


**Note**: This is different from the [google/adk-samples](https://github.com/google/adk-samples) repo, which hosts more complex e2e samples for customers to use or modify directly.


# Launcher
In many examples you can see such lines:
```go
l := full.NewLauncher()
err = l.ParseAndRun(ctx, config, os.Args[1:], universal.ErrorOnUnparsedArgs)
if err != nil {
    log.Fatalf("run failed: %v\n\n%s", err, l.FormatSyntax())
}
```

it allows to decide, which launching options are supported in the run-time. 
`full.NewLauncher()` includes all major ways you can run the example:
* console
* restapi
* a2a
* webui (it can run standalone or with restapi or a2a).

Run `go run ./example/quickstart/main.go help` for details

As an alternative, you may want to use `prod.NewLauncher()` which only builds-in restapi and a2a launchers.
