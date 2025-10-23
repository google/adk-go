# ADK GO samples
This folder hosts examples to test different features. The examples are usually minimal and simplistic to test one or a few scenarios.


**Note**: This is different from the [google/adk-samples](https://github.com/google/adk-samples) repo, which hosts more complex e2e samples for customers to use or modify directly.


# Launcher
In many examples you can see such a line:
```go
err = universal.Run(ctx, config)
```
it allows you to choose running mode using command line arguments:
 - empty argument list executes the application using console interface
 - optionally you may provide the mode in the first argument: 
    - api - run only ADK REST API server
    - apiweb - run ADK REST API server together with ADK Web UI  (embeds web UI static files)
    - console - run simple console app

You may also decide to run the application in one particular mode - uncomment the appropiate one:
```go
// err= universal.Run(ctx, config)
// err= console.Run(ctx, config)
// err= api.Run(ctx, config)
err= apiweb.Run(ctx, config)
if err != nil {
    log.Fatalf("run failed: %v", err)
}
```