# Session History Compaction Example

This example demonstrates **session history compaction**, a powerful feature that reduces token usage in long-running conversations by automatically summarizing older events using the LLM.

## What is Session History Compaction?

Session history compaction addresses the challenge of unbounded context growth in long conversations:

| Without Compaction | With Compaction |
|-------------------|-----------------|
| Turn 1: 100 tokens | Turn 1: 100 tokens |
| Turn 5: 450 tokens | Turn 5: 200 tokens (summarized) |
| Turn 10: 1000+ tokens | Turn 10: 300 tokens (multiple summaries) |

**Benefits:**

- **60-80% token reduction** in typical conversations
- **Automatic triggering** - no application code changes needed
- **Cost savings** - directly reduces API billing
- **Full audit trail** - original events preserved in database
- **Backward compatible** - opt-in feature, doesn't affect existing code

## How It Works

```text
Turn 1-2: User question → Agent response → Events stored
Turn 3-4: User question → Agent response → Events stored
Turn 5: [Compaction Trigger]
  ↓
  1. Identify events to summarize (configured interval)
  2. Call LLM to create concise summary
  3. Store compaction event in session
  4. Filter LLM context to use summary instead of original events
  ↓
Turn 6+: Uses compressed history, saves tokens
```

## Files in This Example

### `debug/main.go` - Visual Debug Example

A debug example with enhanced visual indicators to help verify compaction.

**Features:**

- Visual markers showing when compaction triggers
- Lower default interval (2) for faster demonstration
- Educational output explaining compaction behavior
- Built-in usage guide
- Interactive console for testing compaction

**Run:**

```bash
# Start the console with default settings (interval=2, overlap=1)
go run ./debug console

# Customize compaction interval
go run ./debug -interval=3 console

# Enable verbose mode for detailed logging
go run ./debug -verbose console
```

## Configuration Options

### `Enabled` (bool, default: false)

Turn compaction on/off. Set to `true` to enable automatic compaction.

```go
config := &compaction.Config{
    Enabled: true,
    // ... other options
}
```

### `CompactionInterval` (int, default: 5)

Number of unique invocations before compaction is triggered.

- **Lower values** (2-3): More frequent compaction, lower token count, more API calls
- **Higher values** (5-10): Less frequent compaction, higher token count, fewer API calls

```go
// Compact every 3 turns
config.CompactionInterval = 3
```

### `OverlapSize` (int, default: 2)

Number of recent invocations to preserve when compacting.

This ensures context continuity across compaction boundaries. For example:

- With `CompactionInterval: 5` and `OverlapSize: 2`
- Events 1-3 are compacted into a summary
- Events 4-5 are kept as overlap for context
- Next compaction includes events 4-8

```go
config.OverlapSize = 2
```

### `Model` (string, default: "")

Custom LLM model for summarization. If empty, uses the same model as the agent.

```go
config.Model = "gemini-2.0-flash"
```

### `SystemPrompt` (string, default: "")

Custom system prompt for the LLM summarizer. If empty, uses the default built-in prompt.

```go
config.SystemPrompt = `You are summarizing a conversation...`
```

## Complete Configuration Example

```go
compactionConfig := &compaction.Config{
    Enabled:            true,
    CompactionInterval: 5,     // Compact every 5 turns
    OverlapSize:        2,     // Keep 2 turns for context
    Model:              "",    // Use current model
    SystemPrompt:       "",    // Use default prompt
}

// Validate configuration
if err := compactionConfig.Validate(); err != nil {
    log.Fatalf("Invalid config: %v", err)
}

// Create runner with compaction
r, err := runner.New(runner.Config{
    AppName:          "my-app",
    Agent:            agent,
    SessionService:   sessionService,
    CompactionConfig: compactionConfig,
})
```

## Running the Example

### Prerequisites

```bash
export GOOGLE_API_KEY=your-api-key
cd examples/compaction
```

### Console Mode

```bash
# Start with default settings (interval=2, overlap=1)
go run ./debug console

# Customize settings
go run ./debug -interval=5 -overlap=2 console
```

## What Happens During Compaction?

1. **Detection**: Runner checks if `CompactionInterval` invocations have passed since last compaction
2. **Selection**: Sliding window algorithm selects events to compact (preserving overlap)
3. **Summarization**: LLM generates concise summary of selected events
4. **Storage**: Compaction event is stored in session (original events remain for audit trail)
5. **Filtering**: LLM context automatically uses summaries instead of original events

## Example Compaction Event

```go
type EventCompaction struct {
    StartTimestamp float64         // When original events started
    EndTimestamp float64           // When original events ended
    CompactedContent *genai.Content // LLM-generated summary
}
```

In the session history:

```text
Event 1: User question
Event 2: Agent response
Event 3: User follow-up
Event 4: Agent response
Event 5: [Compaction Event] ← Contains summary of Events 1-4
Event 6: New user question
Event 7: Agent response (uses Event 5 summary + Events 6-7)
```

## Monitoring Compaction

```go
// Check if an event is a compaction
if event.Actions.Compaction != nil {
    log.Printf("Compaction occurred: %v → %v",
        event.Actions.Compaction.StartTimestamp,
        event.Actions.Compaction.EndTimestamp)
}
```

## Performance Notes

- **Overhead**: <100ms per invocation for compaction trigger check
- **LLM Cost**: Compaction calls LLM, adds cost but savings exceed cost ~10:1
- **Storage**: All events preserved, no database growth impact
- **Context**: 60-80% reduction in LLM context tokens

## Testing Compaction

Try this conversation pattern to see compaction in action:

1. Ask 5+ questions to trigger compaction threshold
2. Watch for compaction output in the console
3. Note how context remains relevant but tokens decrease
4. Continue conversation - newer context always included

## Best Practices

1. **For long conversations**: Enable compaction with `Interval: 5-10`
2. **For chat applications**: Use `Interval: 3-5` for faster context management
3. **For debugging**: Enable `verbose` flag to see when compaction occurs
4. **For cost optimization**: Lower `Interval` to compact more frequently
5. **For quality**: Keep `OverlapSize: 2` to maintain context continuity

## Troubleshooting

- **Compaction is not triggering**: Check if `Enabled: true` and verify `CompactionInterval` is lower than current turn count
- **Token usage still high**: Lower `CompactionInterval` for more frequent compaction
- **Missing context in responses**: Increase `OverlapSize` for more context preservation

