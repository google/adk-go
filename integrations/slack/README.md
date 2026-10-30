# Slack integration for ADK Go

Go module `google.golang.org/adk/integrations/slack`: Slack-specific packages
for agents built with [ADK Go](https://github.com/google/adk-go). It is a
separate module from `google.golang.org/adk/v2`, so add it to your `go.mod` on
its own:

```sh
go get google.golang.org/adk/integrations/slack
```

## Packages

- `mrkdwn` converts the Markdown that models write into Slack mrkdwn (bold,
  italic, strikethrough, headings, lists, links, code and the `& < >` escaping
  Slack requires), so a reply posted through the Slack API renders as intended
  instead of showing literal asterisks.
