# ADK Go Issue Triaging Agent

This is an automatic issue triaging and labeling bot for the ADK Go repository, implemented in Go using the ADK framework.
It automatically analyzes GitHub issues and assigns appropriate labels and owners based on the issue content.

## Overview

The triaging agent uses Google's Gemini model to analyze GitHub issues and automatically:
- Assign appropriate labels based on issue content
- Assign issues to the correct owner based on the label
- Change issue types (currently limited to `Bug` and `Feature`) when applicable
- Process both individual issues (on open/reopen) and batches of unlabeled issues

### Environment Variables

The following environment variables must be set:

**Required:**
- `GITHUB_TOKEN`: GitHub personal access token or GitHub Actions token with issue write permissions
- `GOOGLE_API_KEY`: Google API key for Gemini model access

**Optional:**
- `OWNER`: GitHub repository owner (default: "google")
- `REPO`: Repository name (default: "adk-go")
- `EVENT_NAME`: Event type ("issues" for issue events, or "schedule" for batch processing)
- `ISSUE_NUMBER`: Specific issue number to process (when EVENT_NAME is "issues")
- `ISSUE_TITLE`: Issue title (optional, fetched from API if not provided)
- `ISSUE_BODY`: Issue body (optional, fetched from API if not provided)
- `ISSUE_COUNT_TO_PROCESS`: Number of issues to process in batch mode (default: 3)
- `INTERACTIVE`: Set to "1" or "true" for interactive mode requiring approval (default: "0")
- `GOOGLE_GENAI_USE_VERTEXAI`: Set to "1" to use Vertex AI instead of API key (default: "0")
- `GEMINI_MODEL`: Specific gemini model. Default is `gemini-2.5-pro`

### GitHub Actions Setup

The agent is configured to run automatically via GitHub Actions. To set it up:

1. **Add Secrets to Repository**:
    - Go to repository Settings → Secrets and variables → Actions
    - Add `GITHUB_TOKEN`: GitHub token with issue write permissions
    - Add `GOOGLE_API_KEY`: Your Google API key

2. **Workflow Configuration**:
   The workflow file (`.github/workflows/triage.yml`) is already configured to:
    - Trigger on `issues.opened` and `issues.reopened` events
    - Run the agent with appropriate environment variables
    - Use the repository's setup action for Go environment

## Usage

### Running Locally

To run the triaging agent locally:

```bash
cd contributing/samples/triaging_agent

# Set required environment variables
export GITHUB_TOKEN="your_github_token"
export GOOGLE_API_KEY="your_google_api_key"
export OWNER="google"
export REPO="adk-go"

# For processing a specific issue
export EVENT_NAME="issues"
export ISSUE_NUMBER="123"

# Run the agent
go run .
```

### Processing Modes

#### Single Issue Mode
When `EVENT_NAME=issues` and `ISSUE_NUMBER` is set, the agent processes a specific issue:

```bash
export EVENT_NAME="issues"
export ISSUE_NUMBER="123"
go run .
```

#### Batch Mode
When processing multiple unlabeled issues:

```bash
export EVENT_NAME="schedule"
export ISSUE_COUNT_TO_PROCESS="5"
go run .
```

## References

- [ADK Python Triaging Agent](https://github.com/google/adk-python/tree/main/contributing/samples/adk_triaging_agent) - Original Python implementation
- [GitHub Actions Workflow](../.github/workflows/triage.yml) - Automation configuration


