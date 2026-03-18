# zane-fastsearch (Go)

High-performance web search helper for ZANE synthesis research workflows.

## Build

```bash
go build -o tools/bin/zane-fastsearch ./tools/go/fastsearch
```

Or from project root:

```bash
make build-go-fastsearch
```

## Use Directly

```bash
tools/bin/zane-fastsearch --query "EGFR inhibitor synthesis route" --max-results 5
```

Outputs JSON array:

```json
[
  {
    "title": "Example result",
    "url": "https://example.org",
    "snippet": "",
    "source": "go-fastsearch"
  }
]
```

## Integrate with Python Pipeline

Set environment variable so `InternetSearchClient` can invoke this binary:

```bash
export ZANE_GO_SEARCH_BIN="$PWD/tools/bin/zane-fastsearch"
```

After this, synthesis research commands can automatically use the Go backend for faster retrieval when Google CSE is not configured.
