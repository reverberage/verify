# rvrb-verify

**Claim verification engine.** Claim in, verdict out.

Part of the [reverberage](https://github.com/reverberage) ecosystem — composable MCP-native toolkits for audio, video, and text.

## Install

```bash
pip install rvrb-verify
```

Requires `DASHSCOPE_API_KEY` env var (or install `n3rverberage` for multi-provider support).

## Use

### CLI

```bash
rvrb-verify "The sky is blue"
rvrb-verify "Water boils at 100°C" --strategy research --json
```

### Python

```python
from rvrb_verify import verify

verdict = verify("The sky is green")

print(f"Verdict: {verdict.verdict.value}")
print(f"Confidence: {verdict.confidence:.1%}")
print(f"Summary: {verdict.summary}")
```

### MCP

```bash
rvrb-verify-mcp
```

Compatible with any MCP client (Claude Desktop, Continue.dev, etc.).

## Strategies

| Strategy | Description | Tools |
|----------|-------------|-------|
| `fact-check` | General claim verification | Web search, news search |
| `legal` | Legal analysis | Statute search, case law search |
| `research` | Academic validation | Paper search, arXiv search |

## License

Apache-2.0 — same as the reverberage ecosystem.
