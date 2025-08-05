# LangExtract MCP Server Setup Guide

## Quick Setup (No Config Files Needed!)

This MCP server doesn't use separate configuration files. Everything is handled through environment variables and tool parameters.

### Step 1: Get Your API Key
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Copy the key (keep it secure!)

### Step 2: Install with Claude Code
```bash
# Single command installation - no config files needed!
claude mcp add langextract-mcp -e LANGEXTRACT_API_KEY=your-gemini-api-key -- uv run --with fastmcp fastmcp run src/langextract_mcp/server.py
```

That's it! The server will start automatically when Claude Code needs it.

## Configuration Details

### Environment Variables (Set Once)
```bash
# Required
LANGEXTRACT_API_KEY=your-gemini-api-key

# Optional
LANGEXTRACT_DEFAULT_MODEL=gemini-2.5-flash
LANGEXTRACT_MAX_WORKERS=10
```

### Per-Request Configuration (In Tool Calls)
When using tools, you can configure behavior per request:

```python
{
    "text": "Your text to extract from",
    "config": {
        "model_id": "gemini-2.5-flash",     # Which model to use
        "temperature": 0.5,                 # Randomness (0.0-1.0)
        "extraction_passes": 1,             # How many extraction attempts
        "max_workers": 10                   # Parallel processing
    }
}
```

## Verification
After installation, ask Claude Code:
```
Use the get_server_info tool to show the LangExtract server status
```

You should see:
- Server running: ✅
- API key configured: ✅
- Optimization features enabled: ✅

## Troubleshooting

**"Server not found"**
```bash
# Check if registered
claude mcp list

# Re-add if missing
claude mcp add langextract-mcp -e LANGEXTRACT_API_KEY=your-key -- uv run --with fastmcp fastmcp run src/langextract_mcp/server.py
```

**"API key not set"**
```bash
# Check environment
echo $LANGEXTRACT_API_KEY

# Set if missing (permanent)
echo 'export LANGEXTRACT_API_KEY=your-key' >> ~/.bashrc
source ~/.bashrc
```

**"Tools not working"**
- Verify API key is valid at [Google AI Studio](https://aistudio.google.com/app/apikey)
- Check network connectivity
- Try with different model (e.g., "gemini-2.5-pro")
