# Claude Code Integration Guide

This guide walks you through setting up the LangExtract MCP Server with Claude Code for seamless text extraction capabilities.

## Prerequisites

1. **Claude Code** installed and working
2. **API Key** from Google AI Studio for Gemini models
3. **Python 3.10+** available on your system

## Installation Methods

### Method 1: FastMCP CLI (Recommended)

This is the easiest method for Claude Code integration:

```bash
# Basic installation
fastmcp install claude-code claude_code_install.py --env LANGEXTRACT_API_KEY=your-gemini-api-key

# With explicit dependencies (more reliable)
fastmcp install claude-code claude_code_install.py \
    --with langextract \
    --with fastmcp \
    --with pydantic \
    --env LANGEXTRACT_API_KEY=your-gemini-api-key
```

### Method 2: Manual Installation

If you prefer manual setup:

```bash
# Add server to Claude Code
claude mcp add langextract-server -- uv run --with fastmcp fastmcp run claude_code_install.py

# Set environment variable separately
export LANGEXTRACT_API_KEY=your-gemini-api-key
```

### Method 3: Development Setup

For development or customization:

```bash
# Clone and setup
git clone https://github.com/your-org/langextract-mcp.git
cd langextract-mcp
pip install -e .

# Add to Claude Code with local path
claude mcp add langextract-dev -- python -m langextract_mcp.server
```

## Getting Your API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Copy the key (keep it secure!)
4. Use it in the installation command above

## Verification

After installation, you can verify the server is working:

1. **Open Claude Code**
2. **Check available tools** - you should see:
   - `extract_from_text`
   - `extract_from_url`
   - `save_extraction_results`
   - `generate_visualization`
   - `list_supported_models`
   - `get_server_info`

3. **Test with a simple extraction**:
   ```
   Use the extract_from_text tool to extract names from: "Hello, I'm John Smith and this is Jane Doe."
   ```

## Usage Examples

### Basic Text Extraction

```python
# Tool: extract_from_text
{
    "text": "Dr. Smith prescribed 500mg amoxicillin twice daily.",
    "prompt_description": "Extract medication information including drug names and dosages.",
    "examples": [
        {
            "text": "Take 250mg ibuprofen every 4 hours.",
            "extractions": [
                {
                    "extraction_class": "medication",
                    "extraction_text": "ibuprofen",
                    "attributes": {"type": "NSAID"}
                },
                {
                    "extraction_class": "dosage",
                    "extraction_text": "250mg", 
                    "attributes": {"amount": "250", "unit": "mg"}
                }
            ]
        }
    ],
    "config": {
        "model_id": "gemini-2.5-flash",
        "temperature": 0.3
    }
}
```

### URL-based Extraction

```python
# Tool: extract_from_url
{
    "url": "https://example.com/article.html",
    "prompt_description": "Extract key facts and main points from this article.",
    "examples": [
        {
            "text": "The study found that 85% of participants improved.",
            "extractions": [
                {
                    "extraction_class": "statistic",
                    "extraction_text": "85% of participants improved",
                    "attributes": {"percentage": "85", "outcome": "improved"}
                }
            ]
        }
    ]
}
```

## Performance Features

This MCP server includes several optimizations for Claude Code:

- **Persistent Connections**: Language model clients are cached across requests
- **Schema Caching**: Generated schemas are reused for similar examples
- **Connection Pooling**: Maintains API connections for better latency
- **Memory Efficiency**: Smart caching prevents memory leaks

## Security Features

- **Server-side API Keys**: Your API key never leaves the server
- **No Client Credentials**: Claude Code never sees sensitive information
- **Secure Transport**: Uses STDIO for secure communication
- **Environment Variables**: Credentials stored securely in environment

## Troubleshooting

### Server Not Found
```bash
# Check if server is registered
claude mcp list

# Re-add if missing
fastmcp install claude-code claude_code_install.py --env LANGEXTRACT_API_KEY=your-key
```

### API Key Issues
```bash
# Check if environment variable is set
echo $LANGEXTRACT_API_KEY

# Set if missing
export LANGEXTRACT_API_KEY=your-gemini-api-key
```

### Tool Not Available
```bash
# Check server info
Use get_server_info tool in Claude Code

# Should show claude_code_ready: true
```

### Performance Issues
```bash
# Check connection caching
Use get_server_info tool to see optimization status

# Server should show cached connections
```

## Advanced Configuration

### Custom Model Settings

You can configure different models per request:

```python
{
    "config": {
        "model_id": "gemini-2.5-pro",    # Higher quality
        "temperature": 0.2,              # More deterministic
        "extraction_passes": 3,          # Better recall
        "max_workers": 15               # Faster processing
    }
}
```

### Environment Variables

```bash
# Required
export LANGEXTRACT_API_KEY=your-gemini-api-key

# Optional tuning
export LANGEXTRACT_DEFAULT_MODEL=gemini-2.5-pro
export LANGEXTRACT_MAX_WORKERS=20
```

## Best Practices

1. **Start Simple**: Begin with basic examples, then add complexity
2. **Good Examples**: Provide clear, diverse examples for better results
3. **Appropriate Models**: Use `gemini-2.5-flash` for speed, `gemini-2.5-pro` for quality
4. **Monitor Usage**: Be aware of API costs with multiple passes
5. **Cache Results**: Save important results for future reference

## Support

- **GitHub Issues**: Report problems or feature requests
- **Documentation**: Check the main README for detailed information
- **Examples**: See the examples directory for more use cases

## Security Note

This server follows FastMCP security best practices:
- API keys are never exposed to MCP clients
- All sensitive operations happen server-side
- Uses secure STDIO transport protocol
- Follows principle of least privilege

Happy extracting with Claude Code! 🚀