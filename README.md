# LangExtract MCP Server

A FastMCP server that provides Model Context Protocol (MCP) tools for Google's [langextract](https://github.com/google/langextract) library. This server enables AI assistants like Claude Code to extract structured information from unstructured text using Large Language Models through a secure, optimized MCP interface.

<a href="https://glama.ai/mcp/servers/@larsenweigle/langextract-mcp">
  <img width="380" height="200" src="https://glama.ai/mcp/servers/@larsenweigle/langextract-mcp/badge" alt="LangExtract Server MCP server" />
</a>

## Overview

LangExtract is a Python library that uses LLMs to extract structured information from text documents while maintaining precise source grounding. This MCP server exposes langextract's capabilities through the Model Context Protocol with advanced performance optimizations and enterprise-grade security.

The server includes intelligent caching, persistent connections, and server-side credential management to provide optimal performance in long-running environments like Claude Code while maintaining complete security isolation.

## Quick Setup for Claude Code

### Prerequisites

- Claude Code installed and configured
- Google Gemini API key ([Get one here](https://aistudio.google.com/app/apikey))
- Python 3.10 or higher

### Installation

Install directly into Claude Code using the FastMCP CLI:

```bash
fastmcp install claude-code claude_code_install.py --env LANGEXTRACT_API_KEY=your-gemini-api-key
```

The server will automatically start and integrate with Claude Code. No additional configuration is required.

### Verification

After installation, verify the integration by asking Claude Code:

```
Use the get_server_info tool to show the LangExtract server capabilities
```

You should see output indicating the server is running with optimization features enabled.

## Available Tools

The server provides six MCP tools optimized for text extraction workflows:

**Core Extraction**
- `extract_from_text` - Extract structured information from provided text
- `extract_from_url` - Extract information from web content
- `save_extraction_results` - Save results to JSONL format
- `generate_visualization` - Create interactive HTML visualizations

**Server Information**
- `list_supported_models` - View available language models and recommendations  
- `get_server_info` - Check server status and capabilities

## Usage Examples

### Basic Text Extraction

Ask Claude Code to extract information using natural language:

```
Extract medication information from this text: "Patient prescribed 500mg amoxicillin twice daily for infection"

Use these examples to guide the extraction:
- Text: "Take 250mg ibuprofen every 4 hours"
- Expected: medication=ibuprofen, dosage=250mg, frequency=every 4 hours
```

### Advanced Configuration

For complex extractions, specify configuration parameters:

```
Extract character emotions from Shakespeare using:
- Model: gemini-2.5-pro for better literary analysis
- Multiple passes: 3 for comprehensive extraction
- Temperature: 0.2 for consistent results
```

### URL Processing

Extract information directly from web content:

```
Extract key findings from this research paper: https://arxiv.org/abs/example
Focus on methodology, results, and conclusions
```

## Supported Models

This server currently supports **Google Gemini models only**, optimized for reliable structured extraction with advanced schema constraints:

- `gemini-2.5-flash` - **Recommended default** - Optimal balance of speed, cost, and quality
- `gemini-2.5-pro` - Best for complex reasoning and analysis tasks requiring highest accuracy

The server uses persistent connections, schema caching, and connection pooling for optimal performance with Gemini models. Support for additional providers may be added in future versions.

## Configuration Reference

### Environment Variables

Set during installation or in server environment:

```bash
LANGEXTRACT_API_KEY=your-gemini-api-key  # Required
```

### Tool Parameters

Configure extraction behavior through tool parameters:

```python
{
    "model_id": "gemini-2.5-flash",     # Language model selection
    "max_char_buffer": 1000,            # Text chunk size
    "temperature": 0.5,                 # Sampling temperature (0.0-1.0)  
    "extraction_passes": 1,             # Number of extraction attempts
    "max_workers": 10                   # Parallel processing threads
}
```

### Output Format

All extractions return consistent structured data:

```python
{
    "document_id": "doc_123",
    "total_extractions": 5,
    "extractions": [
        {
            "extraction_class": "medication", 
            "extraction_text": "amoxicillin",
            "attributes": {"type": "antibiotic"},
            "start_char": 25,
            "end_char": 35
        }
    ],
    "metadata": {
        "model_id": "gemini-2.5-flash",
        "extraction_passes": 1,
        "temperature": 0.5
    }
}
```

## Use Cases

LangExtract MCP Server supports a wide range of use cases across multiple domains. In healthcare and life sciences, it can extract medications, dosages, and treatment protocols from clinical notes, structure radiology and pathology reports, and process research papers or clinical trial data. For legal and compliance applications, it enables extraction of contract terms, parties, and obligations, as well as analysis of regulatory documents, compliance reports, and case law. In research and academia, the server is useful for extracting methodologies, findings, and citations from papers, analyzing survey responses and interview transcripts, and processing historical or archival materials. For business intelligence, it helps extract insights from customer feedback and reviews, analyze news articles and market reports, and process financial documents and earnings reports.

## Support and Documentation

**Primary Resources:**
- [LangExtract Documentation](https://github.com/google/langextract) - Core library reference
- [FastMCP Documentation](https://gofastmcp.com/) - MCP server framework
- [Model Context Protocol](https://modelcontextprotocol.io/) - Protocol specification