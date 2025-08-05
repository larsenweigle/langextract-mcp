#!/usr/bin/env python3
"""
Claude Code installation script for LangExtract MCP Server.

This script can be used with FastMCP CLI for easy Claude Code integration:
    fastmcp install claude-code claude_code_install.py --env LANGEXTRACT_API_KEY=your-key
"""

from src.langextract_mcp.server import main

if __name__ == "__main__":
    main()