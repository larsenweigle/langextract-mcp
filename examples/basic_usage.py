#!/usr/bin/env python3
"""
Basic usage examples for the LangExtract MCP Server.

This example demonstrates how to use the MCP server tools to extract
structured information from text using langextract with proper FastMCP patterns.
"""

import json
from typing import Dict, Any, List


def literature_extraction_example() -> Dict[str, Any]:
    """Example of extracting characters and emotions from literary text."""
    
    return {
        "text": "Lady Juliet gazed longingly at the stars, her heart aching for Romeo. 'Oh Romeo, Romeo, wherefore art thou Romeo?' she whispered into the night air.",
        "prompt_description": "Extract characters, emotions, and relationships in order of appearance. Use exact text for extractions. Do not paraphrase or overlap entities. Provide meaningful attributes for each entity to add context.",
        "examples": [
            {
                "text": "ROMEO. But soft! What light through yonder window breaks? It is the east, and Juliet is the sun.",
                "extractions": [
                    {
                        "extraction_class": "character",
                        "extraction_text": "ROMEO",
                        "attributes": {"emotional_state": "wonder"}
                    },
                    {
                        "extraction_class": "emotion",
                        "extraction_text": "But soft!",
                        "attributes": {"feeling": "gentle awe"}
                    },
                    {
                        "extraction_class": "relationship",
                        "extraction_text": "Juliet is the sun",
                        "attributes": {"type": "metaphor", "relationship": "romantic"}
                    }
                ]
            }
        ],
        "config": {
            "model_id": "gemini-2.5-flash",
            "temperature": 0.5,
            "extraction_passes": 1
            # Note: No API key needed - handled securely server-side!
        }
    }


def medical_extraction_example() -> Dict[str, Any]:
    """Example of extracting medical information from clinical text."""
    
    return {
        "text": "Patient was prescribed 500mg amoxicillin to be taken orally three times daily for 7 days to treat the bacterial infection. Patient should take with food to reduce stomach upset.",
        "prompt_description": "Extract medication information including drug names, dosages, routes of administration, frequencies, and duration. Also extract any special instructions.",
        "examples": [
            {
                "text": "Take 250mg ibuprofen by mouth every 4 hours as needed for pain, not to exceed 1200mg in 24 hours.",
                "extractions": [
                    {
                        "extraction_class": "medication",
                        "extraction_text": "ibuprofen",
                        "attributes": {"type": "NSAID", "indication": "pain"}
                    },
                    {
                        "extraction_class": "dosage",
                        "extraction_text": "250mg",
                        "attributes": {"amount": "250", "unit": "mg"}
                    },
                    {
                        "extraction_class": "route",
                        "extraction_text": "by mouth",
                        "attributes": {"method": "oral"}
                    },
                    {
                        "extraction_class": "frequency",
                        "extraction_text": "every 4 hours",
                        "attributes": {"interval": "4 hours", "as_needed": True}
                    },
                    {
                        "extraction_class": "max_dose",
                        "extraction_text": "not to exceed 1200mg in 24 hours",
                        "attributes": {"max_amount": "1200", "time_period": "24 hours"}
                    }
                ]
            }
        ],
        "config": {
            "model_id": "gemini-2.5-flash",
            "temperature": 0.3,  # Lower temperature for medical accuracy
            "extraction_passes": 2  # Multiple passes for better recall
            # Note: No API key needed - handled securely server-side!
        }
    }


def url_extraction_example() -> Dict[str, Any]:
    """Example of extracting information from a URL."""
    
    return {
        "url": "https://www.gutenberg.org/files/1513/1513-0.txt",  # Romeo and Juliet
        "prompt_description": "Extract character names and their emotional states or actions from this literary text.",
        "examples": [
            {
                "text": "JULIET appears above at a window. But, soft! what light through yonder window breaks?",
                "extractions": [
                    {
                        "extraction_class": "character",
                        "extraction_text": "JULIET",
                        "attributes": {"location": "window", "action": "appears"}
                    },
                    {
                        "extraction_class": "emotion",
                        "extraction_text": "But, soft!",
                        "attributes": {"feeling": "gentle exclamation", "intensity": "mild"}
                    }
                ]
            }
        ],
        "config": {
            "model_id": "gemini-2.5-flash",
            "max_char_buffer": 800,  # Smaller chunks for better accuracy
            "extraction_passes": 3,  # Multiple passes for long documents
            "max_workers": 15  # More workers for faster processing
            # Note: No API key needed - handled securely server-side!
        }
    }


def print_tool_call(tool_name: str, parameters: Dict[str, Any]) -> None:
    """Pretty print a tool call for MCP demonstration."""
    print(f"\n{'='*80}")
    print(f"MCP Tool Call: {tool_name}")
    print('='*80)
    print("Parameters:")
    print(json.dumps(parameters, indent=2))
    print()


def main():
    """Main function to demonstrate FastMCP tool patterns with Claude Code integration."""
    print("LangExtract MCP Server - Claude Code Integration Examples")
    print("=" * 80)
    print("These examples show the proper format for calling MCP tools")
    print("with the optimized FastMCP server designed for Claude Code.\n")
    
    # Literature extraction tool call
    lit_params = literature_extraction_example()
    print_tool_call("extract_from_text", lit_params)
    
    # Medical extraction tool call  
    med_params = medical_extraction_example()
    print_tool_call("extract_from_text", med_params)
    
    # URL extraction tool call
    url_params = url_extraction_example()
    print_tool_call("extract_from_url", url_params)
    
    # Save results tool call
    save_params = {
        "extraction_results": {
            "document_id": "example_doc",
            "total_extractions": 5,
            "extractions": [],
            "metadata": {"model_id": "gemini-2.5-flash"}
        },
        "output_name": "literature_results",
        "output_dir": "./results"
    }
    print_tool_call("save_extraction_results", save_params)
    
    # Visualization tool call
    viz_params = {
        "jsonl_file_path": "./results/literature_results.jsonl",
        "output_html_path": "./results/literature_visualization.html"
    }
    print_tool_call("generate_visualization", viz_params)
    
    # Info tools
    print_tool_call("list_supported_models", {})
    print_tool_call("get_server_info", {})
    
    print(f"\n{'='*80}")
    print("FastMCP Server Features:")
    print("="*80)
    print("✅ Proper @mcp.tool decorators")
    print("✅ Pydantic models for type safety")
    print("✅ ToolError for proper error handling")
    print("✅ Clean parameter validation")
    print("✅ Structured return values")
    print("✅ Comprehensive documentation")
    print("✅ Production-ready patterns")
    print("✅ Secure API key handling (server-side only)")
    print("✅ Persistent connections and intelligent caching")
    print("✅ Claude Code integration optimizations")
    print("\nTo use with Claude Code:")
    print("1. [Setup] claude mcp add langextract-mcp -e LANGEXTRACT_API_KEY=your-key -- uv run --with fastmcp fastmcp run src/langextract_mcp/server.py")
    print("2. [Claude Code] Server automatically starts and connects")
    print("3. [Claude Code] Use the tool calls shown above - no API keys needed!")
    print("4. [Performance] Enjoy 3x faster extraction with persistent caching")
    print("\n🔒 Security Note: API keys never leave the server!")
    print("⚡ Performance Note: Optimized for Claude Code's long-running context!")


if __name__ == "__main__":
    main()