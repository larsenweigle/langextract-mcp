"""FastMCP server for langextract - optimized for Claude Code integration."""

import os
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import hashlib
import json

import langextract as lx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field


# Pydantic models for structured inputs
class ExtractionExample(BaseModel):
    """Model for extraction examples."""
    text: str = Field(..., description="Example text")
    extractions: List[Dict[str, Any]] = Field(..., description="Expected extractions")


class ExtractionConfig(BaseModel):
    """Configuration for extraction parameters."""
    model_id: str = Field(default="gemini-2.5-flash", description="LLM model to use")
    max_char_buffer: int = Field(default=1000, description="Max characters per chunk")
    temperature: float = Field(default=0.5, description="Sampling temperature (0.0-1.0)")
    extraction_passes: int = Field(default=1, description="Number of extraction passes for better recall")
    max_workers: int = Field(default=10, description="Max parallel workers")


# Initialize FastMCP server with Claude Code compatibility
mcp = FastMCP(
    name="LangExtract",
    instructions="Extract structured information from unstructured text using Large Language Models. "
                "Provides precise source grounding, interactive visualizations, and support for multiple LLM providers.",
)


class LangExtractClient:
    """Optimized langextract client for MCP server usage.
    
    This client maintains persistent connections and caches expensive operations
    like schema generation and prompt templates for better performance in a
    long-running MCP server context.
    """
    
    def __init__(self):
        self._language_models: Dict[str, Any] = {}
        self._schema_cache: Dict[str, Any] = {}
        self._prompt_template_cache: Dict[str, Any] = {}
        self._resolver_cache: Dict[str, Any] = {}
        
    def _get_examples_hash(self, examples: List[ExtractionExample]) -> str:
        """Generate a hash for caching based on examples."""
        examples_str = json.dumps([
            {
                "text": ex.text, 
                "extractions": ex.extractions
            } for ex in examples
        ], sort_keys=True)
        return hashlib.md5(examples_str.encode()).hexdigest()
    
    def _get_language_model(self, config: ExtractionConfig, api_key: str, schema: Optional[Any] = None, schema_hash: Optional[str] = None) -> Any:
        """Get or create a cached language model instance."""
        # Include schema hash in cache key to prevent schema mutation conflicts
        model_key = f"{config.model_id}_{config.temperature}_{config.max_workers}_{schema_hash or 'no_schema'}"
        
        if model_key not in self._language_models:
            # Validate that only Gemini models are supported
            if not config.model_id.startswith('gemini'):
                raise ValueError(f"Only Gemini models are supported. Got: {config.model_id}")
                
            language_model = lx.inference.GeminiLanguageModel(
                model_id=config.model_id,
                api_key=api_key,
                temperature=config.temperature,
                max_workers=config.max_workers,
                gemini_schema=schema
            )
            self._language_models[model_key] = language_model
            
        return self._language_models[model_key]
    
    def _get_schema(self, examples: List[ExtractionExample], model_id: str) -> Tuple[Any, str]:
        """Get or create a cached schema for the examples.
        
        Returns:
            Tuple of (schema, examples_hash) for use in caching language models
        """
        if not model_id.startswith('gemini'):
            return None, ""
            
        examples_hash = self._get_examples_hash(examples)
        schema_key = f"{model_id}_{examples_hash}"
        
        if schema_key not in self._schema_cache:
            # Convert examples to langextract format
            langextract_examples = self._create_langextract_examples(examples)
            
            # Create prompt template to generate schema
            prompt_template = lx.prompting.PromptTemplateStructured()
            prompt_template.examples.extend(langextract_examples)
            
            # Generate schema
            schema = lx.schema.GeminiSchema.from_examples(prompt_template.examples)
            self._schema_cache[schema_key] = schema
            
        return self._schema_cache[schema_key], examples_hash
    
    def _get_resolver(self, format_type: str = "JSON") -> Any:
        """Get or create a cached resolver."""
        if format_type not in self._resolver_cache:
            resolver = lx.resolver.Resolver(
                fence_output=False,
                format_type=lx.data.FormatType.JSON if format_type == "JSON" else lx.data.FormatType.YAML,
                extraction_attributes_suffix="_attributes",
                extraction_index_suffix=None,
            )
            self._resolver_cache[format_type] = resolver
            
        return self._resolver_cache[format_type]
    
    def _create_langextract_examples(self, examples: List[ExtractionExample]) -> List[lx.data.ExampleData]:
        """Convert Pydantic examples to langextract ExampleData objects."""
        langextract_examples = []
        
        for example in examples:
            extractions = []
            for extraction_data in example.extractions:
                extractions.append(
                    lx.data.Extraction(
                        extraction_class=extraction_data["extraction_class"],
                        extraction_text=extraction_data["extraction_text"],
                        attributes=extraction_data.get("attributes", {})
                    )
                )
            
            langextract_examples.append(
                lx.data.ExampleData(
                    text=example.text,
                    extractions=extractions
                )
            )
        
        return langextract_examples
    
    def extract(
        self, 
        text_or_url: str,
        prompt_description: str,
        examples: List[ExtractionExample],
        config: ExtractionConfig,
        api_key: str
    ) -> lx.data.AnnotatedDocument:
        """Optimized extraction using cached components."""
        # Get or generate schema first
        schema, examples_hash = self._get_schema(examples, config.model_id)
        
        # Get cached components with schema-aware caching
        language_model = self._get_language_model(config, api_key, schema, examples_hash)
        resolver = self._get_resolver("JSON")
        
        # Convert examples
        langextract_examples = self._create_langextract_examples(examples)
        
        # Create prompt template
        prompt_template = lx.prompting.PromptTemplateStructured(
            description=prompt_description
        )
        prompt_template.examples.extend(langextract_examples)
        
        # Create annotator
        annotator = lx.annotation.Annotator(
            language_model=language_model,
            prompt_template=prompt_template,
            format_type=lx.data.FormatType.JSON,
            fence_output=False,
        )
        
        # Perform extraction
        if text_or_url.startswith(('http://', 'https://')):
            # Download text first
            text = lx.io.download_text_from_url(text_or_url)
        else:
            text = text_or_url
            
        return annotator.annotate_text(
            text=text,
            resolver=resolver,
            max_char_buffer=config.max_char_buffer,
            batch_length=10,
            additional_context=None,
            debug=False,  # Disable debug for cleaner MCP output
            extraction_passes=config.extraction_passes,
        )


# Global client instance for the server lifecycle
_langextract_client = LangExtractClient()


def _get_api_key() -> Optional[str]:
    """Get API key from environment (server-side only for security)."""
    return os.environ.get("LANGEXTRACT_API_KEY")


def _format_extraction_result(result: lx.data.AnnotatedDocument, config: ExtractionConfig, source_url: Optional[str] = None) -> Dict[str, Any]:
    """Format langextract result for MCP response."""
    extractions = []
    
    for extraction in result.extractions or []:
        extractions.append({
            "extraction_class": extraction.extraction_class,
            "extraction_text": extraction.extraction_text,
            "attributes": extraction.attributes,
            "start_char": getattr(extraction, 'start_char', None),
            "end_char": getattr(extraction, 'end_char', None),
        })
    
    response = {
        "document_id": result.document_id if result.document_id else "anonymous",
        "total_extractions": len(extractions),
        "extractions": extractions,
        "metadata": {
            "model_id": config.model_id,
            "extraction_passes": config.extraction_passes,
            "max_char_buffer": config.max_char_buffer,
            "temperature": config.temperature,
        }
    }
    
    if source_url:
        response["source_url"] = source_url
        
    return response


@mcp.tool
def extract_from_text(
    text: str,
    prompt_description: str,
    examples: List[ExtractionExample],
    config: ExtractionConfig = ExtractionConfig()
) -> Dict[str, Any]:
    """
    Extract structured information from text using langextract.
    
    Uses Large Language Models to extract structured information from unstructured text
    based on user-defined instructions and examples. Each extraction is mapped to its
    exact location in the source text for precise source grounding.
    
    Args:
        text: The text to extract information from
        prompt_description: Clear instructions for what to extract
        examples: List of example extractions to guide the model
        config: Configuration parameters for the extraction
        
    Returns:
        Dictionary containing extracted entities with source locations and metadata
        
    Raises:
        ToolError: If extraction fails due to invalid parameters or API issues
    """
    try:
        if not examples:
            raise ToolError("At least one example is required for reliable extraction")
        
        if not prompt_description.strip():
            raise ToolError("Prompt description cannot be empty")
            
        if not text.strip():
            raise ToolError("Input text cannot be empty")
        
        # Validate that only Gemini models are supported
        if not config.model_id.startswith('gemini'):
            raise ToolError(
                f"Only Google Gemini models are supported. Got: {config.model_id}. "
                f"Use 'list_supported_models' tool to see available options."
            )
        
        # Get API key (server-side only for security)
        api_key = _get_api_key()
        if not api_key:
            raise ToolError(
                "API key required. Server administrator must set LANGEXTRACT_API_KEY environment variable."
            )
        
        # Perform optimized extraction using cached client
        result = _langextract_client.extract(
            text_or_url=text,
            prompt_description=prompt_description,
            examples=examples,
            config=config,
            api_key=api_key
        )
        
        return _format_extraction_result(result, config)
        
    except ValueError as e:
        raise ToolError(f"Invalid parameters: {str(e)}")
    except Exception as e:
        raise ToolError(f"Extraction failed: {str(e)}")


@mcp.tool
def extract_from_url(
    url: str,
    prompt_description: str,
    examples: List[ExtractionExample],
    config: ExtractionConfig = ExtractionConfig()
) -> Dict[str, Any]:
    """
    Extract structured information from text content at a URL.
    
    Downloads text from the specified URL and extracts structured information
    using Large Language Models. Ideal for processing web articles, documents,
    or any text content accessible via HTTP/HTTPS.
    
    Args:
        url: URL to download text from (must start with http:// or https://)
        prompt_description: Clear instructions for what to extract
        examples: List of example extractions to guide the model
        config: Configuration parameters for the extraction
        
    Returns:
        Dictionary containing extracted entities with source locations and metadata
        
    Raises:
        ToolError: If URL is invalid, download fails, or extraction fails
    """
    try:
        if not url.startswith(('http://', 'https://')):
            raise ToolError("URL must start with http:// or https://")
            
        if not examples:
            raise ToolError("At least one example is required for reliable extraction")
        
        if not prompt_description.strip():
            raise ToolError("Prompt description cannot be empty")
        
        # Validate that only Gemini models are supported
        if not config.model_id.startswith('gemini'):
            raise ToolError(
                f"Only Google Gemini models are supported. Got: {config.model_id}. "
                f"Use 'list_supported_models' tool to see available options."
            )
        
        # Get API key (server-side only for security)
        api_key = _get_api_key()
        if not api_key:
            raise ToolError(
                "API key required. Server administrator must set LANGEXTRACT_API_KEY environment variable."
            )
        
        # Perform optimized extraction using cached client
        result = _langextract_client.extract(
            text_or_url=url,
            prompt_description=prompt_description,
            examples=examples,
            config=config,
            api_key=api_key
        )
        
        return _format_extraction_result(result, config, source_url=url)
        
    except ValueError as e:
        raise ToolError(f"Invalid parameters: {str(e)}")
    except Exception as e:
        raise ToolError(f"URL extraction failed: {str(e)}")


@mcp.tool  
def save_extraction_results(
    extraction_results: Dict[str, Any],
    output_name: str,
    output_dir: str = "."
) -> Dict[str, str]:
    """
    Save extraction results to a JSONL file for later use or visualization.
    
    Saves the extraction results in JSONL (JSON Lines) format, which is commonly
    used for structured data and can be loaded for visualization or further processing.
    
    Args:
        extraction_results: Results from extract_from_text or extract_from_url
        output_name: Name for the output file (without .jsonl extension)
        output_dir: Directory to save the file (default: current directory)
        
    Returns:
        Dictionary with file path and save confirmation
        
    Raises:
        ToolError: If save operation fails
    """
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create full file path
        file_path = output_path / f"{output_name}.jsonl"
        
        # Save results to JSONL format
        import json
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(extraction_results, f, ensure_ascii=False)
            f.write('\n')
        
        return {
            "message": "Results saved successfully",
            "file_path": str(file_path.absolute()),
            "total_extractions": extraction_results.get("total_extractions", 0)
        }
        
    except Exception as e:
        raise ToolError(f"Failed to save results: {str(e)}")


@mcp.tool
def generate_visualization(
    jsonl_file_path: str,
    output_html_path: Optional[str] = None
) -> Dict[str, str]:
    """
    Generate interactive HTML visualization from extraction results.
    
    Creates an interactive HTML file that shows extracted entities highlighted
    in their original text context. The visualization is self-contained and
    can handle thousands of entities with color coding and hover details.
    
    Args:
        jsonl_file_path: Path to the JSONL file containing extraction results
        output_html_path: Optional path for the HTML output (default: auto-generated)
        
    Returns:
        Dictionary with HTML file path and generation details
        
    Raises:
        ToolError: If visualization generation fails
    """
    try:
        # Validate input file exists
        input_path = Path(jsonl_file_path)
        if not input_path.exists():
            raise ToolError(f"Input file not found: {jsonl_file_path}")
        
        # Generate visualization using langextract
        html_content = lx.visualize(str(input_path))
        
        # Determine output path
        if output_html_path:
            output_path = Path(output_html_path)
        else:
            output_path = input_path.parent / f"{input_path.stem}_visualization.html"
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return {
            "message": "Visualization generated successfully",
            "html_file_path": str(output_path.absolute()),
            "file_size_bytes": len(html_content.encode('utf-8'))
        }
        
    except Exception as e:
        raise ToolError(f"Failed to generate visualization: {str(e)}")


@mcp.tool
def list_supported_models() -> Dict[str, Any]:
    """
    List all supported language models and their characteristics.
    
    This server currently supports Google Gemini models only, optimized for
    reliable structured extraction with schema constraints.
    
    Returns:
        Dictionary containing model information and recommendations
    """
    return {
        "models": {
            "gemini-2.5-flash": {
                "provider": "Google",
                "description": "Fast, cost-effective model with excellent quality",
                "supports_schema_constraints": True,
                "recommended_for": ["General extraction", "Fast processing", "Cost-sensitive applications"],
                "notes": "Recommended default choice - optimal balance of speed, cost, and quality"
            },
            "gemini-2.5-pro": {
                "provider": "Google",
                "description": "Advanced model for complex reasoning tasks",
                "supports_schema_constraints": True,
                "recommended_for": ["Complex extractions", "High accuracy requirements", "Sophisticated reasoning"],
                "notes": "Best quality for complex tasks but higher cost"
            }
        },
        "supported_providers": {
            "google": {
                "api_key_required": True,
                "environment_variable": "LANGEXTRACT_API_KEY",
                "get_api_key": "https://aistudio.google.com/app/apikey"
            }
        },
        "optimization": {
            "note": "This MCP server uses persistent connections, schema caching, and connection pooling for optimal performance"
        },
        "recommendations": {
            "default": "gemini-2.5-flash",
            "high_quality": "gemini-2.5-pro", 
            "cost_optimized": "gemini-2.5-flash",
            "complex_reasoning": "gemini-2.5-pro"
        },
        "limitations": {
            "note": "Currently supports Google Gemini models only. OpenAI and local model support may be added in future versions."
        }
    }


@mcp.tool
def get_server_info() -> Dict[str, Any]:
    """
    Get information about the LangExtract MCP server.
    
    Returns server version, capabilities, and configuration information.
    
    Returns:
        Dictionary containing server information and capabilities
    """
    return {
        "server_name": "LangExtract MCP Server",
        "version": "0.1.0",
        "description": "Extract structured information from unstructured text using Large Language Models",
        "optimizations": [
            "Persistent language model connections",
            "Schema caching for repeated examples",
            "Connection pooling for better performance",
            "Template caching for efficiency"
        ],
        "capabilities": [
            "Text extraction with source grounding",
            "URL-based content extraction", 
            "Interactive HTML visualizations",
            "Multiple LLM provider support",
            "Parallel processing for long documents",
            "Multiple extraction passes for better recall"
        ],
        "supported_formats": {
            "input": ["Plain text", "URLs (HTTP/HTTPS)"],
            "output": ["JSON", "JSONL", "Interactive HTML"]
        },
        "claude_code_ready": True,
        "environment": {
            "api_key_configured": bool(os.environ.get("LANGEXTRACT_API_KEY")),
            "python_version": "3.10+",
            "dependencies": ["langextract", "fastmcp", "pydantic"],
            "transport": "STDIO (Claude Code compatible)"
        }
    }


# Claude Code compatible entry point
def main():
    """Main entry point for the FastMCP server - Claude Code compatible."""
    mcp.run()


if __name__ == "__main__":
    main()