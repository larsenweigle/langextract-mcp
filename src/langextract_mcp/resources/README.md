# LangExtract MCP Server - Client Guide

A Model Context Protocol (MCP) server that provides structured information extraction from unstructured text using Google's LangExtract library and Gemini models.

## Overview

This MCP server enables AI assistants to extract structured information from text documents while maintaining precise source grounding. Each extraction is mapped to its exact location in the source text, enabling visual highlighting and verification.

## Available Tools

### Core Extraction Tools

#### `extract_from_text`
Extract structured information from provided text using Large Language Models.

**Parameters:**
- `text` (string): The text to extract information from
- `prompt_description` (string): Clear instructions for what to extract
- `examples` (array): List of example extractions to guide the model
- `config` (object, optional): Configuration parameters

#### `extract_from_url`
Extract structured information from web content by downloading and processing the text.

**Parameters:**
- `url` (string): URL to download text from (must start with http:// or https://)
- `prompt_description` (string): Clear instructions for what to extract
- `examples` (array): List of example extractions to guide the model
- `config` (object, optional): Configuration parameters

#### `save_extraction_results`
Save extraction results to a JSONL file for later use or visualization.

**Parameters:**
- `extraction_results` (object): Results from extract_from_text or extract_from_url
- `output_name` (string): Name for the output file (without .jsonl extension)
- `output_dir` (string, optional): Directory to save the file (default: current directory)

#### `generate_visualization`
Generate interactive HTML visualization from extraction results.

**Parameters:**
- `jsonl_file_path` (string): Path to the JSONL file containing extraction results
- `output_html_path` (string, optional): Optional path for the HTML output

## How to Structure Examples

Examples are critical for guiding the extraction model. Each example should follow this structure:

```json
{
  "text": "Example input text",
  "extractions": [
    {
      "extraction_class": "category_name",
      "extraction_text": "exact text from input",
      "attributes": {
        "key1": "value1",
        "key2": "value2"
      }
    }
  ]
}
```

### Key Principles for Examples:

1. **Use exact text**: `extraction_text` should be verbatim from the input text
2. **Don't paraphrase**: Extract the actual words, not interpretations
3. **Provide meaningful attributes**: Add context through the attributes dictionary
4. **Cover all extraction classes**: Include examples for each type you want to extract
5. **Show variety**: Demonstrate different patterns and edge cases

## Configuration Options

The `config` parameter accepts these options:

- `model_id` (string): Gemini model to use (default: "gemini-2.5-flash")
- `max_char_buffer` (integer): Text chunk size (default: 1000)
- `temperature` (float): Sampling temperature 0.0-1.0 (default: 0.5)
- `extraction_passes` (integer): Number of extraction attempts for better recall (default: 1)
- `max_workers` (integer): Parallel processing threads (default: 10)

## Supported Models

This server only supports Google Gemini models:
- `gemini-2.5-flash` - **Recommended default** - Optimal balance of speed, cost, and quality
- `gemini-2.5-pro` - Best for complex reasoning and analysis tasks

## Complete Usage Examples

### Example 1: Medical Information Extraction

```json
{
  "tool": "extract_from_text",
  "parameters": {
    "text": "Patient prescribed 500mg amoxicillin twice daily for bacterial infection. Take with food to reduce stomach upset.",
    "prompt_description": "Extract medication information including drug names, dosages, frequencies, and administration instructions. Use exact text for extractions.",
    "examples": [
      {
        "text": "Take 250mg ibuprofen every 4 hours as needed for pain",
        "extractions": [
          {
            "extraction_class": "medication",
            "extraction_text": "ibuprofen",
            "attributes": {
              "type": "pain_reliever",
              "category": "NSAID"
            }
          },
          {
            "extraction_class": "dosage",
            "extraction_text": "250mg",
            "attributes": {
              "amount": "250",
              "unit": "mg"
            }
          },
          {
            "extraction_class": "frequency",
            "extraction_text": "every 4 hours",
            "attributes": {
              "interval": "4 hours",
              "schedule_type": "as_needed"
            }
          }
        ]
      }
    ],
    "config": {
      "model_id": "gemini-2.5-flash",
      "temperature": 0.2
    }
  }
}
```

### Example 2: Document Analysis from URL

```json
{
  "tool": "extract_from_url",
  "parameters": {
    "url": "https://example.com/research-paper.html",
    "prompt_description": "Extract research findings, methodologies, and key statistics from academic papers. Focus on quantitative results and experimental methods.",
    "examples": [
      {
        "text": "Our study of 500 participants showed a 23% improvement in accuracy using the new method compared to baseline.",
        "extractions": [
          {
            "extraction_class": "finding",
            "extraction_text": "23% improvement in accuracy",
            "attributes": {
              "metric": "accuracy",
              "change": "improvement",
              "magnitude": "23%"
            }
          },
          {
            "extraction_class": "methodology",
            "extraction_text": "study of 500 participants",
            "attributes": {
              "sample_size": "500",
              "study_type": "comparative"
            }
          }
        ]
      }
    ],
    "config": {
      "model_id": "gemini-2.5-pro",
      "extraction_passes": 2,
      "max_char_buffer": 1500
    }
  }
}
```

### Example 3: Literary Character Analysis

```json
{
  "tool": "extract_from_text",
  "parameters": {
    "text": "ROMEO: But soft! What light through yonder window breaks? It is the east, and Juliet is the sun.",
    "prompt_description": "Extract characters, emotions, and literary devices from Shakespeare. Capture the emotional context and relationships between characters.",
    "examples": [
      {
        "text": "HAMLET: To be or not to be, that is the question.",
        "extractions": [
          {
            "extraction_class": "character",
            "extraction_text": "HAMLET",
            "attributes": {
              "play": "Hamlet",
              "emotional_state": "contemplative"
            }
          },
          {
            "extraction_class": "philosophical_statement",
            "extraction_text": "To be or not to be, that is the question",
            "attributes": {
              "theme": "existential",
              "type": "soliloquy"
            }
          }
        ]
      }
    ]
  }
}
```

### Example 4: Business Intelligence from Customer Feedback

```json
{
  "tool": "extract_from_text",
  "parameters": {
    "text": "The new software update is fantastic! Loading times are 50% faster and the interface is much more intuitive. However, the mobile app still crashes occasionally.",
    "prompt_description": "Extract customer sentiments, specific feedback points, and performance metrics from reviews. Identify both positive and negative aspects.",
    "examples": [
      {
        "text": "Love the new design but the checkout process takes too long - about 3 minutes.",
        "extractions": [
          {
            "extraction_class": "positive_feedback",
            "extraction_text": "Love the new design",
            "attributes": {
              "aspect": "design",
              "sentiment": "positive"
            }
          },
          {
            "extraction_class": "negative_feedback",
            "extraction_text": "checkout process takes too long",
            "attributes": {
              "aspect": "checkout",
              "sentiment": "negative"
            }
          },
          {
            "extraction_class": "metric",
            "extraction_text": "about 3 minutes",
            "attributes": {
              "measurement": "time",
              "value": "3",
              "unit": "minutes"
            }
          }
        ]
      }
    ]
  }
}
```

## Working with Results

### Saving and Visualizing Extractions

After running an extraction, you can save the results and create an interactive visualization:

```json
{
  "tool": "save_extraction_results",
  "parameters": {
    "extraction_results": {...}, // Results from previous extraction
    "output_name": "medical_extractions",
    "output_dir": "./results"
  }
}
```

```json
{
  "tool": "generate_visualization",
  "parameters": {
    "jsonl_file_path": "./results/medical_extractions.jsonl",
    "output_html_path": "./results/medical_visualization.html"
  }
}
```

### Expected Output Format

All extractions return this structured format:

```json
{
  "document_id": "doc_123",
  "total_extractions": 5,
  "extractions": [
    {
      "extraction_class": "medication",
      "extraction_text": "amoxicillin",
      "attributes": {
        "type": "antibiotic"
      },
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

## Best Practices

### Creating Effective Examples

1. **Quality over quantity**: 1-3 high-quality examples are better than many poor ones
2. **Representative patterns**: Cover the main patterns you expect to see
3. **Exact text matching**: Always use verbatim text from the input
4. **Rich attributes**: Use attributes to provide context and categorization
5. **Edge cases**: Include examples of challenging or ambiguous cases

### Optimizing Performance

- Use `gemini-2.5-flash` for most tasks (faster, cost-effective)
- Use `gemini-2.5-pro` for complex reasoning or analysis
- Increase `extraction_passes` for higher recall on long documents
- Decrease `max_char_buffer` for better accuracy on dense text
- Lower `temperature` (0.1-0.3) for consistent, factual extractions
- Higher `temperature` (0.7-0.9) for creative or interpretive tasks

### Error Handling

Common issues and solutions:

- **"At least one example is required"**: Always provide examples array
- **"Only Gemini models are supported"**: Use `gemini-2.5-flash` or `gemini-2.5-pro`
- **"API key required"**: Server administrator must set LANGEXTRACT_API_KEY
- **"Input text cannot be empty"**: Ensure text parameter has content
- **"URL must start with http://"**: Use full URLs for extract_from_url

## Advanced Features

### Multi-pass Extraction
For comprehensive extraction from long documents:

```json
{
  "config": {
    "extraction_passes": 3,
    "max_workers": 20,
    "max_char_buffer": 800
  }
}
```

### Precision vs. Recall Tuning
- **High precision**: Lower temperature (0.1-0.3), single pass
- **High recall**: Multiple passes (2-3), higher temperature (0.5-0.7)

### Domain-Specific Configurations
- **Medical texts**: Use `gemini-2.5-pro`, low temperature, multiple passes
- **Legal documents**: Smaller chunks (500-800 chars), precise examples
- **Literary analysis**: Higher temperature, rich attribute examples
- **Technical documentation**: Structured examples, consistent terminology

This MCP server provides a powerful interface to Google's LangExtract library, enabling precise structured information extraction with source grounding and interactive visualization capabilities.