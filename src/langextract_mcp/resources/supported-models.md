# Supported Language Models

This document provides comprehensive information about the language models supported by the langextract-mcp server.

## Currently Supported Models

The langextract-mcp server currently supports **Google Gemini models only**, which are optimized for reliable structured extraction with schema constraints.

### Gemini 2.5 Flash
- **Provider**: Google
- **Model ID**: `gemini-2.5-flash`
- **Description**: Fast, cost-effective model with excellent quality
- **Schema Constraints**: ✅ Supported
- **Recommended For**:
  - General extraction tasks
  - Fast processing requirements
  - Cost-sensitive applications
- **Notes**: Recommended default choice - optimal balance of speed, cost, and quality

### Gemini 2.5 Pro
- **Provider**: Google
- **Model ID**: `gemini-2.5-pro`
- **Description**: Advanced model for complex reasoning tasks
- **Schema Constraints**: ✅ Supported
- **Recommended For**:
  - Complex extractions
  - High accuracy requirements
  - Sophisticated reasoning tasks
- **Notes**: Best quality for complex tasks but higher cost

## Model Recommendations

| Use Case | Recommended Model | Reason |
|----------|------------------|---------|
| **Default/General** | `gemini-2.5-flash` | Best balance of speed, cost, and quality |
| **High Quality** | `gemini-2.5-pro` | Superior accuracy and reasoning capabilities |
| **Cost Optimized** | `gemini-2.5-flash` | Most cost-effective option |
| **Complex Reasoning** | `gemini-2.5-pro` | Advanced reasoning for complex extraction tasks |

## Configuration Parameters

When using any supported model, you can configure the following parameters:

- **`model_id`**: The model identifier (e.g., "gemini-2.5-flash")
- **`max_char_buffer`**: Maximum characters per chunk (default: 1000)
- **`temperature`**: Sampling temperature 0.0-1.0 (default: 0.5)
- **`extraction_passes`**: Number of extraction passes for better recall (default: 1)
- **`max_workers`**: Maximum parallel workers (default: 10)

## Limitations

- **Provider Support**: Currently supports Google Gemini models only
- **Future Support**: OpenAI and local model support may be added in future versions
- **API Dependencies**: Requires active internet connection and valid API keys

## Schema Constraints

All supported Gemini models include schema constraint capabilities, which means:

- **Structured Output**: Guaranteed JSON structure based on your examples
- **Type Safety**: Consistent field types across extractions
- **Validation**: Automatic validation of extracted data against schema
- **Reliability**: Reduced hallucination and improved consistency

This makes the langextract-mcp server particularly reliable for production applications requiring consistent structured data extraction.
