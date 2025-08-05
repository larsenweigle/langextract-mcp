# LangExtract Library Study Notes

## Overview
LangExtract is a Python library developed by Google that uses Large Language Models (LLMs) to extract structured information from unstructured text documents based on user-defined instructions. It's designed to process materials like clinical notes, reports, and other documents while maintaining precise source grounding.

## Key Features & Differentiators

### 1. Precise Source Grounding
- **Capability**: Maps every extraction to its exact location in the source text
- **Benefit**: Enables visual highlighting for easy traceability and verification
- **Implementation**: Through annotation system that tracks character positions

### 2. Reliable Structured Outputs
- **Schema Enforcement**: Consistent output schema based on few-shot examples
- **Controlled Generation**: Leverages structured output capabilities in supported models (Gemini)
- **Format Support**: JSON and YAML output formats

### 3. Long Document Optimization
- **Challenge Addressed**: "Needle-in-a-haystack" problem in large documents
- **Strategy**: Text chunking + parallel processing + multiple extraction passes
- **Benefit**: Higher recall on complex documents

### 4. Interactive Visualization
- **Output**: Self-contained HTML files for reviewing extractions
- **Scalability**: Handles thousands of extracted entities
- **Context**: Shows entities in their original document context

### 5. Flexible LLM Support
- **Cloud Models**: Google Gemini family, OpenAI models
- **Local Models**: Built-in Ollama interface
- **Extensibility**: Can be extended to other APIs

### 6. Domain Adaptability
- **No Fine-tuning**: Uses few-shot examples instead of model training
- **Flexibility**: Works across any domain with proper examples
- **Customization**: Leverages LLM world knowledge through prompt engineering

## Core Architecture

### Main Components

#### 1. Data Models (`data.py`)
- **ExampleData**: Defines extraction examples with text and expected extractions
- **Extraction**: Individual extracted entity with class, text, and attributes
- **Document**: Input document container
- **AnnotatedDocument**: Result container with extractions and metadata

#### 2. Inference Engine (`inference.py`)
- **GeminiLanguageModel**: Google Gemini API integration
- **OpenAILanguageModel**: OpenAI API integration
- **BaseLanguageModel**: Abstract base for language model implementations
- **Schema Support**: Structured output generation for supported models

#### 3. Annotation System (`annotation.py`)
- **Annotator**: Core extraction orchestrator
- **Text Processing**: Handles chunking and parallel processing
- **Progress Tracking**: Monitors extraction progress

#### 4. Resolver System (`resolver.py`)
- **Purpose**: Parses raw LLM output into structured Extraction objects
- **Fence Handling**: Extracts content from markdown code blocks
- **Format Parsing**: Handles JSON/YAML parsing and validation

#### 5. Chunking Engine (`chunking.py`)
- **Text Segmentation**: Breaks long documents into processable chunks
- **Buffer Management**: Handles max_char_buffer limits
- **Overlap Strategy**: Maintains context across chunk boundaries

#### 6. Visualization (`visualization.py`)
- **HTML Generation**: Creates interactive visualization files
- **Entity Highlighting**: Shows extractions in original context
- **Scalable Interface**: Handles large result sets efficiently

#### 7. I/O Operations (`io.py`)
- **URL Download**: Fetches text from web URLs
- **File Operations**: Saves results to JSONL format
- **Document Loading**: Handles various input formats

### Key API Functions

#### Primary Interface
```python
lx.extract(
    text_or_documents,      # Input text, URL, or Document objects
    prompt_description,     # Extraction instructions
    examples,              # Few-shot examples
    model_id="gemini-2.5-flash",
    # Configuration options...
)
```

#### Visualization
```python
lx.visualize(jsonl_file_path)  # Generate HTML visualization
```

#### I/O Operations
```python
lx.io.save_annotated_documents(results, output_name, output_dir)
```

## Configuration Parameters

### Core Parameters
- **model_id**: LLM model selection
- **api_key**: Authentication for cloud models
- **temperature**: Sampling temperature (0.5 recommended)
- **max_char_buffer**: Chunk size limit (1000 default)

### Performance Parameters
- **max_workers**: Parallel processing workers (10 default)
- **batch_length**: Chunks per batch (10 default)
- **extraction_passes**: Multiple extraction attempts (1 default)

### Output Control
- **format_type**: JSON or YAML output
- **fence_output**: Code fence expectations
- **use_schema_constraints**: Structured output enforcement

## Supported Models

### Google Gemini
- **gemini-2.5-flash**: Recommended default (speed/cost/quality balance)
- **gemini-2.5-pro**: For complex reasoning tasks
- **Schema Support**: Full structured output support
- **Rate Limits**: Tier 2 quota recommended for production

### OpenAI
- **gpt-4o**: Supported with limitations
- **Requirements**: `fence_output=True`, `use_schema_constraints=False`
- **Note**: Schema constraints not yet implemented for OpenAI

### Local Models
- **Ollama**: Built-in support
- **Extension**: Can be extended to other local APIs

## Use Cases & Examples

### 1. Literary Analysis
- **Characters**: Extract character names and emotional states
- **Relationships**: Identify character interactions and metaphors
- **Context**: Track narrative elements across long texts

### 2. Medical Document Processing
- **Medications**: Extract drug names, dosages, routes, frequencies
- **Clinical Notes**: Structure unstructured medical reports
- **Compliance**: Maintain source grounding for medical accuracy

### 3. Radiology Reports
- **Structured Data**: Convert free-text reports to structured findings
- **Demo Available**: RadExtract on HuggingFace Spaces

### 4. Long Document Processing
- **Full Novels**: Process complete books (e.g., Romeo & Juliet - 147k chars)
- **Performance**: Parallel processing with multiple passes
- **Visualization**: Handle hundreds of entities in context

## Technical Implementation Details

### Text Processing Pipeline
1. **Input Validation**: Validate text/documents and examples
2. **URL Handling**: Download content if URL provided
3. **Chunking**: Break long texts into manageable pieces
4. **Parallel Processing**: Distribute chunks across workers
5. **Multiple Passes**: Optional additional extraction rounds
6. **Resolution**: Parse LLM outputs into structured data
7. **Annotation**: Create AnnotatedDocument with source grounding
8. **Visualization**: Generate interactive HTML output

### Error Handling
- **API Failures**: Graceful handling of LLM API errors
- **Parsing Errors**: Robust JSON/YAML parsing with fallbacks
- **Validation**: Schema validation for structured outputs

### Performance Optimization
- **Concurrent Processing**: Parallel chunk processing
- **Efficient Chunking**: Smart text segmentation
- **Progressive Enhancement**: Multiple passes for better recall
- **Memory Management**: Efficient handling of large documents

## MCP Server Design Implications

Based on langextract's architecture, a FastMCP server should expose:

### Core Tools
1. **extract_text**: Main extraction function
2. **extract_from_url**: URL-based extraction
3. **visualize_results**: Generate HTML visualization
4. **validate_examples**: Validate extraction examples

### Configuration Management
1. **set_model**: Configure LLM model
2. **set_api_key**: Set authentication
3. **configure_extraction**: Set extraction parameters

### File Operations
1. **save_results**: Save to JSONL format
2. **load_results**: Load previous results
3. **export_visualization**: Generate and save HTML

### Advanced Features
1. **batch_extract**: Process multiple documents
2. **progressive_extract**: Multi-pass extraction
3. **compare_results**: Compare extraction results

### Resource Management
- **Model Configurations**: Manage different model setups
- **Example Templates**: Store reusable extraction examples
- **Result Archives**: Access previous extraction results

## Dependencies & Installation
- **Core**: Python 3.10+, requests, dotenv
- **LLM APIs**: google-generativeai, openai
- **Processing**: concurrent.futures for parallelization
- **Visualization**: HTML/CSS/JS generation
- **Format Support**: JSON, YAML parsing

## Licensing & Usage
- **License**: Apache 2.0
- **Disclaimer**: Not officially supported Google product
- **Health Applications**: Subject to Health AI Developer Foundations Terms
- **Citation**: Recommended for production/publication use