# FastMCP Framework Study Notes - Deep Analysis

## Overview
FastMCP is a Python framework for building Model Context Protocol (MCP) servers and clients, designed to enable sophisticated interactions between AI systems and various services. It provides a "fast, Pythonic way" to build MCP servers with comprehensive functionality and enterprise-grade features.

## Core Architecture

### 1. Servers
- **Primary Function**: Expose tools as executable capabilities
- **Authentication**: Support multiple authentication mechanisms
- **Middleware**: Enable cross-cutting functionality for request/response processing
- **Resource Management**: Allow resource and prompt management
- **Monitoring**: Support progress reporting and logging

### 2. Clients
- **Purpose**: Provide programmatic interaction with MCP servers
- **Authentication**: Support multiple methods (Bearer Token, OAuth)
- **Processing**: Handle message processing, logging, and progress monitoring

## Key Features

### Tool Operations
- Define tools as executable functions
- Structured user input handling
- Comprehensive tool management

### Resource Management
- Create and manage resources
- Prompt templating capabilities
- Resource organization and access

### Authentication & Security
- Flexible authentication strategies
- Bearer Token support
- OAuth integration
- Authorization provider compatibility

### Middleware System
- Request/response processing
- Cross-cutting concerns handling
- Extensible middleware chain

### Monitoring & Logging
- Progress tracking
- Comprehensive logging
- User interaction context

## Integration Capabilities

### Supported Platforms
- OpenAI API
- Anthropic
- Google Gemini
- FastAPI
- Starlette/ASGI

### Authorization Providers
- Various authorization providers supported
- Flexible configuration options

## Server Development Guidelines

### 1. Tool Definition
- Define tools as executable functions
- Implement clear input/output schemas
- Handle errors gracefully

### 2. Authentication Setup
- Choose appropriate authentication strategy
- Configure security mechanisms
- Implement user context handling

### 3. Context Configuration
- Set up logging context
- Configure user interactions
- Implement progress tracking

### 4. Middleware Implementation
- Use middleware for common functionality
- Process requests and responses
- Handle cross-cutting concerns

### 5. Resource Creation
- Define resources and prompt templates
- Organize resource access patterns
- Implement resource management

## Unique Selling Points

1. **Pythonic Interface**: Natural Python API design
2. **Flexible Composition**: Modular server composition
3. **Structured Input**: Sophisticated user input handling
4. **Comprehensive SDK**: Extensive documentation and tooling
5. **Standardized Protocol**: Uses MCP for consistent interactions

## FastMCP Implementation Patterns

### 1. Server Instantiation
```python
from fastmcp import FastMCP

# Basic server
mcp = FastMCP("Demo 🚀")

# Server with configuration
mcp = FastMCP(
    name="LangExtractServer",
    instructions="Extract structured information from text using LLMs",
    include_tags={"public"},
    exclude_tags={"internal"}
)
```

### 2. Tool Definition Patterns
```python
@mcp.tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

# Complex parameters with validation
@mcp.tool
def process_data(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance",
    category: str | None = None
) -> dict:
    """Process data with parameters"""
    return {"results": []}
```

### 3. Error Handling
```python
from fastmcp.exceptions import ToolError

@mcp.tool
def divide(a: float, b: float) -> float:
    if b == 0:
        raise ToolError("Cannot divide by zero")
    return a / b
```

### 4. Authentication Patterns
```python
from fastmcp.server.auth.providers.jwt import JWTVerifier

auth = JWTVerifier(
    jwks_uri="https://your-auth-system.com/.well-known/jwks.json",
    issuer="https://your-auth-system.com",
    audience="your-mcp-server"
)

mcp = FastMCP(name="Protected Server", auth=auth)
```

### 5. Server Execution
```python
# STDIO transport (default for MCP clients)
mcp.run()

# HTTP transport
mcp.run(transport="http", host="0.0.0.0", port=9000)
```

### 6. Server Composition
```python
main = FastMCP(name="MainServer")
sub = FastMCP(name="SubServer")
main.mount(sub, prefix="sub")
```

## Key Insights for LangExtract MCP Server

1. **Simple Decorator Pattern**: Use `@mcp.tool` for all langextract functions
2. **Type Safety**: Leverage Python type hints for automatic validation
3. **Proper Error Handling**: Use `ToolError` for controlled error messaging
4. **Clean Architecture**: Keep tools simple and focused
5. **Context Management**: Use FastMCP's built-in context for logging/progress
6. **Transport Flexibility**: Support both STDIO and HTTP transports
7. **Authentication Ready**: Design with auth in mind for production use

## Implementation Strategy for langextract

Based on deeper FastMCP understanding:

1. **Clean Tool Interface**: Each langextract function as a simple `@mcp.tool`
2. **Type-Safe Parameters**: Use Pydantic models for complex inputs
3. **Structured Outputs**: Return proper dictionaries/models
4. **Error Management**: Comprehensive error handling with `ToolError`
5. **Context Integration**: Use FastMCP context for progress/logging
6. **Resource Management**: Expose example templates as MCP resources
7. **Production Ready**: Authentication and deployment configuration