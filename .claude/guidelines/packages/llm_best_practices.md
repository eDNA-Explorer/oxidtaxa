## LLM Service Integration

All LLM integrations use the `llm-lib` library (`libraries/llm-lib/`) with hexagonal architecture (ports and adapters pattern). Currently Gemini-only (`GeminiAdapter`).

### Using LLM Services

```python
from llm_lib import LLMService, LLMRequest, LLMServiceFactory

# Create LLM service (Gemini is the only provider)
llm_service = LLMServiceFactory.create("gemini", api_key="your-key")

# Make request
request = LLMRequest(
    prompt="Analyze this data...",
    system_instruction="You are an expert...",
    temperature=0.7,
)

response = llm_service.generate(request)
print(response.text)
```

### Injecting LLM Services in Core Functions

All core functions that use LLMs accept `llm_service: LLMService | None = None` for dependency injection. When `None`, they create a default Gemini instance internally.

```python
from edna_explorer_reports.core.organisms.verify_taxonomy import (
    verify_taxonomic_consistency
)

# Use default Gemini (reads GEMINI_API_KEY env var)
result, usage = verify_taxonomic_consistency(
    inaturalist_data=data,
    ncbi_data=data,
    gbif_data=data,
    wikipedia_data=data,
)

# Or inject custom service (e.g., for testing)
from llm_lib.mocks import MockLLMService
mock_llm = MockLLMService(response_text="Test response")
result, usage = verify_taxonomic_consistency(
    inaturalist_data=data,
    ncbi_data=data,
    gbif_data=data,
    wikipedia_data=data,
    llm_service=mock_llm
)
```

### Testing with Mock LLM

```python
from llm_lib.mocks import MockLLMService

mock_llm = MockLLMService(response_text="Test response")
result = my_function(data, llm_service=mock_llm)

# Verify calls
assert len(mock_llm.calls) == 1
assert mock_llm.calls[0].temperature == 0.2
```

`MockLLMService` also supports `structured_response` (pre-built Pydantic instance for `generate_structured()`) and `should_fail=True` (raises `RuntimeError` to test error paths).

### Error Handling

There is **no retry logic or rate limiting** in the library — each call is a single attempt. LLM calls raise `RuntimeError` on failure. Callers must wrap calls in try/except.
