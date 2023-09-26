# AskCode

Ask questions to your code base using the power of LLMs.

# Installation

```shell
pip install git+https://github.com/abdeladim-s/askcode
```

# Usage

```shell
askcode --help

NAME
    askcode - Chat with your code base with the power of LLMs.

SYNOPSIS
    askcode <flags>

DESCRIPTION
    Chat with your code base with the power of LLMs.

FLAGS
    -c, --codebase_path=CODEBASE_PATH
        Type: str
        Default: '.'
        path to your codebase
    --language=LANGUAGE
        Type: str
        Default: 'python'
        programming language ['python', 'javascript'] at the moment
    -p, --parser_threshold=PARSER_THRESHOLD
        Type: int
        Default: 0
        minimum lines needed to activate parsing (0 by default).
    --text_splitter_chunk_size=TEXT_SPLITTER_CHUNK_SIZE
        Type: int
        Default: 256
        Maximum size of chunks to return
    --text_splitter_chunk_overlap=TEXT_SPLITTER_CHUNK_OVERLAP
        Type: int
        Default: 50
        Overlap in characters between chunks
    --use_HF=USE_HF
        Type: bool
        Default: True
        use hugging face models, if False OpenAI models will be used
    --llm_model=LLM_MODEL
        Type: str
        Default: 'TheBloke/CodeLlama-7B-...
        Large language model name (HF model name or OpenAI model)
    -e, --embeddings_model=EMBEDDINGS_MODEL
        Type: str
        Default: 'sentence-...
        Embeddings model (HF model name or OpenAI model)
    --retriever_search_type=RETRIEVER_SEARCH_TYPE
        Type: str
        Default: 'mmr'
        Defines the type of search that the Retriever should perform. Can be "similarity" (default), "mmr", or "similarity_score_threshold".
    --retriever_k=RETRIEVER_K
        Type: int
        Default: 4
        Amount of documents to return (Default: 4)
    -m, --max_new_tokens=MAX_NEW_TOKENS
        Type: int
        Default: 50
        Maximum tokens to generate
    --temperature=TEMPERATURE
        Type: float
        Default: 0.1
        sampling temperature
    --top_p=TOP_P
        Type: float
        Default: 0.9
        sampling top_p
    --repetition_penalty=REPETITION_PENALTY
        Type: float
        Default: 1.0
        sampling repetition_penalty
    --use_autogptq=USE_AUTOGPTQ
        Type: bool
        Default: True
        Set it to True to use Quantized AutoGPTQ models
```

# License

MIT