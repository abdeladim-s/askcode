#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Command Line Interface
"""


from askcode.main import AskCode, console
import fire
import importlib.metadata
from langchain.llms.utils import enforce_stop_tokens

__author__ = "abdeladim-s"
__copyright__ = "Copyright 2023,"
__version__ = importlib.metadata.version('askcode')
__github__ = "https://github.com/abdeladim-s/askcode"


__header__ = f"""

   ▄████████    ▄████████    ▄█   ▄█▄       ▄████████  ▄██████▄  ████████▄     ▄████████ 
  ███    ███   ███    ███   ███ ▄███▀      ███    ███ ███    ███ ███   ▀███   ███    ███ 
  ███    ███   ███    █▀    ███▐██▀        ███    █▀  ███    ███ ███    ███   ███    █▀  
  ███    ███   ███         ▄█████▀         ███        ███    ███ ███    ███  ▄███▄▄▄     
▀███████████ ▀███████████ ▀▀█████▄         ███        ███    ███ ███    ███ ▀▀███▀▀▀     
  ███    ███          ███   ███▐██▄        ███    █▄  ███    ███ ███    ███   ███    █▄  
  ███    ███    ▄█    ███   ███ ▀███▄      ███    ███ ███    ███ ███   ▄███   ███    ███ 
  ███    █▀   ▄████████▀    ███   ▀█▀      ████████▀   ▀██████▀  ████████▀    ██████████ 
                            ▀                                                            
[*] GITHUB: {__github__}
[*] VERSION: {__version__}

"""
def main(
    codebase_path: str = '.',
    language: str = 'python',
    parser_threshold: int = 0,
    text_splitter_chunk_size: int = 256,
    text_splitter_chunk_overlap: int = 50,
    use_HF: bool = True,
    llm_model: str = "TheBloke/CodeLlama-7B-GPTQ",
    embeddings_model: str = "sentence-transformers/all-MiniLM-L12-v2",
    retriever_search_type: str = "mmr",
    retriever_k: int = 4,
    max_new_tokens: int = 50,
    temperature: float = 0.1,
    top_p: float = 0.9,
    repetition_penalty: float = 1.,
    use_autogptq: bool = True):
    """
    Chat with your code base with the power of LLMs.

    :param codebase_path: path to your codebase
    :param language: programming language ['python', 'javascript'] at the moment
    :param parser_threshold: minimum lines needed to activate parsing (0 by default).
    :param text_splitter_chunk_size: Maximum size of chunks to return
    :param text_splitter_chunk_overlap: Overlap in characters between chunks
    :param use_HF: use hugging face models, if False OpenAI models will be used
    :param llm_model: Large language model name (HF model name or OpenAI model)
    :param embeddings_model: Embeddings model (HF model name or OpenAI model)
    :param retriever_search_type: Defines the type of search that
                the Retriever should perform.
                Can be "similarity" (default), "mmr", or
                "similarity_score_threshold".
    :param retriever_k: Amount of documents to return (Default: 4)
    :param max_new_tokens: Maximum tokens to generate
    :param temperature: sampling temperature
    :param top_p: sampling top_p
    :param repetition_penalty: sampling repetition_penalty
    :param use_autogptq: Set it to True to use Quantized AutoGPTQ models

    :return: None
    """

    ask_code = AskCode(codebase_path,
                 language,
                 parser_threshold,
                 text_splitter_chunk_size,
                 text_splitter_chunk_overlap,
                 use_HF,
                 llm_model,
                 embeddings_model,
                 retriever_search_type,
                 retriever_k,
                 max_new_tokens,
                 temperature,
                 top_p,
                 repetition_penalty,
                 use_autogptq)
    console.print(__header__, style="blue")
    ask_code.setup()
    console.print("CTRL+C To stop ...", style="bold red")
    print()
    while True:
        try:
            q = console.input("[yellow][-] How can I help you: ")
            with console.status("[bold green]Searching ...") as status:
                res = ask_code.ask(q)
                ans = enforce_stop_tokens(res['output_text'], ["Question"])
                # ans = res['output_text']
                console.print(f"[+] Answer: {ans}", style="bold green")
        except KeyboardInterrupt:
            break

def run():
    fire.Fire(main)


if __name__ == '__main__':
    run()
