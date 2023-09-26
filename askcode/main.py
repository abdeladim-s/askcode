#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file contains the definition of AskCode main class
"""

__author__ = "abdeladim-s"
__copyright__ = "Copyright 2023,"


from pathlib import Path
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from dataclasses import dataclass
from rich.console import Console

console = Console()

@dataclass
class Args:
    codebase_path: str = '.'
    language: str = 'python'
    parser_threshold: int = 500
    text_splitter_chunk_size: int = 256
    text_splitter_chunk_overlap: int = 50
    use_HF: bool = True
    llm_model: str = "TheBloke/CodeLlama-7B-GPTQ"
    embeddings_model: str = "sentence-transformers/all-MiniLM-L12-v2"
    retriever_search_type: str = "mmr"
    retriever_k: int = 4
    max_new_tokens: int = 50,
    temperature: float = 0.1
    top_p: float = 0.9
    repetition_penalty: float = 1.
    use_autogptq: bool = True



class AskCode:
    def __init__(self,
                 codebase_path: str,
                 language: str,
                 parser_threshold: int,
                 text_splitter_chunk_size: int,
                 text_splitter_chunk_overlap: int,
                 use_HF: bool,
                 llm_model: str,
                 embeddings_model: str,
                 retriever_search_type: str,
                 retriever_k: int,
                 max_new_tokens: int,
                 temperature: float,
                 top_p: float,
                 repetition_penalty: float,
                 use_autogptq: bool,
                 ):
        self.codebase_path = Path(codebase_path)
        self.language = language
        self.parser_threshold = parser_threshold
        self.text_splitter_chunk_size = text_splitter_chunk_size
        self.text_splitter_chunk_overlap = text_splitter_chunk_overlap

        self.use_HF = use_HF
        self.llm_model = llm_model
        self.embeddings_model = embeddings_model

        self.retriever_search_type = retriever_search_type
        self.retriever_k = retriever_k

        self.use_autogptq = use_autogptq

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty


    def setup(self) -> None:
        """
         Sets up the Necessary components for the langchain chain

        :return: None
        """
        with console.status("[bold green]Loading files ...") as status:
            self.retriever = self.load_retriever()
            console.log(f"[bold green]Files loaded successfully")

        with console.status("[bold green]Loading LLM ...") as status:
            self.llm = self.load_llm()
            console.log(f"[bold green]LLM loaded successfully")

        self.prompt_template = self.get_prompt_template()

    def load_retriever(self):
        """
        Loads the files from the codebase and sets up the retriever
        """
        try:
            with open(self.codebase_path / '.gitignore') as f:
                exclude = f.readlines()
        except Exception as e:
            # no gitignore found
            exclude = []

        loader = GenericLoader.from_filesystem(
            self.codebase_path,
            glob="**/[!.]*",
            exclude=exclude,
            suffixes=[".py", ".js"],  # only python and javascript atm
            show_progress=True,
            parser=LanguageParser(language=self.language, parser_threshold=self.parser_threshold)
        )
        files = loader.load()
        splitter = RecursiveCharacterTextSplitter.from_language(language=self.language,
                                                       chunk_size=self.text_splitter_chunk_size,
                                                       chunk_overlap=self.text_splitter_chunk_overlap)
        docs = splitter.split_documents(files)

        if self.use_HF:
            db = Chroma.from_documents(docs, HuggingFaceEmbeddings(model_name=self.embeddings_model))
        else:
            # defaults to OpenAI
            from langchain.embeddings import OpenAIEmbeddings
            db = Chroma.from_documents(docs, OpenAIEmbeddings(disallowed_special=()))

        retriever = db.as_retriever(
            search_type=self.retriever_search_type,
            search_kwargs={"k": self.retriever_k},
        )

        return retriever

    def load_llm(self):
        """
        Sets up the LLM
        """
        if self.use_HF:
            tokenizer = AutoTokenizer.from_pretrained(self.llm_model, use_fast=True)
            if self.use_autogptq:
                from auto_gptq import AutoGPTQForCausalLM
                model = AutoGPTQForCausalLM.from_quantized(self.llm_model, use_safetensors=True)
            else:
                model = AutoModelForCausalLM.from_pretrained(self.llm_model)

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
            )
            llm = HuggingFacePipeline(pipeline=pipe)
            return llm
        else:
            # defaults to OpenAI
            from langchain.chat_models import ChatOpenAI
            return ChatOpenAI(model_name=self.llm_model)

    def get_prompt_template(self):
        """
        Sets up the prompt template
        """
        template = """Use the following pieces of context to answer the question at the end. 
           If you don't know the answer, just say that you don't know, don't try to make up an answer. 
           Use three sentences maximum and keep the answer as concise as possible. 
           {context}
           Question: {question}
           Helpful Answer:"""

        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )

        return prompt_template

    def chain(self, retriever, llm, prompt_template, question: str):
        """
        Runs a question through Langchain Chain

        :param retriever: the docs Retriever
        :param llm: the large language model
        :param prompt_template: the prompt template
        :param question: the question

        :return: chain results
        """
        relevant_docs = retriever.get_relevant_documents(question)
        chain = load_qa_chain(llm, prompt=prompt_template)
        return chain({"input_documents": relevant_docs, "question": question}, return_only_outputs=False)

    def ask(self, question: str):
        """
        Ask a question to the codebase
        You need to call `self.setup` before calling this function

        :param question: the question :)
        :return: chain results
        """
        return self.chain(self.retriever, self.llm, self.prompt_template, question)
