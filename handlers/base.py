import tempfile
import pinecone
import os
from utils.alerts import alert_exception, alert_info
from typing import List
from pinecone.core.client.exceptions import ApiException
from langchain.chains import ConversationalRetrievalChain
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, Docx2txtLoader
from fastapi import UploadFile
from fastapi import HTTPException
from dotenv import load_dotenv
import threading
from langchain.text_splitter import (
    TokenTextSplitter,
    TextSplitter,
    Tokenizer,
    Language,
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
    LatexTextSplitter,
    PythonCodeTextSplitter,
    KonlpyTextSplitter,
    SpacyTextSplitter,
    NLTKTextSplitter,
    SentenceTransformersTokenTextSplitter,
    ElementType,
    HeaderType,
    LineType,
    HTMLHeaderTextSplitter,
    MarkdownHeaderTextSplitter,
    MarkdownTextSplitter,
    CharacterTextSplitter,
)
from langchain_milvus import Milvus
from handlers.constant import (
    INDEX_PARAMS,
    CONTENT,    
    CONTENT_VECTOR,
    SPLITTER_MAP,
)
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
import time
import logging
import httpx

load_dotenv()

class BaseHandler():
    def __init__(
            self,
            chat_model: str = 'gpt-4-0613',
            temperature: float = 0.7,
            **kwargs
        ):
        
        self.milvus_db_name = os.getenv("MILVUS_DB_NAME")
        self.milvusClient = MilvusClient(
            uri=os.getenv("MILVUS_URL"),
            token=os.getenv("MILVUS_TOKEN"),
            db_name=self.milvus_db_name,
        )
        self.milvus_connection_args = {
            "uri": os.getenv("MILVUS_URL"),
            "host": os.getenv("MILVUS_HOST"),
            "port": os.getenv("MILVUS_PORT"),
            "user": os.getenv("MILVUS_USER"),
            "password": os.getenv("MILVUS_PASSWORD"),
            "db_name": self.milvus_db_name,
            "secure": False,
        }
        self.content_vector = "content_vector"
        self.content = "content"
        self.index_params = {
            "field_name": self.content_vector,
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "index_name": "vector_index",
            "params": {"nlist": 1024},
        }
        client = httpx.Client(verify=False)
        self.llm_map = {
            "gpt-4-0613": lambda: ChatOpenAI(
                model=os.getenv("SILICONFLOW_LLM_MODEL"),
                openai_api_base=os.getenv("SILICONFLOW_URL"),
                openai_api_key=os.getenv("SILICONFLOW_API_KEY"),
                temperature=temperature,
                http_client=client,
            ),
            "gpt-4o-2024-05-13": lambda: ChatOpenAI(
                model=os.getenv("SILICONFLOW_LLM_MODEL"),
                openai_api_base=os.getenv("SILICONFLOW_URL"),
                openai_api_key=os.getenv("SILICONFLOW_API_KEY"),
                temperature=temperature,
                http_client=client,
            ),
        }
        self.chat_model = chat_model
        self.dimensions = 3072
        # self.embeddings = OpenAIEmbeddings(
        #     model='text-embedding-3-small',  
        #     openai_api_key=os.getenv("OPENAI_API_KEY"),
        # )
        
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv("SILICONFLOW_EMBEDDINGS_MODEL"),
            base_url=os.getenv("SILICONFLOW_URL"),
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            http_client=client,
        )


    def load_documents(self, files: list[UploadFile], namespace: str = None) -> list[list[str]]:
        documents = []
        loader_map = {
            'txt': TextLoader,
            'pdf': PyMuPDFLoader, 
            'docx': Docx2txtLoader,
        }

        allowed_extensions = [key for key in loader_map.keys()]
        try: 
            for file in files:
                if file.filename.split(".")[-1] not in allowed_extensions:
                    raise HTTPException(status_code=400, detail="File type not permitted")
                
                with tempfile.NamedTemporaryFile(delete=True, prefix=file.filename + '___') as temp:
                    temp.write(file.file.read())
                    temp.seek(0)
                    loader = loader_map[file.filename.split(".")[-1]](temp.name)
                    documents.append(loader.load())
        except Exception as e:
            alert_exception(e, "Error loading documents")
            raise HTTPException(status_code=500, detail=f"Error loading documents: {str(e)}")

        
        return documents
    
    def ingest_documents(
        self,
        documents: list[list[str]],
        namespace: str = None,
        drop_value: str = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        **kwargs,
    ):
        """
        documents: list of loaded documents
        chunk_size: number of documents to ingest at a time
        chunk_overlap: number of documents to overlap when ingesting

        kwargs:
            split_method: 'recursive', 'token', 'text', 'tokenizer', 'language', 'json', 'latex', 'python', 'konlpy', 'spacy', 'nltk', 'sentence_transformers', 'element_type', 'header_type', 'line_type', 'html_header', 'markdown_header', 'markdown', 'character'
        """
        # self.createDatabase()
        if namespace is None:
            namespace = ""
        splitter_map = {
            "recursive": RecursiveCharacterTextSplitter,
            "token": TokenTextSplitter,
            "text": TextSplitter,
            "json": RecursiveJsonSplitter,
            "latex": LatexTextSplitter,
            "python": PythonCodeTextSplitter,
            "konlpy": KonlpyTextSplitter,
            "spacy": SpacyTextSplitter,
            "nltk": NLTKTextSplitter,
            "sentence_transformers": SentenceTransformersTokenTextSplitter,
            "html_header": HTMLHeaderTextSplitter,
            "markdown_header": MarkdownHeaderTextSplitter,
            "markdown": MarkdownTextSplitter,
            "character": CharacterTextSplitter,
        }

        split_method = kwargs.get("split_method", "markdown")
        test_splitter = splitter_map[split_method](
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        alert_info(
            f"Ingesting {len(documents)} document(s)...\nParams: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, split_method={split_method}"
        )
        for document in documents:
            split_document = test_splitter.split_documents(document)
            # print(split_document)
            drop_old = True if str(drop_value) == "True" else False
            try:
                Milvus.from_documents(
                    documents=split_document,
                    embedding=self.embeddings,
                    collection_name=namespace,
                    collection_description=namespace + "doc desc",
                    connection_args=self.milvus_connection_args,
                    index_params=self.index_params,
                    text_field=self.content,
                    vector_field=self.content_vector,
                    num_shards=4,
                    drop_old=drop_old,
                )
            except Exception as e:
                alert_exception(
                    e,
                    "Error ingesting documents - Make sure you're dimensions match the embeddings model (1536 for text-embedding-3-small, 3072 for text-embedding-3-large)",
                )
                raise HTTPException(
                    status_code=500, detail=f"Error ingesting documents: {str(e)}"
                )

    def chat(
        self,
        query: str,
        templature: float = 0.7,
        chat_history: list[tuple[str, str]] = [],
        namespace: str = None,
        **kwargs,
    ):
        """
        query: str
        chat_history: list of previous chat messages
        kwargs:
            namespace: str
            search_kwargs: dict
        """
        try:
            print(query)
            print(namespace)
            print("start query")
            # Define the retriever
            vectorstore = Milvus(
                collection_name=namespace,
                connection_args=self.milvus_connection_args,
                embedding_function=self.embeddings,
                index_params=self.index_params,
                consistency_level="STRONG",
                text_field=self.content,
                vector_field=self.content_vector,
            )
            search_kwargs = kwargs.get("search_kwargs", {"k": 5})
            search_kwargs["nprobe"] = 10  # Add nprobe parameter
            search_kwargs["score_threshold"] = templature
            retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold", search_kwargs=search_kwargs
            )

            if not chat_history:
                template = (
                    "You are a helpful assistant. Your task is to convert the follow-up question into a standalone question.\n\n"
                    "Follow-up Question:\n{question}\n\n"
                    "Standalone Question:"
                )
            else:
                template = (
                    "You are a helpful assistant. Your task is to combine the chat history and the follow-up question into a standalone question.\n\n"
                    "Chat History: {chat_history}"
                    "Follow-up Question: {question}"
                    "Standalone Question:"
                )
            prompt = PromptTemplate.from_template(template)

            # Initialize the language model
            llm = self.llm_map[self.chat_model]()

            # Create the conversational retrieval chain
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                condense_question_prompt=prompt,
                chain_type="stuff",
                verbose=True,
                return_source_documents=True,
            )

            result = chain.invoke({"question": query, "chat_history": chat_history})
            return result
        except Exception as e:
            alert_exception(e, "Error chatting")
            raise HTTPException(status_code=500, detail=f"Error chatting: {str(e)}")

    