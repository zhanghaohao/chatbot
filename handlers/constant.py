from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    TextSplitter,
    RecursiveJsonSplitter,
    SentenceTransformersTokenTextSplitter,
    HTMLHeaderTextSplitter,
    MarkdownHeaderTextSplitter,
    MarkdownTextSplitter,
)

CONTENT_VECTOR = "content_vector"
CONTENT = "content"
INDEX_PARAMS = {
    "field_name": CONTENT_VECTOR,
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "index_name": "vector_index",
    "params": {"nlist": 1024},
}
SPLITTER_MAP = {
    "recursive": RecursiveCharacterTextSplitter,
    "token": TokenTextSplitter,
    "text": TextSplitter,
    "json": RecursiveJsonSplitter,
    "sentence_transformers": SentenceTransformersTokenTextSplitter,
    "html_header": HTMLHeaderTextSplitter,
    "markdown_header": MarkdownHeaderTextSplitter,
    "markdown": MarkdownTextSplitter,
}