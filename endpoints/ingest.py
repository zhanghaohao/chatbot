from fastapi import APIRouter
from typing import List
from fastapi import UploadFile, Form
from handlers.base import BaseHandler
from typing import Optional
import logging

router = APIRouter()

@router.post("/ingest")
async def ingest_documents(
    files: List[UploadFile],
    namespace: Optional[str] = Form(None), 
):
    logging.info("Ingesting documents...")
    handler = BaseHandler()
    documents = handler.load_documents(files, namespace)
    handler.ingest_documents(documents=documents, namespace=namespace)
    return {"message": "Documents ingested"}