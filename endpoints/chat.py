from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from handlers.base import BaseHandler
from typing import Optional
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


class ChatModel(BaseModel):
    query: str
    model: str = "gpt-4-0613"
    temperature: float
    vector_fetch_k: Optional[int] = 5 # Number of vectors to fetch from Pinecone as source documents
    chat_history: list[str] = [] # Example input: [("You are a helpful assistant.", "What is your name?")]
    namespace: Optional[str] = None 

@router.post("/chat")
async def chat( 
    chat_model: ChatModel,
):
    available_models = ["gpt-4-0613", "gpt-4o-2024-05-13"]

    if chat_model.model not in available_models:
        raise HTTPException(status_code=400, detail=f"Invalid model name. Please select a valid model from the list of available models: \n{str(available_models)}")
    
    if chat_model.temperature < 0.0 or chat_model.temperature > 2.0:
        raise HTTPException(status_code=400, detail="Invalid temperature value. Please select a value between 0.0 and 2.0")

    handler = BaseHandler(chat_model=chat_model.model, openai_chat_temperature=chat_model.temperature)
    logging.info("Chat model initialized with query: %s", chat_model.query)
    logging.info("Chat history: %s", chat_model.chat_history)
    logging.info("Namespace: %s", chat_model.namespace)
    logging.info("Vector fetch k: %s", chat_model.vector_fetch_k)
    response = handler.chat(
        chat_model.query, 
        chat_model.temperature,
        chat_model.chat_history,
        namespace=(chat_model.namespace or None),
        search_kwargs=({"k": chat_model.vector_fetch_k} or {"k": 5})
    )
    return {"response": response}
    