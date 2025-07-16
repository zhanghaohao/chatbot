from fastapi import FastAPI
import uvicorn
import logging
from colorama import Fore, Style, init

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logging.addLevelName(logging.INFO, f"{Fore.GREEN}{Style.BRIGHT}{logging.getLevelName(logging.INFO)}{Style.RESET_ALL}")

app = FastAPI()

@app.get("/")
def read_root():
    return {"detail": "Langchain Chatbot is Running!"}

from endpoints import (
    ingest,
    chat,
)

for endpoint in [ingest, chat]:
    app.include_router(endpoint.router)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("Starting Langchain Chatbot API")
    uvicorn.run('app:app', port=9092, reload=True)