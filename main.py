from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, Query, Form, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import Chroma
import chromadb
from sentence_transformers import SentenceTransformer
from model import database, AI
# Load environment variables from .env file (if any)
load_dotenv()

chroma_client = chromadb.Client()
collection= chroma_client.get_or_create_collection(name="new_collection")
directory = Path() / 'Data'

Database= database(collection)
model = AI(method='api')


class Response(BaseModel):
    result: str | None



origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000"
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




@app.post("/upload", response_model=Response)
async def upload(file: UploadFile = File(...)) -> Any:
    # Save the file to the specified directory
    global collection
    data= await file.read()
    
    file_path = directory / file.filename

    with open(file_path, "wb") as buffer:
        buffer.write(data)
        buffer.close()
    
   
    collection = Database.load_data(str(file_path))
    
# print results
    
        
    
    
    return {"result": f"{file.filename}"}



@app.post("/predict", response_model=Response)
def predict(question: dict) -> Any:
    Question=question.get('quetion','')

    docs=collection.query(query_texts=[Question],n_results=2)
    
    result= model.answer(docs['documents'][0][0],Question)
    return {"result": f"{result}"}


    