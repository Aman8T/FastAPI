from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Optional


# Load environment variables from .env file (if any)
load_dotenv()

class Response(BaseModel):
    result: Optional[str] = None 

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000"
    "https://web-production-41bf.up.railway.app"
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"Hello": "World from FastAPI"}


@app.post('/predict',response_model = Response)
def predict() -> Any:
  
  #implement this code block
  
  return {"result": "hello world!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=os.getenv("PORT", default=8080), log_level="info")
