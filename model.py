from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_community.vectorstores import Chroma
import chromadb #python -m chromadb.cli.cli run --host localhost --port 9010
from sentence_transformers import SentenceTransformer
import torch


chroma_client = chromadb.Client()

# tokenizer = AutoTokenizer.from_pretrained("./gemma-2b-it")
# model = AutoModelForCausalLM.from_pretrained("./gemma-2b-it", device_map="auto")

# input_text = "Write me a poem about Machine Learning."
# input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# outputs = model.generate(**input_ids)
# print(tokenizer.decode(outputs[0])
collection= chroma_client.get_or_create_collection(name="new_collection")
class database:
    def __init__(self,collection):
        self.data = []
        self.collection=collection
        self.embeddings_func= SentenceTransformer("./all-MiniLM-L6-v2")
        
        
    def load_data(self,filename):
         
         data=self.data
         if '.csv' in filename:
              loader = CSVLoader(filename)
              self.data = loader.load()
         elif '.txt' in filename:
              loader=TextLoader(filename)
              self.data = loader.load()
         elif '.docx' in filename:
              loader=Docx2txtLoader(filename)
              self.data = loader.load()
         elif '.pdf' in filename:
              loader=PyPDFLoader(filename)
              self.data= loader.load() 
    
         data=self.data
         collection=self.collection
         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
         texts = text_splitter.split_documents(data)
         texts = [texts[i].page_content for i in range(len(texts))]
         embeddings = self.embeddings_func.encode(texts) 
         print(embeddings.shape)
         collection.upsert(embeddings=embeddings,documents=texts,ids=[str(i) for i in range(0,len(texts))])
         return collection




class AI:
     def  __init__(self,model_id="./gemma-2b-it",method='gpu'):
          self.method=method
          
          if method=='cpu':
               self.tokenizer = AutoTokenizer.from_pretrained(model_id)
               self.model = AutoModelForCausalLM.from_pretrained(model_id)
          if method =='gpu':
               quantization_config = BitsAndBytesConfig(load_in_4bit=True)
               self.tokenizer = AutoTokenizer.from_pretrained(model_id)
               self.model = AutoModelForCausalLM.from_pretrained("./gemma-2b-it",device_map="auto", quantization_config=quantization_config)


     def answer(self,data,question):
          method=self.method
          
          if method=='api':
               API_URL = "https://api-inference.huggingface.co/models/google/gemma-2b-it"
               headers = ""
               def query(payload):
                    response = requests.post(API_URL, headers=headers, json=payload)
                    return response.json()

               input= f"Answer the question given the context\n {question} \n {data} "    
               output = query({
                    "inputs": input,
               })
               return output[0]['generated_text'][len(input):]
          else:
               tokenizer=self.tokenizer
               model=self.model
               chat = [{ "role": "user", "content": f"Answer  context given below /n 'Question': {question}/n 'Data': {data}"  }]
               prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
               inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
               outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=150)

               return tokenizer.decode(outputs[0])
          

          



