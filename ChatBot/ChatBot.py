from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings

from vectordatabase import RAG, get_vectorstore
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv

#Load environmental variables from .env-file
load_dotenv(find_dotenv())
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGACEHUB_API_TOKEN")


embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
llm = HuggingFaceHub(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="text-generation",
    model_kwargs={
        "max_new_tokens": 512,
        "top_k": 30,
        "temperature": 0.1,
        "repetition_penalty": 1.03,
        }
)
# To Do: Experiment with different templates replying in german or english depending on the input language
prompt1 = ChatPromptTemplate.from_template("""<s>[INST] 
                    Instruction: Beantworte die folgende Frage auf deutsch und nur auf der Grundlage des angegebenen Kontexts:

                    Context: {context}

                    Question: {input}  
                    [/INST]"""
                    # Returns the answer in Emglish!?
) 

prompt2 = ChatPromptTemplate.from_template("""Beantworte die folgende Frage auf deutsch und nur auf der Grundlage des angegebenen Kontexts:

        <context>
        {context}
        </context>

        Question: {input}
        Gebe nur die Antwort auf die Queston zurück""")

 
folder_path = "./vector_store"
index_name = "legislature20"
db = get_vectorstore(embeddings=embeddings, folder_path=folder_path, index_name=index_name)

def chatbot(message, history, db=db, llm=llm, prompt=prompt2):
    raw_response = RAG(llm=llm, prompt=prompt, db=db, question=message)
    response = raw_response['answer'].split("Antwort: ")[1]
    return response  