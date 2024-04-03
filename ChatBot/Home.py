import gradio as gr
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from vectordatabase import load_documents
from vectordatabase import get_vectorstore
from vectordatabase import RAG

from dotenv import load_dotenv, find_dotenv
import os

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
    #documents = load_documents(index_name="speeches2021.csv")
    #db = get_vectorstore(index_name="legislature20")
db = FAISS.load_local(folder_path="./vector_store", index_name="legislature20",
                                            embeddings=embeddings, allow_dangerous_deserialization=True)
def chatBot(message, history):
    raw_response = RAG(llm=llm, prompt=None, db=db, question=message)
    response = raw_response['answer'].split("Antwort: ")[1]
    return response
   
gr.ChatInterface(chatBot,
                 title="PoliticsToYou",
                 description= "This chatbot uses the infomation of all speeches of the german parliament \
                     to get insight on the view points of the german parties and the debate of the parliament.",
                 examples=["Wie steht die CDU zur Cannabislegalisierung?","Was waren die wichtigsten Themen in der aktuellen Legislaturperiode?"], #change to meaningful examples
                 cache_examples=False, #true increases the loading time
                 ).launch(share=True)

