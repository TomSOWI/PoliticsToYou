# Create vectorstore
import pandas as pd
from vectordatabase import load_documents
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


df = pd.read_pickle("ChatBot\Speeches\speeches_1949_09_12.pkl")
# Split speeches into documents
documents = load_documents(df)
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
db = FAISS.from_documents(documents, embeddings)
db.save_local(folder_path="ChatBot\FAISS", index_name="speeches_1949_09_12")


