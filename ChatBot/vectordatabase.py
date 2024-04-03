from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

import os
#from dotenv import load_dotenv

#Load environmental variables from .env-file
#load_dotenv()

def load_documents(index_name):
    loader = CSVLoader(index_name, source_column="speech_content") #unprocessed csv file
    #loader = DataFrameLoader(data_frame=df, page_content_column='speech_content') #df
    data = loader.load()
    splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=32,
            length_function=len,
            is_separator_regex=False,
        )
    documents = splitter.split_documents(documents=data)
    return documents

def get_vectorstore(documents, embeddings, index_name): 
    if embeddings == None:
        embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(folder_path="./vector_store", index_name=index_name)
    return db

def RAG(llm, prompt, db, question):
    if llm == None:
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

    if prompt == None:
        prompt = ChatPromptTemplate.from_template("""Beantworte die folgende Frage auf deutsch und nur auf der Grundlage des angegebenen Kontexts:

        <context>
        {context}
        </context>

        Question: {input}
        Gebe nur die Antwort auf die Queston zurück""")
       
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retriever = db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": question})
    return response