from langchain_community.document_loaders import DataFrameLoader
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


# Load documents to create a vectorstore later
def load_documents(df):
    # To Do: Create one initial vectore store loading all the documents with this function
    #loader = CSVLoader(index_name, source_column="speech_content") #unprocessed csv file
    loader = DataFrameLoader(data_frame=df, page_content_column='speech_content') #df
    data = loader.load()
    splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=32,
            length_function=len,
            is_separator_regex=False,
        )
    documents = splitter.split_documents(documents=data)
    return documents

def get_vectorstore(embeddings, folder_path, index_name): 
    path = folder_path + "/" + index_name
    print(path)
    # To Do: Dynamicly update and merge verctorstores
    #if os.path.exists(path):
    db = FAISS.load_local(folder_path=folder_path, index_name=index_name,
                                            embeddings=embeddings, allow_dangerous_deserialization=True)
    #else:
        #db = FAISS.from_documents(documents, embeddings)
        #db.save_local(folder_path=folder_path, index_name=index_name)
        #pass
    return db

# Apply RAG by providing the context and the question to the LLM using the predefined template
def RAG(llm, prompt, db, question):       
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retriever = db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": question})
    return response

