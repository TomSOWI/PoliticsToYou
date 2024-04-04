import gradio as gr
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from ChatBot import chatbot
 

with gr.Blocks() as App:
    with gr.Tab("ChatBot"):
        # Apply RAG using chatbut function from local file ChatBot.py
        gr.ChatInterface(chatbot,
                    title="PoliticsToYou",
                    description= "This chatbot uses the infomation of all speeches of the german parliament \
                        to get insight on the view points of the german parties and the debate of the parliament.",
                    examples=["Wie steht die CDU zur Cannabislegalisierung?","Was waren die wichtigsten Themen in der aktuellen Legislaturperiode?"], #change to meaningful examples
                    cache_examples=False, #true increases the loading time
                    )
    #with gr.Tab("KeyWordSearch"):
        #keyword = gr.Textbox.input()
        #gr.DataFrame()

App.launch(share=True)

