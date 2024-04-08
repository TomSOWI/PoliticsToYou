import gradio as gr
from ChatBot import chatbot, keyword_search

# Adjust size of each block is not yet working
output = gr.DataFrame(height=1000, show_label=True, scale=2)
input = gr.Textbox(scale=1)

with gr.Blocks() as App:
    with gr.Tab("ChatBot"):
        # Apply RAG using chatbut function from local file ChatBot.py
        gr.ChatInterface(chatbot,
                    title="PoliticsToYou",
                    description= "This chatbot uses the infomation of speeches of the german parliament (since 2021) \
                        to get insight on the view points of the german parties and the debate of the parliament.",
                    examples=["Wie steht die CDU zur Cannabislegalisierung?","Was waren die wichtigsten Themen in der aktuellen Legislaturperiode?"], #change to meaningful examples
                    cache_examples=False, #true increases the loading time
                    )
    with gr.Tab("KeyWordSearch"):
        gr.Interface(fn=keyword_search, inputs=input, outputs=output)
         
        
if __name__ == "__main__":
    App.launch(share=True) 


