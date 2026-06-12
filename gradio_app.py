import gradio as gr

from app import app


def chatbot(question):

    state = {
        "question": question,
        "route": "",
        "context": "",
        "answer": ""
    }

    result = app.invoke(state)

    return result["answer"]


interface = gr.Interface(
    fn=chatbot,
    inputs="text",
    outputs="text",
    title="Agentic RAG Chatbot"
)

interface.launch()