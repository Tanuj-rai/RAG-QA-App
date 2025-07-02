# RAG Question-Answering System with Gradio
A Retrieval-Augmented Generation (RAG) system to answer questions from a PDF using LangChain, FAISS, and Grok, with a Gradio web interface.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Set `GROQ_API_KEY` in Colab secrets or input it in the Gradio interface.
3. Run: `python RAG_with_Gradio.py`

## Requirements
See `requirements.txt` for dependencies.

## Usage
- Upload a PDF via the Gradio interface.
- Enter a question and adjust retrieved documents if needed.
- View the answer and source document excerpts.
