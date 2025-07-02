
import os
import gradio as gr
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain_groq import ChatGroq
from pathlib import Path
import pickle

# Function to load or create vector store
def load_or_create_vector_store(pdf_path, cache_path="vector_store.faiss"):
    try:
        # Initialize embeddings
        embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Check if cached vector store exists
        if Path(cache_path).exists():
            print("Loading cached vector store...")
            return FAISS.load_local(cache_path, embedder, allow_dangerous_deserialization=True)

        # Load and process PDF
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        loader = PDFPlumberLoader(pdf_path)
        docs = loader.load()
        if not docs:
            raise ValueError("No content extracted from the PDF.")

        # Split documents
        text_splitter = SemanticChunker(embedder)
        documents = text_splitter.split_documents(docs)

        # Create and save vector store
        vector_store = FAISS.from_documents(documents, embedder)
        vector_store.save_local(cache_path)
        return vector_store
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")

# Initialize language model
def initialize_llm(api_key):
    try:
        return ChatGroq(
            groq_api_key=api_key,
            temperature=0,
            model_name="deepseek-r1-distill-llama-70b"
        )
    except Exception as e:
        raise Exception(f"Error initializing LLM: {str(e)}")

# Define prompt template
prompt_template = """
Use the following context to answer the question. If you don't know the answer, say so. Keep the answer to 3 sentences. Always end with "Thanks for asking!"
{context}
Question: {question}
Helpful Answer:
"""
QA_PROMPT = PromptTemplate.from_template(prompt_template)

# Gradio interface function
def query_rag(pdf_file, question, api_key, k=2, chat_history=[]):
    try:
        if not question.strip():
            return "Please enter a valid question.", [], chat_history
        if not api_key:
            return "Please provide a valid Grok API key.", [], chat_history
        if not pdf_file:
            return "Please upload a valid PDF file.", [], chat_history

        # Handle Gradio file upload
        pdf_path = "temp_uploaded_pdf.pdf"
        if isinstance(pdf_file, str):  # Gradio provides file path
            pdf_path = pdf_file
        else:  # Gradio provides file-like object
            with open(pdf_path, "wb") as f:
                f.write(pdf_file)  # Write file content directly

        # Load or create vector store
        vector_store = load_or_create_vector_store(pdf_path)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})

        # Initialize LLM
        llm = initialize_llm(api_key)

        # Set up conversational chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            return_source_documents=True
        )

        # Run query
        result = qa_chain({"question": question, "chat_history": chat_history})
        answer = result["answer"]
        sources = [doc.page_content[:200] + "..." for doc in result["source_documents"]]

        # Update chat history
        chat_history.append((question, answer))

        return answer, sources, chat_history
    except Exception as e:
        return f"Error: {str(e)}", [], chat_history

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# RAG Question-Answering System")
    gr.Markdown("Upload a PDF and ask questions about its content. Provide your Grok API key to proceed.")

    api_key = gr.Textbox(label="Grok API Key", type="password", placeholder="Enter your GROQ_API_KEY")
    pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
    question_input = gr.Textbox(label="Ask a Question", placeholder="e.g., What is a cost function in ML?")
    k_slider = gr.Slider(minimum=1, maximum=5, value=2, step=1, label="Number of Retrieved Documents")
    output = gr.Textbox(label="Answer")
    sources = gr.Textbox(label="Source Documents (First 200 Characters)")
    chat_history = gr.State(value=[])

    submit_btn = gr.Button("Submit")
    submit_btn.click(
        fn=query_rag,
        inputs=[pdf_input, question_input, api_key, k_slider, chat_history],
        outputs=[output, sources, chat_history]
    )

# Launch the interface
demo.launch()
