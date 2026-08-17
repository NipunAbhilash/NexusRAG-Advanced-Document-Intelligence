import os
import logging
import warnings
from datetime import datetime

import gradio as gr
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()

for env_var in ["GOOGLE_CLOUD_PROJECT", "GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_LOCATION", "GOOGLE_GENAI_USE_VERTEXAI"]:
    if env_var in os.environ:
        del os.environ[env_var]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Global state
vector_store = None
retriever = None
qa_chain = None
loaded_doc_names = []
doc_summary = ""

def load_css():
    try:
        with open("style.css", "r") as f:
            return f.read()
    except FileNotFoundError:
        return ""

import httpx
from typing import List

class CustomGeminiEmbeddings:
    def __init__(self, api_key: str, model: str = "models/text-embedding-004"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/{self.model}:embedContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        all_embeddings = []
        for text in texts:
            data = {
                "model": self.model,
                "content": {"parts": [{"text": text}]}
            }
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    import time
                    if attempt > 0:
                        time.sleep(1) # delay before retry
                    else:
                        time.sleep(0.1) # tiny delay to prevent rate limit
                    response = httpx.post(url, headers=headers, json=data, timeout=30.0)
                    if response.status_code != 200:
                        logger.error(f"Embeddings API Error for '{text[:20]}...': {response.text}")
                    response.raise_for_status()
                    res_json = response.json()
                    all_embeddings.append(res_json.get("embedding", {}).get("values", []))
                    break # Success!
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    logger.warning(f"Retrying embedding due to error: {e}")
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        url = f"{self.base_url}/{self.model}:embedContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model,
            "content": {"parts": [{"text": text}]}
        }
        response = httpx.post(url, headers=headers, json=data, timeout=30.0)
        if response.status_code != 200:
            logger.error(f"Embeddings API Error: {response.text}")
        response.raise_for_status()
        return response.json().get("embedding", {}).get("values", [])

def initialize_gemini():
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()

    if not api_key or api_key == "your_google_api_key_here":
        raise ValueError("Google API key is missing. Please add it to the .env file.")

    logger.info("Initializing Google Gemini model...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=api_key,
        temperature=0.3,
        max_output_tokens=8192,
    )

    logger.info("Initializing Custom Gemini Embeddings (Bypassing SDK bug)...")
    embeddings = CustomGeminiEmbeddings(
        api_key=api_key,
        model="models/gemini-embedding-2"
    )

    return llm, embeddings

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def process_documents(file_objs):
    global vector_store, retriever, qa_chain, loaded_doc_names, doc_summary

    if not file_objs:
        return "No files uploaded. Please select one or more PDF files.", "", []

    all_chunks = []
    processed_names = []

    try:
        llm, embeddings = initialize_gemini()
    except ValueError as e:
        return f"Configuration Error:\n{str(e)}", "", []
    except Exception as e:
        logger.error(f"Failed to initialize Gemini: {e}")
        return f"Failed to connect to Google Gemini. Error: {str(e)}", "", []

    for file_obj in file_objs if isinstance(file_objs, list) else [file_objs]:
        try:
            file_path = str(file_obj.name) if hasattr(file_obj, "name") else str(file_obj)
            file_name = os.path.basename(file_path)

            if not file_path.lower().endswith(".pdf"):
                logger.warning(f"Skipping non-PDF file: {file_name}")
                continue

            logger.info(f"Loading PDF: {file_name}")
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            if not documents:
                continue

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            chunks = text_splitter.split_documents(documents)
            all_chunks.extend(chunks)
            processed_names.append(file_name)

        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")
            continue

    if not all_chunks:
        return "No valid content could be extracted from the uploaded file(s).", "", []

    try:
        logger.info("Building vector store...")
        vector_store = InMemoryVectorStore.from_documents(all_chunks, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 20})

        template = """You are an expert assistant specialized in document analysis.
Use ONLY the following context to answer the question accurately and thoroughly.
If the context doesn't contain enough information, say "I don't have enough information in the uploaded document to answer that question."

Be concise but comprehensive. Use bullet points for lists. Cite specific sections when possible.

Context:
{context}

Question: {question}

Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        loaded_doc_names = processed_names

        summary_prompt = "Based on the document content, provide a brief 2-3 sentence summary of what this document is about."
        try:
            doc_summary = qa_chain.invoke(summary_prompt)
        except Exception:
            doc_summary = "Summary generation failed."

        file_list = "\n".join(f"- {name}" for name in processed_names)
        status_msg = (
            f"Knowledge Base Built Successfully!\n\n"
            f"Documents Processed:\n{file_list}\n\n"
            f"Total Chunks: {len(all_chunks)}\n"
            f"Ready at: {datetime.now().strftime('%H:%M:%S')}"
        )

        suggestions = [
            "What is this document about?",
            "Summarize the key points.",
            "What are the main conclusions?",
        ]

        logger.info("Pipeline ready.")
        return status_msg, doc_summary, suggestions

    except Exception as e:
        logger.error(f"Failed to build vector store: {e}")
        return f"Error building knowledge base:\n{str(e)}", "", []

def answer_query(question, chat_history):
    global qa_chain

    if qa_chain is None:
        chat_history = chat_history or []
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": "Please upload and process a PDF document first before asking questions."})
        return chat_history, ""

    if not question or not question.strip():
        return chat_history or [], ""

    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": question})

    try:
        response = qa_chain.invoke(question)
        response = response.strip()

        if not response:
            response = "I wasn't able to generate a response. Please try rephrasing your question."

        chat_history.append({"role": "assistant", "content": response})

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        error_msg = f"Error generating response. Details: {str(e)}"
        chat_history.append({"role": "assistant", "content": error_msg})

    return chat_history, ""

def clear_chat():
    return [], ""

def use_suggestion(suggestion, chat_history):
    return answer_query(suggestion, chat_history)

def create_app():
    custom_css = load_css()
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.purple,
        secondary_hue=gr.themes.colors.pink,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    )

    with gr.Blocks(title="NexusRAG", theme=theme, css=custom_css) as app:
        gr.HTML("""
        <div id="hero-header">
            <h1>NexusRAG: Advanced Document Intelligence</h1>
            <p>Built by <b>Nipun Abhilash</b> • Upload PDF documents and ask questions</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["glass-panel"]):
                gr.HTML('<div class="section-title">Document Upload</div>')
                file_input = gr.File(
                    label="Upload PDF Files",
                    file_types=[".pdf"],
                    file_count="multiple",
                    elem_id="file-upload",
                )
                process_btn = gr.Button("Build Knowledge Base", variant="primary", elem_classes=["btn-primary"], size="lg")
                status_output = gr.Textbox(
                    label="System Status",
                    interactive=False,
                    lines=8,
                    elem_id="status-box",
                    placeholder="Upload a PDF and click 'Build Knowledge Base' to begin.",
                )

                with gr.Accordion("Document Summary", open=False):
                    summary_output = gr.Textbox(
                        label="Auto-Generated Summary",
                        interactive=False,
                        lines=4,
                        placeholder="Summary will appear after processing.",
                    )

                gr.HTML('<div class="section-title" style="margin-top: 1rem;">Quick Questions</div>')
                
                suggest_btn_1 = gr.Button("What is this document about?", elem_classes=["suggestion-btn"], size="sm")
                suggest_btn_2 = gr.Button("Summarize the key points", elem_classes=["suggestion-btn"], size="sm")
                suggest_btn_3 = gr.Button("What are the main conclusions?", elem_classes=["suggestion-btn"], size="sm")

            with gr.Column(scale=2, elem_classes=["glass-panel"]):
                gr.HTML('<div class="section-title">Chat Interface</div>')
                chatbot = gr.Chatbot(
                    label="Conversation",
                    elem_id="chatbot",
                    height=480,
                )

                with gr.Row():
                    query_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask about the document...",
                        lines=1,
                        scale=5,
                        elem_id="query-input",
                    )
                    submit_btn = gr.Button("Send", variant="primary", elem_classes=["btn-primary"], scale=1)

                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", variant="secondary", elem_classes=["btn-secondary"], size="sm")

        process_btn.click(
            fn=process_documents,
            inputs=[file_input],
            outputs=[status_output, summary_output, gr.State()],
        )

        submit_btn.click(
            fn=answer_query,
            inputs=[query_input, chatbot],
            outputs=[chatbot, query_input],
        )

        query_input.submit(
            fn=answer_query,
            inputs=[query_input, chatbot],
            outputs=[chatbot, query_input],
        )

        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, query_input],
        )

        for btn, question_text in [
            (suggest_btn_1, "What is this document about?"),
            (suggest_btn_2, "Summarize the key points of this document."),
            (suggest_btn_3, "What are the main conclusions or takeaways?"),
        ]:
            btn.click(
                fn=answer_query,
                inputs=[gr.State(question_text), chatbot],
                outputs=[chatbot, query_input],
            )

    return app

# Remove any conflicting Google Cloud credentials that might force OAuth
if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
    del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

if __name__ == "__main__":
    logger.info("Starting NexusRAG System...")
    app = create_app()
    app.launch(share=False, show_error=True)