# 🧠 NexusRAG: Advanced Document Intelligence by Nipun Abhilash

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-green)
![Gemini](https://img.shields.io/badge/Google%20Gemini-3.6%20Flash-orange)
![Gradio](https://img.shields.io/badge/UI-Gradio-red)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

An advanced, full-stack **Retrieval-Augmented Generation (RAG)** application built by **Nipun Abhilash**. This project allows users to upload complex PDF documents and instantly query them using the state-of-the-art **Google Gemini 3.6 Flash** reasoning model.

By leveraging a custom HTTP-level embedding engine (bypassing legacy SDK bugs) and a robust LangChain pipeline, the system parses, chunks, and vectorizes large PDFs with zero data loss. It features a modern, glassmorphic UI built with Gradio and custom CSS.

---

## ✨ Key Features

- **Advanced Document Processing:** Automatically parses and splits multi-page PDFs using LangChain's `RecursiveCharacterTextSplitter`.
- **Custom Embedding Engine:** Utilizes a custom `httpx`-based wrapper around the `models/gemini-embedding-2` endpoint, featuring built-in exponential backoff and rate-limit bypassing to prevent SSL EOF crashes.
- **Deep Reasoning AI:** Powered by Google's latest **Gemini 3.6 Flash** model, with a massive 8,192 token output limit to allow the AI to "think" via Chain-of-Thought (CoT) before producing highly detailed, comprehensive answers.
- **Dynamic Context Retrieval:** Implements an `InMemoryVectorStore` retriever with `k=20`, ensuring the AI has maximum context to answer complex questions that span across dozens of pages simultaneously.
- **Beautiful Glassmorphic UI:** A fully custom Gradio interface styled with native CSS for a premium, responsive user experience.
- **Auto-Summarization:** Instantly generates an abstract summary of the uploaded document immediately after processing.

---

## 🏗️ Architecture

1. **Ingestion:** `PyPDFLoader` extracts text from uploaded PDFs.
2. **Chunking:** Text is recursively split into 1000-character chunks with a 200-character overlap to preserve semantic context.
3. **Vectorization:** `CustomGeminiEmbeddings` vectorizes the chunks and stores them in memory.
4. **Retrieval:** User queries are embedded and compared against the vector store to fetch the top 20 most relevant chunks.
5. **Generation:** The context and query are piped through an LCEL (LangChain Expression Language) chain into `gemini-3.6-flash` for answer generation.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10 or higher
- A valid Google API Key (`generativelanguage.googleapis.com`)

### 2. Installation
Clone this repository and set up your virtual environment:

```bash
git clone https://github.com/nipun-abhilash/NexusRAG.git
cd NexusRAG
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

### 3. Configuration
Rename the provided `.env.example` file to `.env` and add your Google API Key:

```env
GOOGLE_API_KEY=your_actual_api_key_here
```

### 4. Running the App
Launch the Gradio server:

```bash
python qabot.py
```
Open your browser and navigate to `http://localhost:7860`.

---

## 👨‍💻 About the Author

**Nipun Abhilash**  
*B.Tech in Computer Science (Specialization in AI & Data Science) | IIIT Kottayam*

Passionate about building end-to-end AI applications, solving complex infrastructure bugs, and turning theoretical machine learning models into highly practical software tools. 

Feel free to connect with me on LinkedIn to discuss AI, Data Science, or Software Engineering!

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
