
# 🎥 VidSense AI — Chat With Any YouTube Video

**VidSense AI** is an end-to-end Retrieval-Augmented Generation (RAG) system that allows users to **ask natural language questions about any YouTube video** and receive **context-aware answers in real time**, directly inside the browser via a Chrome extension.

It combines **YouTube transcript ingestion, semantic search, conversational memory, web fallback, and LLM evaluation**, all built with production-grade engineering practices.

Demo Video Available -[**Click Here**](https://drive.google.com/file/d/1zWL_7J6eRFZOCl56Q3OHiUoSFndeo3cX/view?usp=sharing)

## 🚀 Key Features

### 🔹 Transcript-First RAG

* Automatically fetches YouTube transcripts
* Chunks and embeds content
* Retrieves only the most relevant segments per query

### 🔹 Intelligent Web Fallback

* If the transcript lacks sufficient information, the system **automatically augments answers using web search**
* Ensures higher answer completeness for opinion-based or contextual questions

### 🔹 Conversational Memory

* Maintains session-level memory across multiple questions
* Follow-up questions are answered with awareness of prior context

### 🔹 Chrome Extension UI

* Ask questions while watching YouTube
* Clean, scrollable chat interface
* Persistent session identity per user

### 🔹 LLM-Agnostic Design

* Fully powered by **open-source Hugging Face models**
* No vendor lock-in (Gemini / OpenAI optional)
* Easy to swap models at any stage

### 🔹 Evaluation & Observability

* Uses **RAGAS** for automatic evaluation:

  * Context relevance
  * Faithfulness
  * Answer relevancy
* Logs query source (transcript vs web) for monitoring



## 🧱 Architecture Diagram

```
┌─────────────────────────────┐
│        Chrome Extension     │
│  (YouTube Chat Interface)   │
└───────────────┬─────────────┘
                │
                ▼
┌─────────────────────────────┐
│        FastAPI Backend      │
│                             │
│  /ingest/youtube/{video_id} │
│  /ask                       │
└───────────────┬─────────────┘
                │
                ▼
┌─────────────────────────────┐
│   YouTube Transcript Loader │
│   + Text Chunking           │
└───────────────┬─────────────┘
                │
                ▼
┌─────────────────────────────┐
│   Embeddings (Hugging Face) │
│   + FAISS Vector Store      │
└───────────────┬─────────────┘
                │
                ▼
┌─────────────────────────────┐
│     Retrieval + Rewriting   │
│   (Memory-Aware Queries)    │
└───────────────┬─────────────┘
                │
                ▼
┌─────────────────────────────┐
│     Answer Generation LLM   │
│   (Transcript-First RAG)    │
└───────────────┬─────────────┘
                │
        ┌───────┴────────┐
        │                │
        ▼                ▼
┌───────────────┐  ┌────────────────┐
│ Transcript    │  │ Web Search     │
│ Answer        │  │ (Fallback)     │
└───────────────┘  └────────────────┘
```

## 🔍 Detailed RAG Flow

```
User Question
     │
     ▼
Session Memory (previous turns)
     │
     ▼
Query Rewriter (LLM)
     │
     ▼
FAISS Similarity Search
     │
     ▼
Top-K Transcript Chunks
     │
     ▼
Transcript Relevance Check
     │
     ├── Relevant ──► Answer Generator (LLM)
     │
     └── Insufficient ──► Web Search Tool
                              │
                              ▼
                    Web-Augmented Context
                              │
                              ▼
                    Answer Generator (LLM)
```



## 🧩 Tech Stack

### Backend

* **FastAPI** — API layer
* **LangChain** — RAG orchestration
* **FAISS** — Vector similarity search
* **YouTube Transcript API** — Transcript ingestion

### LLMs & Embeddings

* **Mistral-7B-Instruct (Hugging Face)** — Generation & rewriting
* **BGE / MixedBread embeddings** — Semantic retrieval
* Fully open-source, provider-agnostic

### Frontend

* **Chrome Extension (Manifest v3)**
* Vanilla JS + CSS
* Real-time chat UI

### Evaluation & Ops

* **RAGAS** — LLM evaluation
* Custom logging for observability
* Session-level analytics



## 📊 Evaluation Metrics (Sample)

```
Context Relevance : 1.00
Answer Relevancy  : 0.75
Faithfulness      : High (Transcript-grounded)
```

These metrics ensure the system answers are:

* Grounded in retrieved context
* Relevant to the question
* Not hallucinated



## 🛠 How It Works

1. User opens a YouTube video
2. Chrome extension extracts video ID
3. Backend ingests transcript (once)
4. User asks a question
5. Query is rewritten (memory-aware)
6. Relevant transcript chunks are retrieved
7. Answer is generated from transcript
8. If insufficient → web augmentation is applied
9. Response is returned with source attribution


## 🧪 Example Queries

* *“How did Beijing reduce air pollution?”*
  → Answered from transcript

* *“Is living in Delhi worth it considering pollution?”*
  → Transcript + Web augmentation

* *“What about electric vehicles?”*
  → Uses memory from prior questions



## 🧠 Design Decisions (Why This Matters)

* **Transcript-first** avoids hallucination
* **Web fallback** improves robustness
* **Memory-aware queries** enable natural conversations
* **Open-source models** prevent quota failures
* **Evaluation built-in** from day one



## 📌 Future Improvements

* Token streaming
* Source highlighting per answer
* Dockerized deployment
* Multi-video knowledge graphs

