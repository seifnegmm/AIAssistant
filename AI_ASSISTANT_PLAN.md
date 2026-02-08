# AI Assistant — Technical Plan v3.0

> **Version**: 3.0 — Docker Compose · MongoDB · ChromaDB · Simli Avatar
> **Last Updated**: 2026-02-08
> **Status**: Approved for implementation

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Technology Stack](#3-technology-stack)
4. [Infrastructure — Docker Compose](#4-infrastructure--docker-compose)
5. [Backend — FastAPI Service](#5-backend--fastapi-service)
6. [Frontend — Next.js Application](#6-frontend--nextjs-application)
7. [LLM & Agent Layer — Gemini + LangGraph](#7-llm--agent-layer--gemini--langgraph)
8. [Voice Pipeline — STT & TTS](#8-voice-pipeline--stt--tts)
9. [Avatar System — Simli](#9-avatar-system--simli)
10. [Memory & Learning System](#10-memory--learning-system)
11. [Data Flow Diagrams](#11-data-flow-diagrams)
12. [API Contracts](#12-api-contracts)
13. [Environment & Configuration](#13-environment--configuration)
14. [Dependencies](#14-dependencies)
15. [Project Structure](#15-project-structure)
16. [Implementation Roadmap](#16-implementation-roadmap)
17. [Architecture Decision Records (ADRs)](#17-architecture-decision-records-adrs)
18. [Cost Estimates](#18-cost-estimates)
19. [Future Extensions](#19-future-extensions)

---

## 1. Executive Summary

### What Are We Building?

A conversational AI assistant that:

- **Talks** — real-time speech input/output via Google Cloud STT/TTS
- **Thinks** — powered by Google Gemini 2.5 Flash through LangChain/LangGraph
- **Remembers** — conversation history in MongoDB, semantic long-term memory in ChromaDB
- **Learns** — extracts and recalls user preferences without fine-tuning
- **Shows a face** — photorealistic human avatar via Simli with real-time lip-sync

### Core Constraints

| Constraint | Decision |
|---|---|
| LLM | Google Gemini exclusively (via `langchain-google-genai`) |
| Orchestration | LangChain + LangGraph |
| Auth | API key only (no service accounts) |
| Deployment | Local development only (Docker Compose) |
| Avatar | Must look as human as possible (Simli) |
| Learning | No model fine-tuning — memory/retrieval only |

---

## 2. Architecture Overview

### High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Docker Compose                          │
│                                                              │
│  ┌──────────────────┐       ┌──────────────────────────┐    │
│  │   Next.js App    │       │     FastAPI Backend       │    │
│  │   (Port 3000)    │◄─────►│     (Port 8000)          │    │
│  │                  │  WS   │                          │    │
│  │  ┌────────────┐  │       │  ┌────────────────────┐  │    │
│  │  │ Simli SDK  │  │       │  │ LangGraph Agent    │  │    │
│  │  │ (WebRTC)   │  │       │  │ (Gemini 2.5 Flash) │  │    │
│  │  └────────────┘  │       │  └────────────────────┘  │    │
│  │                  │       │  ┌────────────────────┐  │    │
│  │  ┌────────────┐  │       │  │ Voice Pipeline     │  │    │
│  │  │ Zustand    │  │       │  │ (STT ↔ TTS)       │  │    │
│  │  │ State Mgmt │  │       │  └────────────────────┘  │    │
│  │  └────────────┘  │       │  ┌────────────────────┐  │    │
│  │                  │       │  │ Memory Manager     │  │    │
│  └──────────────────┘       │  │ (Mongo + Chroma)   │  │    │
│                              │  └────────────────────┘  │    │
│                              └──────────────────────────┘    │
│                                                              │
│  ┌──────────────────┐       ┌──────────────────────────┐    │
│  │    MongoDB        │       │      ChromaDB             │    │
│  │    (Port 27017)   │       │      (Port 8100)          │    │
│  │                  │       │                          │    │
│  │  • chat_history  │       │  • assistant_memory      │    │
│  │  • user_prefs    │       │  • user_preferences      │    │
│  │  • sessions      │       │                          │    │
│  └──────────────────┘       └──────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                          │
                    External APIs
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   Google Gemini   Google Cloud     Simli API
   (LLM API)      STT / TTS       (Avatar)
```

### Service Communication

| From | To | Protocol | Purpose |
|---|---|---|---|
| Frontend | Backend | WebSocket (`/ws/chat`) | Streaming chat + audio |
| Frontend | Backend | REST (`/api/*`) | Session management, preferences, health |
| Frontend | Simli Cloud | WebRTC | Real-time avatar video/audio |
| Backend | Google Gemini | HTTPS (gRPC) | LLM inference |
| Backend | Google STT | gRPC Streaming | Speech → text |
| Backend | Google TTS | gRPC | Text → speech (PCM16) |
| Backend | MongoDB | TCP (27017) | Chat history, user data |
| Backend | ChromaDB | HTTP (8100) | Vector similarity search |

---

## 3. Technology Stack

### Backend

| Component | Technology | Version | Purpose |
|---|---|---|---|
| Framework | FastAPI | ≥0.115 | Async REST + WebSocket server |
| Server | Uvicorn | ≥0.32 | ASGI server with WebSocket support |
| LLM | Google Gemini 2.5 Flash | Latest | Primary reasoning model |
| LLM SDK | `langchain-google-genai` | ≥2.0 | LangChain ↔ Gemini bridge |
| Agent | LangGraph | ≥0.2 | ReAct agent with tool calling |
| Orchestration | LangChain | ≥0.3 | Chains, memory, retrieval |
| Chat History | `langchain-mongodb` | ≥0.3 | MongoDB chat message store |
| Vector Store | `langchain-chroma` | ≥0.2 | ChromaDB LangChain integration |
| Embeddings | Google `text-embedding-004` | Latest | 768-dim text embeddings |
| STT | `google-cloud-speech` (V2) | ≥2.29 | Streaming speech-to-text |
| TTS | `google-cloud-texttospeech` | ≥2.24 | Text-to-speech (Chirp3 voices) |
| Validation | Pydantic | ≥2.0 | Data models and settings |
| Config | pyyaml | ≥6.0 | Assistant personality config |
| Database Driver | `motor` | ≥3.6 | Async MongoDB driver |

### Frontend

| Component | Technology | Version | Purpose |
|---|---|---|---|
| Framework | Next.js | ^14.2 | React meta-framework (App Router) |
| UI Library | React | ^18.3 | Component model |
| Avatar | `simli-client` | Latest | Simli WebRTC avatar SDK |
| State | Zustand | ^5.0 | Lightweight global state |
| Styling | Tailwind CSS | ^3.4 | Utility-first CSS |
| Audio | Web Audio API | Browser | Mic capture + audio processing |
| Type Safety | TypeScript | ^5.5 | Static typing |

### Infrastructure

| Component | Technology | Version | Purpose |
|---|---|---|---|
| Orchestration | Docker Compose | v2 | Multi-container management |
| Database | MongoDB | 7.0 | Chat history, user preferences |
| Vector DB | ChromaDB | Latest | Semantic memory, preference embeddings |
| Container Runtime | Docker | ≥24 | Container engine |

---

## 4. Infrastructure — Docker Compose

### `docker-compose.yml`

```yaml
version: "3.9"

services:
  # ──────────────────────────────────────
  # MongoDB — Chat History & User Data
  # ──────────────────────────────────────
  mongodb:
    image: mongo:7
    container_name: ai-assistant-mongo
    restart: unless-stopped
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db
    environment:
      MONGO_INITDB_DATABASE: ai_assistant
    healthcheck:
      test: ["CMD", "mongosh", "--eval", "db.adminCommand('ping')"]
      interval: 10s
      timeout: 5s
      retries: 5

  # ──────────────────────────────────────
  # ChromaDB — Vector / Semantic Memory
  # ──────────────────────────────────────
  chromadb:
    image: chromadb/chroma:latest
    container_name: ai-assistant-chroma
    restart: unless-stopped
    ports:
      - "8100:8000"
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      - ANONYMIZED_TELEMETRY=FALSE
      - IS_PERSISTENT=TRUE
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v2/heartbeat"]
      interval: 10s
      timeout: 5s
      retries: 5

  # ──────────────────────────────────────
  # FastAPI Backend
  # ──────────────────────────────────────
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: ai-assistant-backend
    restart: unless-stopped
    ports:
      - "8000:8000"
    env_file:
      - .env
    environment:
      - MONGODB_URI=mongodb://mongodb:27017
      - CHROMADB_HOST=chromadb
      - CHROMADB_PORT=8000
    depends_on:
      mongodb:
        condition: service_healthy
      chromadb:
        condition: service_healthy
    volumes:
      - ./backend:/app
      - /app/.venv  # exclude venv from mount
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

  # ──────────────────────────────────────
  # Next.js Frontend
  # ──────────────────────────────────────
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: ai-assistant-frontend
    restart: unless-stopped
    ports:
      - "3000:3000"
    env_file:
      - .env.local
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
      - NEXT_PUBLIC_WS_URL=ws://localhost:8000
    depends_on:
      - backend
    volumes:
      - ./frontend:/app
      - /app/node_modules  # exclude node_modules from mount
      - /app/.next         # exclude build cache from mount
    command: npm run dev

volumes:
  mongo_data:
    driver: local
  chroma_data:
    driver: local
```

### Port Allocation

| Service | Host Port | Container Port | Notes |
|---|---|---|---|
| Frontend (Next.js) | 3000 | 3000 | Browser access |
| Backend (FastAPI) | 8000 | 8000 | API + WebSocket |
| MongoDB | 27017 | 27017 | Database |
| ChromaDB | 8100 | 8000 | Vector store (host 8100 to avoid conflict with backend) |

### Backend Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# System dependencies for gRPC and audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Frontend Dockerfile

```dockerfile
FROM node:20-slim

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json* ./
RUN npm ci

COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev"]
```

### Startup Sequence

```bash
# First time
docker compose up --build

# Subsequent starts
docker compose up

# Rebuild single service
docker compose up --build backend

# View logs
docker compose logs -f backend

# Stop everything
docker compose down

# Stop and remove volumes (DELETES DATA)
docker compose down -v
```

---

## 5. Backend — FastAPI Service

### Module Structure

```
backend/
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app, CORS, lifespan
│   ├── config.py                # Pydantic Settings (env vars)
│   ├── dependencies.py          # Dependency injection
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── router.py            # Mounts all sub-routers
│   │   ├── chat.py              # WebSocket /ws/chat endpoint
│   │   ├── sessions.py          # REST session CRUD
│   │   ├── preferences.py       # REST preference management
│   │   └── health.py            # Health + readiness checks
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py             # LangGraph agent definition
│   │   ├── state.py             # Agent state TypedDict
│   │   ├── nodes.py             # Agent graph nodes
│   │   ├── tools.py             # Custom tools for the agent
│   │   └── prompts.py           # System prompts & persona
│   │
│   ├── voice/
│   │   ├── __init__.py
│   │   ├── stt.py               # Google Cloud STT streaming
│   │   ├── tts.py               # Google Cloud TTS synthesis
│   │   └── audio_utils.py       # PCM16 conversion utilities
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── manager.py           # MemoryManager orchestrator
│   │   ├── chat_history.py      # MongoDB chat history wrapper
│   │   ├── vector_store.py      # ChromaDB vector store wrapper
│   │   └── preference_learner.py # Preference extraction & recall
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── messages.py          # WebSocket message schemas
│   │   ├── sessions.py          # Session data models
│   │   └── preferences.py       # Preference data models
│   │
│   └── personality/
│       ├── __init__.py
│       ├── loader.py            # YAML personality loader
│       └── default.yaml         # Default personality config
│
└── tests/
    ├── __init__.py
    ├── conftest.py              # Fixtures (test DB, mock LLM)
    ├── test_agent.py
    ├── test_memory.py
    ├── test_voice.py
    └── test_api.py
```

### Key Backend Components

#### 5.1 Application Entry — `app/main.py`

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api.router import api_router
from app.dependencies import init_services, shutdown_services

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and teardown shared services."""
    await init_services()
    yield
    await shutdown_services()

app = FastAPI(
    title="AI Assistant",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
```

#### 5.2 Configuration — `app/config.py`

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Google
    google_api_key: str
    
    # MongoDB
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "ai_assistant"
    
    # ChromaDB
    chromadb_host: str = "localhost"
    chromadb_port: int = 8100
    
    # Simli
    simli_api_key: str = ""
    simli_face_id: str = ""
    
    # App
    environment: str = "development"
    log_level: str = "debug"
    frontend_url: str = "http://localhost:3000"
    
    # LLM
    gemini_model: str = "gemini-2.5-flash"
    gemini_temperature: float = 0.7
    embedding_model: str = "models/text-embedding-004"
    
    # Voice
    tts_voice: str = "en-US-Chirp3-HD-Leda"
    tts_language_code: str = "en-US"
    stt_language_code: str = "en-US"
    audio_sample_rate: int = 16000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

#### 5.3 WebSocket Chat — `app/api/chat.py`

The primary interface. Handles text + audio streaming in a single connection.

```python
from fastapi import WebSocket, WebSocketDisconnect, Depends
from app.agent.graph import get_agent
from app.memory.manager import MemoryManager
from app.voice.stt import StreamingSTT
from app.voice.tts import TextToSpeech
from app.models.messages import WSMessage, WSMessageType

async def websocket_chat(
    websocket: WebSocket,
    session_id: str,
    memory: MemoryManager = Depends(),
    agent = Depends(get_agent),
):
    await websocket.accept()
    stt = StreamingSTT()
    tts = TextToSpeech()
    
    try:
        while True:
            raw = await websocket.receive_json()
            msg = WSMessage(**raw)
            
            match msg.type:
                case WSMessageType.TEXT:
                    # Text chat flow
                    response = await agent.ainvoke(
                        {"messages": [("user", msg.content)]},
                        config={"configurable": {"session_id": session_id}},
                    )
                    reply = response["messages"][-1].content
                    
                    # Send text reply
                    await websocket.send_json({
                        "type": "text",
                        "content": reply,
                    })
                    
                    # Generate TTS audio for avatar
                    audio_bytes = await tts.synthesize(reply)
                    await websocket.send_bytes(audio_bytes)
                
                case WSMessageType.AUDIO:
                    # Voice chat flow
                    transcript = await stt.transcribe(msg.audio_data)
                    
                    # Process through agent
                    response = await agent.ainvoke(
                        {"messages": [("user", transcript)]},
                        config={"configurable": {"session_id": session_id}},
                    )
                    reply = response["messages"][-1].content
                    
                    # Send transcript + text reply
                    await websocket.send_json({
                        "type": "transcript",
                        "content": transcript,
                    })
                    await websocket.send_json({
                        "type": "text",
                        "content": reply,
                    })
                    
                    # Generate TTS audio
                    audio_bytes = await tts.synthesize(reply)
                    await websocket.send_bytes(audio_bytes)
                    
    except WebSocketDisconnect:
        await memory.save_session(session_id)
```

#### 5.4 Memory Manager — `app/memory/manager.py`

Orchestrates MongoDB (chat history) and ChromaDB (semantic memory).

```python
import chromadb
from langchain_mongodb import MongoDBChatMessageHistory
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings

class MemoryManager:
    """Unified interface for all memory operations."""
    
    def __init__(self):
        # Embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key,
        )
        
        # ChromaDB client (HTTP mode)
        self.chroma_client = chromadb.HttpClient(
            host=settings.chromadb_host,
            port=settings.chromadb_port,
        )
        
        # LangChain vector store wrapper
        self.vector_store = Chroma(
            collection_name="assistant_memory",
            embedding_function=self.embeddings,
            client=self.chroma_client,
        )
        
        # Preference vector store (separate collection)
        self.preference_store = Chroma(
            collection_name="user_preferences",
            embedding_function=self.embeddings,
            client=self.chroma_client,
        )
    
    def get_chat_history(self, session_id: str) -> MongoDBChatMessageHistory:
        """Get MongoDB-backed chat history for a session."""
        return MongoDBChatMessageHistory(
            connection_string=settings.mongodb_uri,
            session_id=session_id,
            database_name=settings.mongodb_database,
            collection_name="chat_history",
            create_index=True,
        )
    
    async def store_memory(self, text: str, metadata: dict) -> str:
        """Store a piece of information in long-term semantic memory."""
        ids = self.vector_store.add_texts(
            texts=[text],
            metadatas=[metadata],
        )
        return ids[0]
    
    async def recall(self, query: str, k: int = 5) -> list[dict]:
        """Retrieve relevant memories by semantic similarity."""
        docs = self.vector_store.similarity_search(query, k=k)
        return [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in docs
        ]
    
    async def store_preference(self, preference: str, metadata: dict) -> str:
        """Store a learned user preference."""
        ids = self.preference_store.add_texts(
            texts=[preference],
            metadatas=[{**metadata, "type": "preference"}],
        )
        return ids[0]
    
    async def recall_preferences(self, context: str, k: int = 3) -> list[str]:
        """Retrieve relevant user preferences for a context."""
        docs = self.preference_store.similarity_search(context, k=k)
        return [doc.page_content for doc in docs]
    
    async def save_session(self, session_id: str):
        """Persist any pending session data."""
        # MongoDB chat history auto-persists on each message
        # This hook is for any cleanup or summary generation
        pass
```

#### 5.5 Preference Learner — `app/memory/preference_learner.py`

Uses Gemini to extract preferences from conversations, stores in ChromaDB.

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from app.config import settings

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Analyze the following conversation and extract any user preferences.
Look for:
- Stated likes/dislikes ("I prefer...", "I don't like...", "I love...")
- Communication style preferences (formal/casual, brief/detailed)
- Topic interests or expertise areas
- Schedule or routine patterns
- Any personal information voluntarily shared

Return a JSON array of preferences. Each preference should have:
- "text": the preference as a clear statement
- "category": one of ["communication", "interest", "routine", "personal", "dislike"]
- "confidence": 0.0-1.0 how confident you are this is a real preference

If no preferences are found, return an empty array: []"""),
    ("human", "{conversation}"),
])

class PreferenceLearner:
    def __init__(self, memory_manager):
        self.memory = memory_manager
        self.llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0.1,  # Low temp for extraction accuracy
        )
        self.chain = EXTRACTION_PROMPT | self.llm
    
    async def analyze_conversation(self, messages: list[dict]) -> list[dict]:
        """Extract preferences from recent conversation messages."""
        conversation_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )
        result = await self.chain.ainvoke({"conversation": conversation_text})
        # Parse JSON from LLM response
        preferences = self._parse_preferences(result.content)
        
        # Store high-confidence preferences
        stored = []
        for pref in preferences:
            if pref["confidence"] >= 0.7:
                await self.memory.store_preference(
                    preference=pref["text"],
                    metadata={
                        "category": pref["category"],
                        "confidence": pref["confidence"],
                    },
                )
                stored.append(pref)
        
        return stored
    
    def _parse_preferences(self, response: str) -> list[dict]:
        """Parse LLM response into structured preferences."""
        import json
        try:
            # Try to extract JSON from response
            start = response.index("[")
            end = response.rindex("]") + 1
            return json.loads(response[start:end])
        except (ValueError, json.JSONDecodeError):
            return []
```

---

## 6. Frontend — Next.js Application

### Module Structure

```
frontend/
├── Dockerfile
├── package.json
├── next.config.js
├── tailwind.config.ts
├── tsconfig.json
├── .env.local.example
│
├── public/
│   └── favicon.ico
│
└── src/
    ├── app/
    │   ├── layout.tsx           # Root layout (providers, fonts)
    │   ├── page.tsx             # Main assistant page
    │   └── globals.css          # Tailwind + custom styles
    │
    ├── components/
    │   ├── AssistantLayout.tsx   # Main 2-panel layout
    │   ├── ChatPanel.tsx         # Message list + input
    │   ├── MessageBubble.tsx     # Individual message display
    │   ├── AvatarPanel.tsx       # Simli avatar container
    │   ├── SimliAvatar.tsx       # Simli WebRTC integration
    │   ├── VoiceButton.tsx       # Push-to-talk / toggle mic
    │   ├── StatusIndicator.tsx   # Connection & processing status
    │   └── SettingsDrawer.tsx    # Preferences & voice settings
    │
    ├── hooks/
    │   ├── useWebSocket.ts      # WebSocket connection manager
    │   ├── useAudioCapture.ts   # Microphone access + PCM16
    │   ├── useSimli.ts          # Simli SDK lifecycle hook
    │   └── useChat.ts           # Chat state orchestrator
    │
    ├── stores/
    │   ├── chatStore.ts         # Zustand: messages, sessions
    │   ├── voiceStore.ts        # Zustand: mic state, STT status
    │   └── settingsStore.ts     # Zustand: user preferences
    │
    ├── lib/
    │   ├── websocket.ts         # WebSocket client singleton
    │   ├── audio.ts             # Audio encoding/decoding utils
    │   └── api.ts               # REST API client (fetch wrapper)
    │
    └── types/
        ├── messages.ts          # WebSocket message types
        ├── chat.ts              # Chat & session types
        └── avatar.ts            # Simli-related types
```

### Key Frontend Components

#### 6.1 Simli Avatar — `components/SimliAvatar.tsx`

```tsx
"use client";

import { useEffect, useRef, useCallback } from "react";
import { SimliClient } from "simli-client";

interface SimliAvatarProps {
  apiKey: string;
  faceId: string;
  audioData: Uint8Array | null;  // PCM16 audio from TTS
  isActive: boolean;
}

export function SimliAvatar({ apiKey, faceId, audioData, isActive }: SimliAvatarProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const clientRef = useRef<SimliClient | null>(null);

  // Initialize Simli client
  useEffect(() => {
    if (!videoRef.current || !audioRef.current) return;
    
    const client = new SimliClient();
    client.Initialize({
      apiKey,
      faceID: faceId,
      handleSilence: true,
      maxSessionLength: 3600,
      maxIdleTime: 600,
      videoRef: videoRef.current,
      audioRef: audioRef.current,
    });
    
    clientRef.current = client;
    
    return () => {
      client.close();
      clientRef.current = null;
    };
  }, [apiKey, faceId]);

  // Start/stop session
  useEffect(() => {
    if (!clientRef.current) return;
    if (isActive) {
      clientRef.current.start();
    }
  }, [isActive]);

  // Send audio to Simli when TTS audio arrives
  useEffect(() => {
    if (!audioData || !clientRef.current) return;
    clientRef.current.sendAudioData(audioData);
  }, [audioData]);

  return (
    <div className="relative w-full aspect-square rounded-2xl overflow-hidden bg-gray-900">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        className="w-full h-full object-cover"
      />
      <audio ref={audioRef} autoPlay />
    </div>
  );
}
```

#### 6.2 WebSocket Hook — `hooks/useWebSocket.ts`

```tsx
import { useEffect, useRef, useCallback, useState } from "react";

type WSMessage = {
  type: "text" | "transcript" | "error" | "status";
  content: string;
};

interface UseWebSocketReturn {
  sendText: (text: string) => void;
  sendAudio: (audioData: ArrayBuffer) => void;
  lastMessage: WSMessage | null;
  lastAudioData: Uint8Array | null;
  isConnected: boolean;
}

export function useWebSocket(sessionId: string): UseWebSocketReturn {
  const wsRef = useRef<WebSocket | null>(null);
  const [lastMessage, setLastMessage] = useState<WSMessage | null>(null);
  const [lastAudioData, setLastAudioData] = useState<Uint8Array | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const wsUrl = `${process.env.NEXT_PUBLIC_WS_URL}/ws/chat?session_id=${sessionId}`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => setIsConnected(true);
    ws.onclose = () => setIsConnected(false);
    
    ws.onmessage = async (event) => {
      if (event.data instanceof Blob) {
        // Binary = TTS audio (PCM16)
        const buffer = await event.data.arrayBuffer();
        setLastAudioData(new Uint8Array(buffer));
      } else {
        // JSON = text message
        const msg: WSMessage = JSON.parse(event.data);
        setLastMessage(msg);
      }
    };

    wsRef.current = ws;
    return () => ws.close();
  }, [sessionId]);

  const sendText = useCallback((text: string) => {
    wsRef.current?.send(JSON.stringify({ type: "text", content: text }));
  }, []);

  const sendAudio = useCallback((audioData: ArrayBuffer) => {
    wsRef.current?.send(JSON.stringify({
      type: "audio",
      audio_data: btoa(String.fromCharCode(...new Uint8Array(audioData))),
    }));
  }, []);

  return { sendText, sendAudio, lastMessage, lastAudioData, isConnected };
}
```

#### 6.3 Chat Store — `stores/chatStore.ts`

```tsx
import { create } from "zustand";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
  audioData?: Uint8Array;
}

interface ChatState {
  messages: Message[];
  sessionId: string;
  isProcessing: boolean;
  addMessage: (msg: Omit<Message, "id" | "timestamp">) => void;
  setProcessing: (val: boolean) => void;
  setSessionId: (id: string) => void;
  clearMessages: () => void;
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  sessionId: crypto.randomUUID(),
  isProcessing: false,
  addMessage: (msg) =>
    set((state) => ({
      messages: [
        ...state.messages,
        { ...msg, id: crypto.randomUUID(), timestamp: Date.now() },
      ],
    })),
  setProcessing: (val) => set({ isProcessing: val }),
  setSessionId: (id) => set({ sessionId: id }),
  clearMessages: () => set({ messages: [] }),
}));
```

---

## 7. LLM & Agent Layer — Gemini + LangGraph

### Agent Architecture

We use a **LangGraph ReAct agent** — a state machine that loops between:
1. **Model node**: Gemini decides whether to respond directly or call a tool
2. **Tool node**: Executes the requested tool and returns the result
3. **END**: Model produces a final response

```
           ┌──────────┐
           │  START    │
           └────┬─────┘
                │
                ▼
        ┌───────────────┐
   ┌───►│  Agent (LLM)  │◄──────┐
   │    └───────┬───────┘       │
   │            │               │
   │     Has tool calls?        │
   │     ┌──────┴──────┐       │
   │     │             │       │
   │    YES            NO      │
   │     │             │       │
   │     ▼             ▼       │
   │  ┌────────┐  ┌────────┐  │
   │  │ Tools  │  │  END   │  │
   │  └───┬────┘  └────────┘  │
   │      │                    │
   │      └────────────────────┘
   │        (tool results)
   └─────────────────────────────
```

### Agent Definition — `app/agent/graph.py`

```python
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from app.agent.tools import get_tools
from app.agent.prompts import SYSTEM_PROMPT
from app.memory.manager import MemoryManager
from app.config import settings

def build_agent(memory: MemoryManager):
    """Build the LangGraph ReAct agent with Gemini and tools."""
    
    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=settings.gemini_temperature,
        convert_system_message_to_human=False,
    )
    
    tools = get_tools(memory)
    
    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=SYSTEM_PROMPT,
    )
    
    # Wrap with message history (MongoDB-backed)
    agent_with_history = RunnableWithMessageHistory(
        agent,
        lambda session_id: memory.get_chat_history(session_id),
        input_messages_key="messages",
        history_messages_key="chat_history",
    )
    
    return agent_with_history
```

### Agent Tools — `app/agent/tools.py`

```python
from langchain_core.tools import tool
from app.memory.manager import MemoryManager

def get_tools(memory: MemoryManager) -> list:
    """Define the tools available to the agent."""
    
    @tool
    async def recall_memory(query: str) -> str:
        """Search long-term memory for relevant past information.
        Use this when the user references something from a past conversation
        or when you need context about the user's history."""
        results = await memory.recall(query, k=5)
        if not results:
            return "No relevant memories found."
        return "\n".join(
            f"- {r['content']} (from: {r['metadata'].get('source', 'unknown')})"
            for r in results
        )
    
    @tool
    async def recall_preferences(context: str) -> str:
        """Retrieve user preferences relevant to the current context.
        Use this to personalize responses based on known user preferences."""
        prefs = await memory.recall_preferences(context, k=5)
        if not prefs:
            return "No relevant preferences found."
        return "Known user preferences:\n" + "\n".join(f"- {p}" for p in prefs)
    
    @tool
    async def save_memory(information: str, source: str = "conversation") -> str:
        """Save an important piece of information to long-term memory.
        Use this for facts, events, or details the user might want recalled later."""
        doc_id = await memory.store_memory(
            text=information,
            metadata={"source": source},
        )
        return f"Memory saved (id: {doc_id})"
    
    @tool
    def get_current_time() -> str:
        """Get the current date and time."""
        from datetime import datetime
        return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    
    return [recall_memory, recall_preferences, save_memory, get_current_time]
```

### System Prompt — `app/agent/prompts.py`

```python
SYSTEM_PROMPT = """You are a helpful AI assistant with a warm, friendly personality.

## Core Behavior
- Be conversational and natural, like talking to a knowledgeable friend
- Adapt your communication style to the user's preferences when known
- Be concise by default, but provide detail when asked or when the topic requires it
- Use the available tools proactively to recall memories and preferences

## Memory & Context
- You have access to long-term memory through the `recall_memory` tool
- You can remember user preferences via the `recall_preferences` tool
- When a user mentions something they told you before, use recall_memory to find it
- Proactively use recall_preferences at the start of conversations to personalize

## Voice Interaction
- When responding to voice input, keep answers concise and conversational
- Use natural speech patterns — contractions, casual phrasing
- Avoid overly long lists or complex formatting in voice mode

## Tool Usage Guidelines
- Use `recall_memory` when the user says "remember when...", "you know...", or references past conversations
- Use `recall_preferences` at conversation start and when making recommendations
- Use `save_memory` when the user shares important facts, events, or preferences
- Use `get_current_time` when time/date is relevant to the conversation

## Personality
- Thoughtful but not overthinking
- Supportive but honest
- Curious and engaged with the user's interests
"""
```

---

## 8. Voice Pipeline — STT & TTS

### Architecture

```
┌───────────┐    WebSocket     ┌───────────┐    gRPC        ┌───────────┐
│  Browser   │ ──────────────► │  Backend   │ ─────────────► │ Google    │
│  (Mic)     │   PCM16 audio   │  (STT)     │   streaming    │ STT V2   │
└───────────┘                  └───────────┘                 └─────┬─────┘
                                                                    │
                                                              transcript
                                                                    │
                                                                    ▼
┌───────────┐    WebSocket     ┌───────────┐    gRPC        ┌───────────┐
│  Browser   │ ◄────────────── │  Backend   │ ◄───────────── │ Google    │
│  (Simli)   │   PCM16 audio   │  (TTS)     │   audio bytes  │ TTS      │
└───────────┘                  └───────────┘                 └───────────┘
```

### Audio Format Standard

All audio in the pipeline uses a single format for simplicity:

| Parameter | Value |
|---|---|
| Encoding | LINEAR16 (PCM16) |
| Sample Rate | 16,000 Hz |
| Channels | 1 (mono) |
| Bit Depth | 16-bit signed integers |
| Byte Order | Little-endian |

This format is natively supported by Google STT, Google TTS, and Simli.

### STT Service — `app/voice/stt.py`

```python
from google.cloud import speech_v2
from app.config import settings

class StreamingSTT:
    """Google Cloud Speech-to-Text V2 streaming transcription."""
    
    def __init__(self):
        self.client = speech_v2.SpeechAsyncClient()
        self.config = speech_v2.RecognitionConfig(
            auto_decoding_config=speech_v2.AutoDetectDecodingConfig(),
            language_codes=[settings.stt_language_code],
            model="long",
        )
    
    async def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe a complete audio segment."""
        request = speech_v2.RecognizeRequest(
            recognizer=f"projects/-/locations/global/recognizers/_",
            config=self.config,
            content=audio_bytes,
        )
        response = await self.client.recognize(request=request)
        
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript
        
        return transcript.strip()
    
    async def stream_transcribe(self, audio_stream):
        """Stream audio chunks and yield partial transcripts."""
        streaming_config = speech_v2.StreamingRecognitionConfig(
            config=self.config,
            streaming_features=speech_v2.StreamingRecognitionFeatures(
                interim_results=True,
            ),
        )
        
        async def request_generator():
            # First request: config only
            yield speech_v2.StreamingRecognizeRequest(
                recognizer=f"projects/-/locations/global/recognizers/_",
                streaming_config=streaming_config,
            )
            # Subsequent requests: audio chunks
            async for chunk in audio_stream:
                yield speech_v2.StreamingRecognizeRequest(audio=chunk)
        
        responses = await self.client.streaming_recognize(
            requests=request_generator()
        )
        
        async for response in responses:
            for result in response.results:
                if result.is_final:
                    yield result.alternatives[0].transcript
```

### TTS Service — `app/voice/tts.py`

```python
from google.cloud import texttospeech
from app.config import settings

class TextToSpeech:
    """Google Cloud Text-to-Speech synthesis."""
    
    def __init__(self):
        self.client = texttospeech.TextToSpeechAsyncClient()
        self.voice = texttospeech.VoiceSelectionParams(
            language_code=settings.tts_language_code,
            name=settings.tts_voice,
        )
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=settings.audio_sample_rate,
        )
    
    async def synthesize(self, text: str) -> bytes:
        """Convert text to PCM16 audio bytes."""
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        response = await self.client.synthesize_speech(
            input=synthesis_input,
            voice=self.voice,
            audio_config=self.audio_config,
        )
        
        # response.audio_content is raw PCM16 bytes (with WAV header)
        # Strip 44-byte WAV header for raw PCM16 (Simli expects raw)
        return response.audio_content[44:]
    
    async def synthesize_chunked(self, text: str, max_chars: int = 200):
        """Split long text and synthesize in chunks for lower latency."""
        sentences = self._split_sentences(text)
        
        for sentence in sentences:
            audio = await self.synthesize(sentence)
            yield audio
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences for chunked TTS."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
```

---

## 9. Avatar System — Simli

### Why Simli?

| Requirement | Simli Capability |
|---|---|
| "Look as human as possible" | Photorealistic AI-generated faces |
| Real-time lip sync | Audio-driven lip sync via WebRTC |
| Low latency | WebRTC peer connection (~100ms) |
| Easy integration | JavaScript SDK (`simli-client`) |
| No GPU required locally | All rendering happens server-side (Simli Cloud) |

### How Simli Works

1. **Initialize**: Frontend creates a `SimliClient` with API key and face ID
2. **Start Session**: SDK opens a WebRTC connection to Simli servers
3. **Send Audio**: Backend sends TTS audio (PCM16 16KHz) → Frontend forwards to Simli
4. **Receive Video**: Simli returns real-time video frames via WebRTC
5. **Display**: `<video>` element shows the lip-syncing avatar

### Integration Flow

```
User speaks → Mic capture → WebSocket → Backend
                                           │
                                    STT (Google) → transcript
                                           │
                                    LLM (Gemini) → reply text
                                           │
                                    TTS (Google) → PCM16 audio
                                           │
                                    WebSocket → Frontend
                                           │
                                    simli-client.sendAudioData(pcm16)
                                           │
                                    WebRTC ↔ Simli Cloud
                                           │
                                    <video> lip-syncing avatar
```

### Simli Configuration

| Parameter | Value | Description |
|---|---|---|
| `apiKey` | From Simli dashboard | Authentication |
| `faceID` | Selected face hash | Which avatar face to use |
| `handleSilence` | `true` | Avatar shows idle animation when no audio |
| `maxSessionLength` | `3600` | Max 1 hour per session |
| `maxIdleTime` | `600` | Close after 10 min idle |

### Face Selection

Simli provides a library of photorealistic faces. The face ID is selected from the Simli dashboard (https://app.simli.com). Choose a face that:
- Looks professional and approachable
- Has natural idle animations
- Supports the desired gender/appearance
- Works well with the selected TTS voice

---

## 10. Memory & Learning System

### Memory Architecture

```
                    ┌──────────────────────────┐
                    │      Memory Manager       │
                    │   (Orchestrates both)      │
                    └─────────┬────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼                               ▼
   ┌─────────────────┐            ┌─────────────────┐
   │    MongoDB       │            │    ChromaDB      │
   │                  │            │                  │
   │  ┌────────────┐  │            │  ┌────────────┐  │
   │  │ Chat       │  │            │  │ assistant_ │  │
   │  │ History    │  │            │  │ memory     │  │
   │  │            │  │            │  │ (general)  │  │
   │  │ Per session│  │            │  └────────────┘  │
   │  │ messages   │  │            │                  │
   │  └────────────┘  │            │  ┌────────────┐  │
   │                  │            │  │ user_      │  │
   │  ┌────────────┐  │            │  │ preferences│  │
   │  │ Sessions   │  │            │  │ (learned)  │  │
   │  │ Metadata   │  │            │  └────────────┘  │
   │  └────────────┘  │            │                  │
   └─────────────────┘            └─────────────────┘
```

### MongoDB Collections

#### `chat_history` (managed by `langchain-mongodb`)

```json
{
  "_id": "ObjectId",
  "SessionId": "session-uuid",
  "History": [
    {
      "type": "human",
      "data": {
        "content": "Hello, how are you?",
        "additional_kwargs": {}
      }
    },
    {
      "type": "ai",
      "data": {
        "content": "I'm doing great! How can I help?",
        "additional_kwargs": {}
      }
    }
  ]
}
```

Index: `SessionId` (created automatically by `create_index=True`).

#### `sessions` (custom collection for metadata)

```json
{
  "_id": "session-uuid",
  "created_at": "2026-02-08T10:00:00Z",
  "updated_at": "2026-02-08T10:30:00Z",
  "title": "Auto-generated session title",
  "message_count": 24,
  "summary": "Discussed project planning and coffee preferences."
}
```

### ChromaDB Collections

#### `assistant_memory`

General long-term memory. Stores facts, events, and contextual information.

- **Embedding model**: Google `text-embedding-004` (768 dimensions)
- **Metadata fields**: `source` (conversation, tool, user), `timestamp`, `session_id`
- **Search**: Cosine similarity, top-k retrieval

#### `user_preferences`

Learned user preferences extracted by the PreferenceLearner.

- **Metadata fields**: `category` (communication, interest, routine, personal, dislike), `confidence`, `timestamp`
- **Search**: Cosine similarity filtered by category when relevant

### Learning Pipeline

```
Conversation (every N messages)
        │
        ▼
┌──────────────────┐
│ PreferenceLearner │
│ (Gemini extract)  │
└────────┬─────────┘
         │
         ▼
   Preferences found?
    ┌─────┴─────┐
   YES          NO
    │            │
    ▼            ▼
┌────────┐   (skip)
│ Filter  │
│ conf≥0.7│
└────┬───┘
     │
     ▼
┌─────────────┐
│ ChromaDB     │
│ user_prefs   │
│ collection   │
└─────────────┘
```

### How Preferences Are Used

1. **At conversation start**: Agent calls `recall_preferences` tool with generic context
2. **During conversation**: Agent calls `recall_preferences` when making recommendations
3. **Preference-aware responses**: System prompt instructs agent to personalize based on retrieved preferences
4. **No fine-tuning**: All personalization is through retrieval-augmented generation (RAG)

---

## 11. Data Flow Diagrams

### Text Chat Flow

```
User types message
        │
        ▼
Frontend (Next.js)
  │ WebSocket send: { type: "text", content: "..." }
  ▼
Backend (FastAPI)
  │
  ├─► Memory: get_chat_history(session_id)
  │     └─► MongoDB: load previous messages
  │
  ├─► Memory: recall_preferences(context)
  │     └─► ChromaDB: similarity search → relevant prefs
  │
  ├─► LangGraph Agent
  │     ├─► System prompt + chat history + preferences
  │     ├─► Gemini 2.5 Flash inference
  │     ├─► (Optional) Tool calls → recall_memory, save_memory
  │     └─► Final response text
  │
  ├─► Memory: add_ai_message(response)
  │     └─► MongoDB: persist
  │
  ├─► TTS: synthesize(response) → PCM16 bytes
  │     └─► Google Cloud TTS
  │
  └─► WebSocket send:
        ├─► JSON: { type: "text", content: "..." }
        └─► Binary: PCM16 audio bytes
                │
                ▼
Frontend receives:
  ├─► Display message in ChatPanel
  └─► Send audio to SimliClient → Avatar lip-syncs
```

### Voice Chat Flow

```
User speaks into mic
        │
        ▼
Frontend (Next.js)
  │ Web Audio API → PCM16 @ 16kHz
  │ WebSocket send: binary audio data
  ▼
Backend (FastAPI)
  │
  ├─► STT: transcribe(audio_bytes)
  │     └─► Google Cloud STT V2 → transcript text
  │
  ├─► WebSocket send: { type: "transcript", content: "..." }
  │
  ├─► (Same flow as text chat from here)
  │     LangGraph Agent → response → TTS → PCM16
  │
  └─► WebSocket send:
        ├─► JSON: { type: "text", content: "..." }
        └─► Binary: PCM16 audio bytes
                │
                ▼
Frontend receives:
  ├─► Display transcript + response in ChatPanel
  └─► Send audio to SimliClient → Avatar lip-syncs
```

---

## 12. API Contracts

### WebSocket — `/ws/chat`

**Connection**: `ws://localhost:8000/ws/chat?session_id={uuid}`

#### Client → Server Messages

```typescript
// Text message
{
  "type": "text",
  "content": "string"
}

// Audio message (base64 encoded PCM16)
{
  "type": "audio",
  "audio_data": "base64-encoded-string"
}

// Control messages
{
  "type": "control",
  "action": "start_listening" | "stop_listening" | "clear_history"
}
```

#### Server → Client Messages

```typescript
// Text response (JSON frame)
{
  "type": "text",
  "content": "string"
}

// Transcript of user speech (JSON frame)
{
  "type": "transcript",
  "content": "string"
}

// Status update (JSON frame)
{
  "type": "status",
  "status": "thinking" | "speaking" | "listening" | "idle"
}

// Error (JSON frame)
{
  "type": "error",
  "message": "string"
}

// TTS Audio (binary frame)
// Raw PCM16 bytes, 16kHz, mono
Binary frame: Uint8Array
```

### REST Endpoints

#### Health Check

```
GET /api/health

Response 200:
{
  "status": "healthy",
  "services": {
    "mongodb": "connected",
    "chromadb": "connected",
    "gemini": "available"
  }
}
```

#### Sessions

```
GET /api/sessions
Response 200: [{ "id": "uuid", "title": "...", "created_at": "...", "message_count": 0 }]

GET /api/sessions/{session_id}
Response 200: { "id": "uuid", "title": "...", "messages": [...], "created_at": "..." }

DELETE /api/sessions/{session_id}
Response 204: (no content)
```

#### Preferences

```
GET /api/preferences
Response 200: [{ "text": "...", "category": "...", "confidence": 0.9 }]

DELETE /api/preferences/{preference_id}
Response 204: (no content)
```

---

## 13. Environment & Configuration

### `.env` (Backend — Root Directory)

```env
# Google
GOOGLE_API_KEY=your-google-api-key

# Simli
SIMLI_API_KEY=your-simli-api-key
SIMLI_FACE_ID=your-selected-face-id

# App
ENVIRONMENT=development
LOG_LEVEL=debug

# These are set by Docker Compose, but can be overridden:
# MONGODB_URI=mongodb://localhost:27017
# CHROMADB_HOST=localhost
# CHROMADB_PORT=8100
```

### `.env.local` (Frontend)

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
NEXT_PUBLIC_SIMLI_API_KEY=your-simli-api-key
NEXT_PUBLIC_SIMLI_FACE_ID=your-selected-face-id
```

### Required API Keys

| Key | Source | Free Tier? |
|---|---|---|
| `GOOGLE_API_KEY` | [Google AI Studio](https://aistudio.google.com/apikey) | Yes — 15 RPM for Gemini Flash |
| `SIMLI_API_KEY` | [Simli Dashboard](https://app.simli.com) | Limited free minutes |
| `SIMLI_FACE_ID` | Simli Face Library | Included with account |

---

## 14. Dependencies

### Backend — `requirements.txt`

```txt
# Framework
fastapi>=0.115
uvicorn[standard]>=0.32
websockets>=13.0
python-multipart>=0.0.9

# LLM & Agent
langchain>=0.3
langgraph>=0.2
langchain-google-genai>=2.0
langchain-core>=0.3

# Memory
langchain-mongodb>=0.3
langchain-chroma>=0.2
motor>=3.6
chromadb>=0.5

# Google Cloud
google-cloud-speech>=2.29
google-cloud-texttospeech>=2.24

# Utilities
pydantic>=2.0
pydantic-settings>=2.0
pyyaml>=6.0
python-dotenv>=1.0

# Dev
pytest>=8.0
pytest-asyncio>=0.24
httpx>=0.27
```

### Frontend — `package.json` (dependencies)

```json
{
  "dependencies": {
    "next": "^14.2",
    "react": "^18.3",
    "react-dom": "^18.3",
    "simli-client": "latest",
    "zustand": "^5.0"
  },
  "devDependencies": {
    "@types/node": "^20",
    "@types/react": "^18",
    "@types/react-dom": "^18",
    "autoprefixer": "^10.4",
    "postcss": "^8.4",
    "tailwindcss": "^3.4",
    "typescript": "^5.5"
  }
}
```

---

## 15. Project Structure

```
AIAssistant/
├── AI_ASSISTANT_PLAN.md        # This file
├── docker-compose.yml          # Multi-container orchestration
├── .env                        # Backend environment variables
├── .env.local                  # Frontend environment variables
├── .gitignore
├── README.md                   # Project overview and setup guide
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── pyproject.toml
│   └── app/
│       ├── __init__.py
│       ├── main.py
│       ├── config.py
│       ├── dependencies.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── router.py
│       │   ├── chat.py
│       │   ├── sessions.py
│       │   ├── preferences.py
│       │   └── health.py
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── graph.py
│       │   ├── state.py
│       │   ├── nodes.py
│       │   ├── tools.py
│       │   └── prompts.py
│       ├── voice/
│       │   ├── __init__.py
│       │   ├── stt.py
│       │   ├── tts.py
│       │   └── audio_utils.py
│       ├── memory/
│       │   ├── __init__.py
│       │   ├── manager.py
│       │   ├── chat_history.py
│       │   ├── vector_store.py
│       │   └── preference_learner.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── messages.py
│       │   ├── sessions.py
│       │   └── preferences.py
│       └── personality/
│           ├── __init__.py
│           ├── loader.py
│           └── default.yaml
│
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.ts
│   ├── tsconfig.json
│   ├── public/
│   │   └── favicon.ico
│   └── src/
│       ├── app/
│       │   ├── layout.tsx
│       │   ├── page.tsx
│       │   └── globals.css
│       ├── components/
│       │   ├── AssistantLayout.tsx
│       │   ├── ChatPanel.tsx
│       │   ├── MessageBubble.tsx
│       │   ├── AvatarPanel.tsx
│       │   ├── SimliAvatar.tsx
│       │   ├── VoiceButton.tsx
│       │   ├── StatusIndicator.tsx
│       │   └── SettingsDrawer.tsx
│       ├── hooks/
│       │   ├── useWebSocket.ts
│       │   ├── useAudioCapture.ts
│       │   ├── useSimli.ts
│       │   └── useChat.ts
│       ├── stores/
│       │   ├── chatStore.ts
│       │   ├── voiceStore.ts
│       │   └── settingsStore.ts
│       ├── lib/
│       │   ├── websocket.ts
│       │   ├── audio.ts
│       │   └── api.ts
│       └── types/
│           ├── messages.ts
│           ├── chat.ts
│           └── avatar.ts
│
└── tests/
    └── (backend tests live in backend/tests/)
```

---

## 16. Implementation Roadmap

### Phase 1: Infrastructure & Foundation (Week 1–2)

> Goal: Docker Compose running, FastAPI + Next.js skeleton, MongoDB + ChromaDB connected.

- [ ] **1.1** Create `docker-compose.yml` with all 4 services
- [ ] **1.2** Create `backend/Dockerfile` and `frontend/Dockerfile`
- [ ] **1.3** Scaffold FastAPI app (`main.py`, `config.py`, health endpoint)
- [ ] **1.4** Scaffold Next.js app (App Router, Tailwind, basic layout)
- [ ] **1.5** Verify MongoDB connection from backend (`motor` ping)
- [ ] **1.6** Verify ChromaDB connection from backend (`HttpClient` heartbeat)
- [ ] **1.7** Set up `.env` and `.env.local` with placeholder values
- [ ] **1.8** Create `.gitignore` (node_modules, .env, __pycache__, .next, data volumes)
- [ ] **1.9** `docker compose up --build` works with all services healthy
- [ ] **1.10** Write health check endpoint that validates all service connections

**Exit Criteria**: `docker compose up` starts all 4 services. `GET /api/health` returns all green. Frontend loads at `localhost:3000`.

### Phase 2: Chat & Agent (Week 3–4)

> Goal: Text chat working end-to-end with Gemini via LangGraph agent.

- [ ] **2.1** Implement `MemoryManager` with MongoDB chat history
- [ ] **2.2** Implement LangGraph ReAct agent (`graph.py`, `tools.py`, `prompts.py`)
- [ ] **2.3** Implement WebSocket `/ws/chat` endpoint (text only)
- [ ] **2.4** Implement `ChatPanel` component with message display
- [ ] **2.5** Implement `useWebSocket` hook
- [ ] **2.6** Implement `useChatStore` (Zustand)
- [ ] **2.7** Wire up: type message → WebSocket → agent → response → display
- [ ] **2.8** Verify chat history persists in MongoDB across sessions
- [ ] **2.9** Implement session management (create, list, switch, delete)
- [ ] **2.10** Add error handling and reconnection logic

**Exit Criteria**: Can have a multi-turn text conversation with the assistant. Chat history persists in MongoDB. Sessions can be created and switched.

### Phase 3: Voice Pipeline (Week 5–6)

> Goal: Speak to the assistant and hear it respond.

- [ ] **3.1** Implement `StreamingSTT` service (Google STT V2)
- [ ] **3.2** Implement `TextToSpeech` service (Google TTS, PCM16 output)
- [ ] **3.3** Implement `useAudioCapture` hook (browser mic → PCM16)
- [ ] **3.4** Implement `VoiceButton` component (push-to-talk)
- [ ] **3.5** Extend WebSocket to handle audio messages (binary frames)
- [ ] **3.6** Wire up: mic → STT → agent → TTS → audio playback
- [ ] **3.7** Add `StatusIndicator` (listening / thinking / speaking / idle)
- [ ] **3.8** Test latency and optimize (chunked TTS, parallel processing)

**Exit Criteria**: Can speak a question and hear the assistant's response. Transcript displayed in chat. Status indicator works correctly.

### Phase 4: Avatar System (Week 7–8)

> Goal: Photorealistic avatar lip-syncs with assistant speech.

- [ ] **4.1** Sign up for Simli, get API key and select face ID
- [ ] **4.2** Implement `SimliAvatar` component (WebRTC setup)
- [ ] **4.3** Implement `useSimli` hook (lifecycle management)
- [ ] **4.4** Implement `AvatarPanel` container with layout
- [ ] **4.5** Wire TTS audio → SimliClient.sendAudioData()
- [ ] **4.6** Handle avatar session lifecycle (start, idle, reconnect)
- [ ] **4.7** Implement `AssistantLayout` (avatar left, chat right)
- [ ] **4.8** Test and tune: lip sync timing, idle animations, error states

**Exit Criteria**: Avatar appears and lip-syncs with every response. Handles idle state gracefully. Clean layout with chat + avatar.

### Phase 5: Memory & Learning (Week 9–10)

> Goal: Assistant remembers facts and learns preferences across sessions.

- [ ] **5.1** Implement ChromaDB vector store wrapper
- [ ] **5.2** Implement `recall_memory` and `save_memory` agent tools
- [ ] **5.3** Implement `PreferenceLearner` (Gemini extraction)
- [ ] **5.4** Implement `recall_preferences` agent tool
- [ ] **5.5** Add preference extraction trigger (every N messages or on session end)
- [ ] **5.6** Implement `SettingsDrawer` with preference list (view/delete)
- [ ] **5.7** Implement REST `/api/preferences` endpoints
- [ ] **5.8** Test: share a preference → new session → verify it's recalled
- [ ] **5.9** Test: save a fact → ask about it later → verify recall

**Exit Criteria**: Tell the assistant "I love coffee" → start new session → ask "what do I like?" → gets "coffee". Facts saved and recalled correctly.

### Phase 6: Polish & Integration (Week 11–12)

> Goal: Production-quality UX, error handling, and testing.

- [ ] **6.1** Personality configuration system (YAML loader)
- [ ] **6.2** Settings drawer (voice selection, avatar toggle, personality)
- [ ] **6.3** Responsive design and dark mode
- [ ] **6.4** Comprehensive error handling (network, API limits, timeouts)
- [ ] **6.5** Backend unit tests (agent, memory, voice)
- [ ] **6.6** Frontend component tests
- [ ] **6.7** End-to-end integration test
- [ ] **6.8** README.md with setup instructions
- [ ] **6.9** Performance optimization (WebSocket keep-alive, connection pooling)
- [ ] **6.10** Final review and documentation

**Exit Criteria**: All tests pass. Clean error states. README allows a new developer to set up and run the project.

---

## 17. Architecture Decision Records (ADRs)

### ADR-001: Local-First Development

- **Status**: Accepted
- **Context**: The assistant uses cloud LLM (Gemini) but should not require cloud deployment.
- **Decision**: Run all services locally via Docker Compose. Only external calls are to Google APIs and Simli API.
- **Consequences**: Simple setup, no cloud billing (except API usage), easy debugging. Cannot scale horizontally.

### ADR-002: MongoDB + ChromaDB for Storage

- **Status**: Accepted (Updated from v2 SQLite + FAISS)
- **Context**: Need structured data storage (chat history, sessions) and vector similarity search (semantic memory, preferences).
- **Decision**: MongoDB for structured data via `langchain-mongodb`. ChromaDB for vectors via `langchain-chroma` HTTP client mode. Both run as Docker containers.
- **Rationale**:
  - MongoDB: Production-grade, excellent LangChain integration (`MongoDBChatMessageHistory`), rich querying, scales if needed later.
  - ChromaDB: Purpose-built for embeddings, simple HTTP API, persistent storage, metadata filtering, native LangChain support.
  - Both: Run easily in Docker, no complex setup, data persists via volumes.
- **Rejected alternatives**:
  - SQLite + FAISS (v2): FAISS has no persistence built-in, SQLite lacks rich querying for chat metadata.
  - PostgreSQL + pgvector: Heavier setup, pgvector less mature than ChromaDB for our use case.
  - MongoDB Atlas Vector Search: Requires Atlas (cloud), not suitable for local-only.

### ADR-003: Separate STT/TTS over Gemini Live Audio

- **Status**: Accepted
- **Context**: Gemini supports live audio mode, but we need fine-grained control over the voice pipeline.
- **Decision**: Use Google Cloud STT (V2) and TTS (Chirp3) as separate services.
- **Rationale**:
  - Separate control over STT (streaming, interim results) and TTS (voice selection, audio format).
  - TTS output (PCM16 16KHz) directly compatible with Simli avatar.
  - Can upgrade individual components independently.
  - Gemini Live Audio is newer and less documented for custom integration.

### ADR-004: Simli for Photorealistic Avatar

- **Status**: Accepted (Updated from v2 Three.js + VRM)
- **Context**: User requires avatar that "looks as human as possible."
- **Decision**: Use Simli (`simli-client` SDK) for a photorealistic, AI-generated avatar with real-time lip sync.
- **Rationale**:
  - Photorealistic quality — far exceeds cartoon/VRM avatars.
  - Audio-driven lip sync via WebRTC (low latency).
  - Simple SDK: just send PCM16 audio, get synced video.
  - No local GPU required (rendering is server-side).
  - Idle animations handled by `handleSilence: true`.
- **Rejected alternatives**:
  - Three.js + VRM (@pixiv/three-vrm): Cartoon-style, not photorealistic.
  - D-ID / HeyGen: More expensive, less real-time.
  - SadTalker / Wav2Lip: Require local GPU, higher latency.
- **Trade-offs**:
  - Depends on Simli cloud service (internet required for avatar).
  - Limited free tier — cost for extended usage.
  - Face options limited to Simli's library.

### ADR-005: Docker Compose Orchestration

- **Status**: Accepted (Updated from v2 monolithic)
- **Context**: Multiple services (backend, frontend, MongoDB, ChromaDB) need to work together locally.
- **Decision**: Docker Compose to orchestrate all 4 services with health checks and volume persistence.
- **Rationale**:
  - Single `docker compose up` starts everything.
  - Service discovery via Docker DNS (backend connects to `mongodb:27017`).
  - Volume persistence across restarts.
  - Health checks ensure startup order.
  - Easy to add new services later.
- **Consequences**: Requires Docker installed. Slightly higher memory usage than bare-metal. Hot-reload via bind mounts.

### ADR-006: LangGraph ReAct Agent

- **Status**: Accepted
- **Context**: Need an agent that can reason, call tools, and maintain conversation context.
- **Decision**: Use LangGraph's `create_react_agent` with custom tools for memory operations.
- **Rationale**:
  - ReAct pattern: model decides when to use tools vs. respond directly.
  - Built-in tool calling support with Gemini.
  - `RunnableWithMessageHistory` integrates cleanly with MongoDB chat history.
  - State machine approach makes the agent debuggable and extensible.
- **Alternative considered**: Plain LangChain `AgentExecutor` — deprecated in favor of LangGraph.

---

## 18. Cost Estimates

### Google Gemini (Monthly, Moderate Usage)

| Model | RPM (Free) | Input | Output | Est. Monthly |
|---|---|---|---|---|
| Gemini 2.5 Flash | 15 RPM | $0.15/1M tokens | $0.60/1M tokens | $5–15 |
| Gemini 2.5 Pro | 5 RPM | $1.25/1M tokens | $5.00/1M tokens | $20–60 |

**Recommendation**: Start with Flash (fast, cheap). Upgrade to Pro for complex reasoning if needed.

### Google Cloud STT / TTS

| Service | Free Tier | Paid Rate |
|---|---|---|
| Speech-to-Text V2 | 60 min/month | $0.016/min |
| Text-to-Speech (Chirp3) | 1M chars/month | $0.016/1K chars |

### Simli

| Plan | Minutes | Cost |
|---|---|---|
| Free | Limited trial | $0 |
| Starter | ~100 min/month | ~$29/month |
| Growth | ~500 min/month | ~$99/month |

*Check [simli.com/pricing](https://simli.com) for current rates.*

### Docker / Infrastructure

| Resource | Cost |
|---|---|
| Docker Desktop | Free (personal use) |
| MongoDB (local) | Free |
| ChromaDB (local) | Free |

### Total Estimated Monthly Cost

| Tier | Cost | Suitable For |
|---|---|---|
| Minimal (free tiers only) | $0 | Testing and development |
| Light usage | $10–30 | Daily personal use |
| Moderate usage | $50–100 | Heavy daily use with Simli |

---

## 19. Future Extensions

| Extension | Complexity | Priority | Description |
|---|---|---|---|
| Multi-language support | Medium | Low | Additional STT/TTS language codes |
| Custom avatar upload | Medium | Medium | Upload own face to Simli |
| Tool plugins | Medium | Medium | Weather, calendar, web search tools |
| Conversation summaries | Low | High | Auto-summarize long sessions |
| Voice cloning | High | Low | Custom TTS voice |
| Multi-user support | High | Low | User accounts and auth |
| Mobile PWA | Medium | Low | Progressive Web App wrapper |
| Cloud deployment | Medium | Low | Deploy to GCP Cloud Run |
| RAG over documents | Medium | Medium | Upload PDFs/docs to vector store |
| Streaming LLM response | Low | High | Token-by-token streaming to frontend |

---

*End of AI Assistant Technical Plan v3.0*
