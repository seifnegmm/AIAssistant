"""Memory manager coordinating MongoDB chat history and ChromaDB vector store.

This module provides the central memory layer for the AI assistant:
- **Chat History**: Persisted in MongoDB via langchain-mongodb
- **Semantic Memory**: Stored in ChromaDB via langchain-chroma with Google embeddings
- **Preference Store**: Dedicated ChromaDB collection for learned user preferences
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

from app.config import settings
from app.memory.chat_store import (
    ChatHistoryStore,
    COLLECTION_NAME as CHAT_SESSIONS_COLLECTION,
)

logger = logging.getLogger(__name__)

# Collection names
GENERAL_MEMORY_COLLECTION = "general_memory"
USER_PREFERENCES_COLLECTION = "user_preferences"


class MemoryManager:
    """Coordinates MongoDB for chat history and ChromaDB for semantic memory.

    Lifecycle:
        manager = MemoryManager()
        await manager.initialize()   # call once at startup
        ...
        await manager.close()        # call once at shutdown
    """

    def __init__(self) -> None:
        self._mongo_client: AsyncIOMotorClient | None = None
        self._mongo_db: AsyncIOMotorDatabase | None = None
        self._chroma_client: chromadb.HttpClient | None = None
        self._embeddings: GoogleGenerativeAIEmbeddings | None = None
        self._general_memory_store: Chroma | None = None
        self._preference_store: Chroma | None = None
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Connect to MongoDB and ChromaDB, prepare LangChain wrappers."""
        if self._initialized:
            logger.warning("MemoryManager already initialized - skipping")
            return

        # 1) MongoDB -------------------------------------------------
        logger.info("Connecting to MongoDB at %s", settings.mongodb_uri)
        self._mongo_client = AsyncIOMotorClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=5_000,
        )
        self._mongo_db = self._mongo_client[settings.mongodb_database]
        await self._mongo_client.admin.command("ping")
        logger.info("MongoDB connection established")

        # 2) ChromaDB ------------------------------------------------
        logger.info(
            "Connecting to ChromaDB at %s:%s",
            settings.chromadb_host,
            settings.chromadb_port,
        )
        self._chroma_client = chromadb.HttpClient(
            host=settings.chromadb_host,
            port=settings.chromadb_port,
        )
        self._chroma_client.heartbeat()
        logger.info("ChromaDB connection established")

        # 3) Google Embeddings ---------------------------------------
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key,
        )
        logger.info("Google embeddings model loaded: %s", settings.embedding_model)

        # 4) LangChain Chroma vector stores --------------------------
        self._general_memory_store = Chroma(
            client=self._chroma_client,
            collection_name=GENERAL_MEMORY_COLLECTION,
            embedding_function=self._embeddings,
        )
        self._preference_store = Chroma(
            client=self._chroma_client,
            collection_name=USER_PREFERENCES_COLLECTION,
            embedding_function=self._embeddings,
        )
        logger.info("ChromaDB vector stores initialized")

        self._initialized = True
        logger.info("MemoryManager fully initialized")

    async def close(self) -> None:
        """Release connections."""
        if self._mongo_client is not None:
            self._mongo_client.close()
            logger.info("MongoDB connection closed")
        self._initialized = False

    # ------------------------------------------------------------------
    # Properties (guard against use before init)
    # ------------------------------------------------------------------

    @property
    def mongo_db(self) -> AsyncIOMotorDatabase:
        if self._mongo_db is None:
            raise RuntimeError(
                "MemoryManager not initialized - call initialize() first"
            )
        return self._mongo_db

    @property
    def chroma_client(self) -> chromadb.HttpClient:
        if self._chroma_client is None:
            raise RuntimeError(
                "MemoryManager not initialized - call initialize() first"
            )
        return self._chroma_client

    @property
    def embeddings(self) -> GoogleGenerativeAIEmbeddings:
        if self._embeddings is None:
            raise RuntimeError(
                "MemoryManager not initialized - call initialize() first"
            )
        return self._embeddings

    @property
    def general_memory(self) -> Chroma:
        if self._general_memory_store is None:
            raise RuntimeError(
                "MemoryManager not initialized - call initialize() first"
            )
        return self._general_memory_store

    @property
    def preference_store(self) -> Chroma:
        if self._preference_store is None:
            raise RuntimeError(
                "MemoryManager not initialized - call initialize() first"
            )
        return self._preference_store

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # ------------------------------------------------------------------
    # Chat History (MongoDB — custom single-doc-per-session store)
    # ------------------------------------------------------------------

    def get_langchain_chat_history(self, session_id: str) -> ChatHistoryStore:
        """Return a LangChain-compatible chat history for a session.

        Uses our custom ``ChatHistoryStore`` which stores all messages in a
        single document per session with a clean, readable schema.
        """
        return ChatHistoryStore(
            connection_string=settings.mongodb_uri,
            session_id=session_id,
            database_name=settings.mongodb_database,
            collection_name=CHAT_SESSIONS_COLLECTION,
        )

    async def get_chat_history(
        self,
        session_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Retrieve recent messages for a session from MongoDB.

        Returns the clean message dicts from the session document.
        """
        store = self.get_langchain_chat_history(session_id)
        messages = store.get_clean_history()
        if limit:
            messages = messages[-limit:]
        return messages

    # ------------------------------------------------------------------
    # Semantic Memory (ChromaDB via langchain-chroma)
    # ------------------------------------------------------------------

    async def save_memory(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        collection_name: str = GENERAL_MEMORY_COLLECTION,
    ) -> str:
        """Store a piece of information in the vector store.

        Returns the document ID.
        """
        doc_id = str(uuid4())
        meta = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }
        doc = Document(page_content=content, metadata=meta)

        store = self._get_vector_store(collection_name)
        store.add_documents(documents=[doc], ids=[doc_id])
        logger.debug("Saved memory %s to %s", doc_id, collection_name)
        return doc_id

    async def recall_memory(
        self,
        query: str,
        k: int = 5,
        collection_name: str = GENERAL_MEMORY_COLLECTION,
    ) -> list[Document]:
        """Retrieve semantically similar documents from the vector store."""
        store = self._get_vector_store(collection_name)
        results = store.similarity_search(query, k=k)
        logger.debug(
            "Recalled %d memories from %s for query: %s",
            len(results),
            collection_name,
            query[:80],
        )
        return results

    async def recall_memory_with_scores(
        self,
        query: str,
        k: int = 5,
        collection_name: str = GENERAL_MEMORY_COLLECTION,
    ) -> list[tuple[Document, float]]:
        """Retrieve documents with similarity scores (lower = more similar)."""
        store = self._get_vector_store(collection_name)
        return store.similarity_search_with_score(query, k=k)

    def delete_memory(self, query: str, session_id: str = "default", k: int = 3) -> int:
        """
        Delete memories matching the given query.

        Args:
            query: Text query to find memories to delete
            session_id: Session identifier for namespacing
            k: Number of top matching memories to delete

        Returns:
            Number of memories deleted
        """
        collection_name = "general_memory"

        # We need the document IDs — ChromaDB stores them in metadata or we
        # can query the raw collection directly.
        raw_collection = self.chroma_client.get_or_create_collection(
            name=collection_name
        )

        # CRITICAL: Manually embed the query using our configured embedding function
        # to avoid ChromaDB's default embedder (384-dim vs our 3072-dim)
        # Use SYNC embed_query to avoid event loop conflicts when called from sync tools
        query_embedding = self._embeddings.embed_query(query)

        # Query the raw collection with the pre-computed embedding
        results = raw_collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        ids_to_delete = results.get("ids", [[]])[0]
        if not ids_to_delete:
            logger.info("No matching memories found for deletion query: %s", query[:80])
            return 0

        raw_collection.delete(ids=ids_to_delete)
        logger.info(
            "Deleted %d memories from %s matching query: %s",
            len(ids_to_delete),
            collection_name,
            query[:80],
        )
        return len(ids_to_delete)

    # ------------------------------------------------------------------
    # Preference Store (ChromaDB collection for user preferences)
    # ------------------------------------------------------------------

    async def save_preference(
        self,
        key: str,
        value: str,
        category: str = "general",
        confidence: float = 0.5,
        source: str = "inferred",
    ) -> str:
        """Store a user preference in the dedicated preference collection."""
        doc_id = str(uuid4())
        content = f"{key}: {value}"
        metadata = {
            "key": key,
            "value": value,
            "category": category,
            "confidence": confidence,
            "source": source,
            "learned_at": datetime.now(timezone.utc).isoformat(),
        }
        doc = Document(page_content=content, metadata=metadata)
        self.preference_store.add_documents(documents=[doc], ids=[doc_id])
        logger.info(
            "Saved preference: %s = %s (confidence=%.2f)", key, value, confidence
        )
        return doc_id

    async def recall_preferences(
        self,
        query: str,
        k: int = 10,
    ) -> list[Document]:
        """Retrieve preferences relevant to a query."""
        return self.preference_store.similarity_search(query, k=k)

    async def get_all_preferences(self) -> list[dict[str, Any]]:
        """Return all stored preferences as dicts (for the API endpoint)."""
        raw_collection = self.chroma_client.get_or_create_collection(
            name=USER_PREFERENCES_COLLECTION
        )
        data = raw_collection.get()
        preferences: list[dict[str, Any]] = []
        if data and data.get("documents"):
            for i, doc in enumerate(data["documents"]):
                meta = data["metadatas"][i] if data.get("metadatas") else {}
                preferences.append(
                    {
                        "id": data["ids"][i] if data.get("ids") else None,
                        "content": doc,
                        **meta,
                    }
                )
        return preferences

    async def clear_preferences(self) -> None:
        """Delete all user preferences."""
        self.chroma_client.delete_collection(name=USER_PREFERENCES_COLLECTION)
        # Re-create empty collection
        self._preference_store = Chroma(
            client=self.chroma_client,
            collection_name=USER_PREFERENCES_COLLECTION,
            embedding_function=self.embeddings,
        )
        logger.info("All user preferences cleared")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_vector_store(self, collection_name: str) -> Chroma:
        """Return the appropriate Chroma vector store for a collection name."""
        if collection_name == USER_PREFERENCES_COLLECTION:
            return self.preference_store
        if collection_name == GENERAL_MEMORY_COLLECTION:
            return self.general_memory
        # Dynamic collection for custom stores
        return Chroma(
            client=self.chroma_client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
        )

    def get_or_create_collection(
        self, name: str = GENERAL_MEMORY_COLLECTION
    ) -> chromadb.Collection:
        """Return a raw ChromaDB collection, creating it if needed."""
        return self.chroma_client.get_or_create_collection(name=name)
