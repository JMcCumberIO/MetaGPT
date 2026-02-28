#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Desc   : the implement of memory storage
"""
import shutil
from pathlib import Path

from llama_index.core.embeddings import BaseEmbedding

from metagpt.const import DATA_PATH, MEM_TTL
from metagpt.logs import logger
from metagpt.rag.engines.simple import SimpleEngine
from metagpt.rag.schema import FAISSIndexConfig, FAISSRetrieverConfig
from metagpt.schema import Message
from metagpt.utils.embedding import get_embedding


class MemoryStorage(object):
    """
    The memory storage with Faiss as ANN search engine
    """

    def __init__(self, mem_ttl: int = MEM_TTL, embedding: BaseEmbedding = None, similarity_threshold: float = 0.5):
        """Initialise MemoryStorage.

        Args:
            mem_ttl: Time-to-live for memories in seconds (reserved for future use).
            embedding: Optional custom embedding model.  Defaults to the globally
                configured embedding.
            similarity_threshold: Minimum relevance score (0–1) that a retrieved
                memory must reach to be included in search results.  A higher value
                means only very relevant memories are returned.  Defaults to 0.5.
        """
        self.role_id: str = None
        self.role_mem_path: str = None
        self.mem_ttl: int = mem_ttl  # reserved for future TTL-based eviction
        self.similarity_threshold: float = similarity_threshold
        self._initialized: bool = False
        self.embedding = embedding or get_embedding()

        self.faiss_engine = None

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def recover_memory(self, role_id: str) -> list[Message]:
        self.role_id = role_id
        self.role_mem_path = Path(DATA_PATH / f"role_mem/{self.role_id}/")
        self.role_mem_path.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.role_mem_path

        if self.role_mem_path.joinpath("default__vector_store.json").exists():
            self.faiss_engine = SimpleEngine.from_index(
                index_config=FAISSIndexConfig(persist_path=self.cache_dir),
                retriever_configs=[FAISSRetrieverConfig()],
                embed_model=self.embedding,
            )
        else:
            self.faiss_engine = SimpleEngine.from_objs(
                objs=[], retriever_configs=[FAISSRetrieverConfig()], embed_model=self.embedding
            )
        self._initialized = True

    def add(self, message: Message) -> bool:
        """add message into memory storage"""
        self.faiss_engine.add_objs([message])
        logger.info(f"Role {self.role_id}'s memory_storage add a message")

    async def search_similar(self, message: Message, k: int = 4) -> list[Message]:
        """Return the *k* most similar stored messages above the similarity threshold.

        Args:
            message: The query message whose content is used for similarity search.
            k: Maximum number of results to return.

        Returns:
            List of ``Message`` objects whose relevance score is at or above
            ``self.similarity_threshold``, ordered by descending relevance.
        """
        resp = await self.faiss_engine.aretrieve(message.content)
        filtered: list[Message] = []
        for item in resp:
            # llama_index retrieval nodes expose a `score` attribute.
            # Higher scores indicate greater similarity.  We keep results that
            # meet or exceed the configured threshold.
            score = item.score if item.score is not None else 0.0
            if score >= self.similarity_threshold:
                obj = item.metadata.get("obj")
                if obj is not None:
                    filtered.append(obj)
            if len(filtered) >= k:
                break
        return filtered

    def clean(self):
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        self._initialized = False

    def persist(self):
        if self.faiss_engine:
            self.faiss_engine.retriever._index.storage_context.persist(self.cache_dir)
