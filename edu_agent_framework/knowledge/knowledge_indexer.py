"""
知识索引器
==========
为私域知识建立向量索引，支持语义检索
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import pickle
import numpy as np

from .document_processor import DocumentChunk


class KnowledgeIndexer:
    """
    知识索引器

    功能：
    1. 文本向量化
    2. 向量索引构建（FAISS）
    3. 语义相似度检索

    技术栈：
    - Embedding: OpenAI / SentenceTransformers / 本地模型
    - 向量库: FAISS
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        embedding_type: str = "openai",  # openai / local
        index_path: str = None
    ):
        self.embedding_model = embedding_model
        self.embedding_type = embedding_type
        self.index_path = index_path

        # 索引和元数据
        self._index = None
        self._chunks: List[DocumentChunk] = []
        self._embeddings: List[np.ndarray] = []

        # 嵌入客户端
        self._embed_client = None

    def _get_embed_client(self):
        """获取嵌入客户端"""
        if self._embed_client is None:
            if self.embedding_type == "openai":
                import os
                try:
                    from openai import OpenAI
                    self._embed_client = OpenAI(
                        api_key=os.getenv('OPENAI_API_KEY'),
                        base_url=os.getenv('OPENAI_BASE_URL')
                    )
                except ImportError:
                    raise ImportError("请安装openai: pip install openai")
            else:
                try:
                    from sentence_transformers import SentenceTransformer
                    self._embed_client = SentenceTransformer(self.embedding_model)
                except ImportError:
                    raise ImportError("请安装sentence-transformers: pip install sentence-transformers")

        return self._embed_client

    def _embed_text(self, text: str) -> np.ndarray:
        """获取文本嵌入向量"""
        client = self._get_embed_client()

        if self.embedding_type == "openai":
            response = client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return np.array(response.data[0].embedding)
        else:
            return client.encode(text)

    def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """批量获取嵌入向量"""
        client = self._get_embed_client()

        if self.embedding_type == "openai":
            # OpenAI批量嵌入
            embeddings = []
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                response = client.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                embeddings.extend([np.array(d.embedding) for d in response.data])
            return embeddings
        else:
            return [client.encode(t) for t in texts]

    def add_chunks(self, chunks: List[DocumentChunk]):
        """添加文档块到索引"""
        if not chunks:
            return

        # 获取嵌入
        texts = [c.content for c in chunks]
        embeddings = self._embed_batch(texts)

        # 更新存储
        self._chunks.extend(chunks)
        self._embeddings.extend(embeddings)

        # 重建FAISS索引
        self._rebuild_index()

    def _rebuild_index(self):
        """重建FAISS索引"""
        if not self._embeddings:
            return

        try:
            import faiss
        except ImportError:
            print("警告: FAISS未安装，将使用简单的numpy搜索")
            return

        # 创建索引
        dim = len(self._embeddings[0])
        self._index = faiss.IndexFlatIP(dim)  # 内积（余弦相似度）

        # 归一化并添加
        vectors = np.array(self._embeddings).astype('float32')
        faiss.normalize_L2(vectors)
        self._index.add(vectors)

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.5
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        语义搜索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            threshold: 相似度阈值

        Returns:
            [(chunk, score), ...] 排序后的结果
        """
        if not self._embeddings:
            return []

        # 获取查询向量
        query_vec = self._embed_text(query).astype('float32').reshape(1, -1)

        if self._index is not None:
            # 使用FAISS搜索
            import faiss
            faiss.normalize_L2(query_vec)
            scores, indices = self._index.search(query_vec, min(top_k, len(self._chunks)))

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if score >= threshold:
                    results.append((self._chunks[idx], float(score)))
            return results
        else:
            # 简单numpy搜索
            embeddings = np.array(self._embeddings)
            # 归一化
            query_norm = query_vec / np.linalg.norm(query_vec)
            emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            # 计算相似度
            similarities = np.dot(emb_norm, query_norm.T).flatten()

            # 排序
            indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in indices:
                if similarities[idx] >= threshold:
                    results.append((self._chunks[idx], float(similarities[idx])))
            return results

    def save(self, path: str = None):
        """保存索引到文件"""
        path = path or self.index_path
        if not path:
            raise ValueError("请指定保存路径")

        save_data = {
            "chunks": [c.to_dict() for c in self._chunks],
            "embeddings": [e.tolist() for e in self._embeddings],
            "embedding_model": self.embedding_model,
            "embedding_type": self.embedding_type
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)

    def load(self, path: str = None):
        """从文件加载索引"""
        path = path or self.index_path
        if not path or not Path(path).exists():
            return

        with open(path, 'rb') as f:
            save_data = pickle.load(f)

        self._chunks = [DocumentChunk(**c) for c in save_data["chunks"]]
        self._embeddings = [np.array(e) for e in save_data["embeddings"]]
        self.embedding_model = save_data.get("embedding_model", self.embedding_model)
        self.embedding_type = save_data.get("embedding_type", self.embedding_type)

        self._rebuild_index()

    def clear(self):
        """清空索引"""
        self._index = None
        self._chunks = []
        self._embeddings = []


class RAGEduAgent:
    """
    支持RAG的教育智能体增强器

    将知识索引器与教育智能体结合
    """

    def __init__(
        self,
        base_agent,
        indexer: KnowledgeIndexer
    ):
        self.agent = base_agent
        self.indexer = indexer

    def chat_with_rag(
        self,
        user_input: str,
        top_k: int = 3,
        **kwargs
    ) -> str:
        """
        带RAG检索的对话

        Args:
            user_input: 用户输入
            top_k: 检索结果数量
            **kwargs: 传递给agent的参数

        Returns:
            智能体回复
        """
        # 检索相关知识
        results = self.indexer.search(user_input, top_k=top_k)

        # 构建上下文
        context_parts = []
        for chunk, score in results:
            context_parts.append(f"[来源: {chunk.source}, 相关度: {score:.2f}]\n{chunk.content}")

        context = "\n---\n".join(context_parts)

        # 调用智能体
        return self.agent.chat(user_input, context=context, **kwargs)
