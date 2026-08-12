import json
import os
import re
import pickle
import numpy as np
from .bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
from typing import List, Dict, Any, Optional
from ..utils.helpers import logger
from ..config import config
from .fusion import reciprocal_rank_fusion

# NOTE: faiss and sentence_transformers are imported lazily inside the classes
# that use them (see FAISSVectorStore) so that `import kssrag` and the
# download-free stores (bm25, bm25s, tfidf, hybrid_offline) do not pull in
# torch/faiss. See helpers.setup_faiss for the AVX-fallback loader.

class BaseVectorStore:
    """Base class for vector stores"""
    
    def __init__(self, persist_path: Optional[str] = None):
        self.persist_path = persist_path
        self.documents: List[Dict[str, Any]] = []
        self.doc_texts: List[str] = []
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the vector store"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def retrieve(self, query: str, top_k: int = 5,
                 metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve documents based on query.

        ``metadata_filter`` restricts candidates to documents whose ``metadata``
        matches every key/value pair (AND semantics).
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    @staticmethod
    def _matches_metadata(doc: Dict[str, Any],
                          metadata_filter: Optional[Dict[str, Any]]) -> bool:
        """True if the doc's metadata satisfies all filter pairs (AND)."""
        if not metadata_filter:
            return True
        metadata = doc.get("metadata", {})
        return all(metadata.get(key) == value for key, value in metadata_filter.items())
    
    def persist(self):
        """Persist the vector store to disk"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def load(self):
        """Load the vector store from disk"""
        raise NotImplementedError("Subclasses must implement this method")

class BM25VectorStore(BaseVectorStore):
    """BM25 vector store implementation"""
    
    def __init__(self, persist_path: Optional[str] = "bm25_index.pkl"):
        super().__init__(persist_path)
        self.bm25 = None
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        self.doc_texts = [doc["content"] for doc in documents]
        
        # Tokenize corpus for BM25
        tokenized_corpus = [self._tokenize(doc) for doc in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        logger.info(f"BM25 index created with {len(self.documents)} documents")
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25"""
        return re.findall(r'\w+', text.lower())
    
    def retrieve(self, query: str, top_k: int = 5,
                 metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self.bm25:
            raise ValueError("BM25 index not initialized. Call add_documents first.")
        
        try:
            tokenized_query = self._tokenize(query)
            doc_scores = self.bm25.get_scores(tokenized_query)
            
            # Rank only docs that match the metadata filter (if any), then
            # take the top_k highest-scoring ones.
            candidates = [
                i for i in range(len(self.documents))
                if self._matches_metadata(self.documents[i], metadata_filter)
            ]
            candidates.sort(key=lambda i: doc_scores[i], reverse=True)
            top_indices = candidates[:top_k]
            
            if not top_indices:
                logger.warning(f"No valid indices found for query: {query}")
                return []
                
            return [self.documents[i] for i in top_indices]
            
        except Exception as e:
            logger.error(f"Error in BM25 retrieval: {str(e)}")
            return []
    
    def persist(self):
        if self.persist_path:
            with open(self.persist_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'doc_texts': self.doc_texts,
                    'bm25': self.bm25
                }, f)
            logger.info(f"BM25 index persisted to {self.persist_path}")
    
    def load(self):
        if self.persist_path and os.path.exists(self.persist_path):
            with open(self.persist_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.doc_texts = data['doc_texts']
                self.bm25 = data['bm25']
            logger.info(f"BM25 index loaded from {self.persist_path}")

import tempfile

class FAISSVectorStore(BaseVectorStore):
    def __init__(self, persist_path: Optional[str] = None, model_name: Optional[str] = None):
        # Only setup FAISS when this vector store is actually used
        from ..utils.helpers import setup_faiss
        faiss_available, _ = setup_faiss("faiss")  # Explicitly request FAISS
        
        if not faiss_available:
            raise ImportError("FAISS is not available. Please install it with 'pip install faiss-cpu' or use a different vector store.")

        import faiss
        from sentence_transformers import SentenceTransformer

        super().__init__(persist_path)
        self.model_name = model_name or config.FAISS_MODEL_NAME
        # Handle cache directory permissions
        try:
            cache_dir = config.CACHE_DIR
            os.makedirs(cache_dir, exist_ok=True)
            # Test if we can write to the cache directory
            test_file = os.path.join(cache_dir, 'write_test.txt')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except PermissionError:
            logger.warning(f"Could not write to cache directory {cache_dir}. Using temp directory.")
            cache_dir = tempfile.gettempdir()
        
        self.model = SentenceTransformer(self.model_name, cache_folder=cache_dir)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata_path = persist_path + ".meta" if persist_path else None
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        self.doc_texts = [doc["content"] for doc in documents]
        
        # Generate embeddings in batches
        embeddings = []
        batch_size = config.BATCH_SIZE
        
        for i in range(0, len(self.doc_texts), batch_size):
            batch_texts = self.doc_texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
            embeddings.append(batch_embeddings)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(self.doc_texts)-1)//batch_size + 1}")
        
        embeddings = np.vstack(embeddings).astype('float32')
        self.index.add(embeddings)
        
        logger.info(f"FAISS index created with {len(self.documents)} documents")
    
    def retrieve(self, query: str, top_k: int = 5,
                 metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self.index or self.index.ntotal == 0:
            raise ValueError("FAISS index not initialized. Call add_documents first.")
        
        try:
            query_embedding = self.model.encode([query])
            # Fetch a wider candidate pool only when filtering, so metadata-
            # filtered docs still have a chance to fill top_k. Unfiltered
            # queries search exactly top_k (no perf regression).
            if metadata_filter:
                search_k = min(max(top_k, top_k * 4), self.index.ntotal)
            else:
                search_k = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(query_embedding.astype('float32'), search_k)
            
            # Filter out invalid indices (FAISS might return -1) and any docs
            # that don't match the metadata filter, preserving distance order.
            valid_indices = [
                i for i in indices[0]
                if 0 <= i < len(self.documents)
                and self._matches_metadata(self.documents[i], metadata_filter)
            ][:top_k]
            
            if not valid_indices:
                logger.warning(f"No valid indices found for query: {query}")
                return []
                
            return [self.documents[i] for i in valid_indices]
        except Exception as e:
            logger.error(f"Error in FAISS retrieval: {str(e)}")
            return []
    
    def persist(self):
        if self.persist_path and self.metadata_path:
            import faiss
            faiss.write_index(self.index, self.persist_path)
            
            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'doc_texts': self.doc_texts
                }, f)
            logger.info(f"FAISS index persisted to {self.persist_path}")
    
    def load(self):
        if (self.persist_path and os.path.exists(self.persist_path) and
            self.metadata_path and os.path.exists(self.metadata_path)):

            import faiss
            self.index = faiss.read_index(self.persist_path)
            
            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.doc_texts = data['doc_texts']
            logger.info(f"FAISS index loaded from {self.persist_path}")

class TFIDFVectorStore(BaseVectorStore):
    """TFIDF vector store implementation"""
    
    def __init__(self, persist_path: Optional[str] = "tfidf_index.pkl", max_features: int = 10000):
        super().__init__(persist_path)
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.tfidf_matrix = None
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        self.doc_texts = [doc["content"] for doc in documents]
        
        # Fit and transform the documents
        self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)
        
        logger.info(f"TFIDF index created with {len(self.documents)} documents")
    
    def retrieve(self, query: str, top_k: int = 5,
                 metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if self.tfidf_matrix is None:
            raise ValueError("TFIDF index not initialized. Call add_documents first.")
        
        try:
            # Transform the query
            query_vec = self.vectorizer.transform([query])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Rank only docs that match the metadata filter (if any).
            candidates = [
                i for i in range(len(self.documents))
                if self._matches_metadata(self.documents[i], metadata_filter)
            ]
            candidates.sort(key=lambda i: similarities[i], reverse=True)
            top_indices = candidates[:top_k]
            
            if not top_indices:
                logger.warning(f"No valid indices found for query: {query}")
                return []
                
            return [self.documents[i] for i in top_indices]
            
        except Exception as e:
            logger.error(f"Error in TFIDF retrieval: {str(e)}")
            return []
    
    def persist(self):
        if self.persist_path:
            with open(self.persist_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'doc_texts': self.doc_texts,
                    'vectorizer': self.vectorizer,
                    'tfidf_matrix': self.tfidf_matrix
                }, f)
            logger.info(f"TFIDF index persisted to {self.persist_path}")
    
    def load(self):
        if self.persist_path and os.path.exists(self.persist_path):
            with open(self.persist_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.doc_texts = data['doc_texts']
                self.vectorizer = data['vectorizer']
                self.tfidf_matrix = data['tfidf_matrix']
            logger.info(f"TFIDF index loaded from {self.persist_path}")

class HybridVectorStore(BaseVectorStore):
    """Hybrid vector store combining BM25 and FAISS"""
    
    def __init__(self, persist_path: Optional[str] = "hybrid_index"):
        super().__init__(persist_path)
        self.bm25_store = BM25VectorStore(persist_path + "_bm25")
        self.faiss_store = FAISSVectorStore(persist_path + "_faiss")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        self.bm25_store.add_documents(documents)
        self.faiss_store.add_documents(documents)
        logger.info(f"Hybrid index created with {len(self.documents)} documents")
    
    def retrieve(self, query: str, top_k: int = 5,
                 metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        try:
            # Retrieve a wider candidate set from each method, then fuse their
            # rankings with RRF. Fusing by rank (not raw score) lets us combine
            # lexical BM25 and semantic FAISS results without normalizing their
            # incompatible score scales.
            bm25_results = self.bm25_store.retrieve(query, top_k * 2, metadata_filter)
            faiss_results = self.faiss_store.retrieve(query, top_k * 2, metadata_filter)

            fused = reciprocal_rank_fusion(
                [bm25_results, faiss_results], top_k=top_k
            )
            return fused

        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {str(e)}")
            return []
    
    def persist(self):
        self.bm25_store.persist()
        self.faiss_store.persist()
        logger.info(f"Hybrid index persisted")
    
    def load(self):
        self.bm25_store.load()
        self.faiss_store.load()
        self.documents = self.bm25_store.documents
        logger.info(f"Hybrid index loaded")

class HybridOfflineVectorStore(BaseVectorStore):
    """Hybrid offline vector store combining BM25 and TFIDF"""
    
    def __init__(self, persist_path: Optional[str] = "hybrid_offline_index"):
        super().__init__(persist_path)
        self.bm25_store = BM25VectorStore(persist_path + "_bm25")
        self.tfidf_store = TFIDFVectorStore(persist_path + "_tfidf")
        self.alpha = 0.5  # Weight for BM25 vs TFIDF
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        self.bm25_store.add_documents(documents)
        self.tfidf_store.add_documents(documents)
        logger.info(f"Hybrid offline index created with {len(self.documents)} documents")
    
    def retrieve(self, query: str, top_k: int = 5,
                 metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        try:
            # Retrieve a wider candidate set from each method, then fuse with RRF,
            # weighting BM25 vs TF-IDF by alpha. RRF fuses by rank position, which
            # avoids the flawed position-normalization the previous approach used.
            bm25_results = self.bm25_store.retrieve(query, top_k * 2, metadata_filter)
            tfidf_results = self.tfidf_store.retrieve(query, top_k * 2, metadata_filter)

            fused = reciprocal_rank_fusion(
                [bm25_results, tfidf_results],
                top_k=top_k,
                weights=[self.alpha, 1.0 - self.alpha],
            )
            return fused

        except Exception as e:
            logger.error(f"Error in hybrid offline retrieval: {str(e)}")
            return []
    
    def persist(self):
        self.bm25_store.persist()
        self.tfidf_store.persist()
        logger.info(f"Hybrid offline index persisted")
    
    def load(self):
        self.bm25_store.load()
        self.tfidf_store.load()
        self.documents = self.bm25_store.documents
        logger.info(f"Hybrid offline index loaded")

class BM25SVectorStore(BaseVectorStore):
    """BM25S vector store using the bm25s library for ultra-fast retrieval"""

    def __init__(self, persist_path: Optional[str] = "bm25s_index.pkl"):
        from Stemmer import Stemmer
        super().__init__(persist_path)
        self.bm25_retriever = None
        self.stemmer = Stemmer("english")
        self.corpus_tokens = None
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        self.doc_texts = [doc["content"] for doc in documents]
        
        try:
            import bm25s
            # Tokenize corpus with BM25S
            self.corpus_tokens = bm25s.tokenize(
                self.doc_texts, 
                stopwords="en", 
                stemmer=self.stemmer
            )
            
            # Create and index with BM25S
            self.bm25_retriever = bm25s.BM25()
            self.bm25_retriever.index(self.corpus_tokens)
            
            logger.info(f"BM25S index created with {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"BM25S initialization failed: {str(e)}")
            raise
    
    def retrieve(self, query: str, top_k: int = 5,
                 metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self.bm25_retriever:
            raise ValueError("BM25S index not initialized. Call add_documents first.")
        
        try:
            import bm25s
            if not self.documents:
                return []
            # Tokenize query with BM25S
            query_tokens = bm25s.tokenize([query], stemmer=self.stemmer)
            
            # bm25s raises if k exceeds the corpus size, so cap it. Fetch a
            # wider candidate pool when filtering so filtered docs can still
            # fill top_k.
            fetch_k = min(
                top_k * 4 if metadata_filter else top_k,
                len(self.documents),
            )
            results, scores = self.bm25_retriever.retrieve(query_tokens, k=fetch_k)
            
            # Format results (respecting the metadata filter). bm25s may pad
            # results with -1 (or repeat docs) when k exceeds the corpus size,
            # so guard indices and dedupe.
            retrieved_docs = []
            seen = set()
            for i in range(results.shape[1]):
                doc_idx = int(results[0, i])
                if doc_idx < 0 or doc_idx >= len(self.documents):
                    continue
                if doc_idx in seen:
                    continue
                if not self._matches_metadata(self.documents[doc_idx], metadata_filter):
                    continue
                seen.add(doc_idx)
                retrieved_docs.append(self.documents[doc_idx])
                if len(retrieved_docs) == top_k:
                    break
            
            logger.info(f"BM25S retrieved {len(retrieved_docs)} documents for query: {query}")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"BM25S retrieval failed for query '{query}': {str(e)}")
            return []
    
    def persist(self):
        if self.persist_path:
            with open(self.persist_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'doc_texts': self.doc_texts,
                    'corpus_tokens': self.corpus_tokens,
                    'bm25_retriever': self.bm25_retriever
                }, f)
            logger.info(f"BM25S index persisted to {self.persist_path}")
    
    def load(self):
        if self.persist_path and os.path.exists(self.persist_path):
            with open(self.persist_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.doc_texts = data['doc_texts']
                self.corpus_tokens = data['corpus_tokens']
                self.bm25_retriever = data['bm25_retriever']
            logger.info(f"BM25S index loaded from {self.persist_path}")