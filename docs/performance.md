# Performance Tuning Guide

## Overview

This guide provides comprehensive performance optimization strategies for KSS RAG, covering everything from basic configuration tuning to advanced scaling techniques.

## Performance Benchmarks

### Baseline Performance

| Operation | Small Instance | Medium Instance | Large Instance |
|-----------|----------------|-----------------|----------------|
| Document Indexing | 5-10 sec/1000 chunks | 2-5 sec/1000 chunks | 1-3 sec/1000 chunks |
| Query Processing | 1000-2000 ms | 500-1000 ms | 200-500 ms |
| OCR Processing | 3-5 sec/image | 2-3 sec/image ((Handwritten would take longer)) | 1-2 sec/image (Handwritten would take longer) |
| Memory Usage | 250Mb - 1gb | 1-4 GB | 4-8 GB |

### Throughput Metrics

| Metric | Value | Optimization Target |
|--------|-------|-------------------|
| Queries per Second | 10-50 QPS | 50-100 QPS |
| Concurrent Users | 50-100 | 100-500 |
| Document Processing | 100-200 docs/min | 200-500 docs/min |

## Configuration Tuning

### Optimal Configuration Settings

```python
# High-performance configuration
high_perf_config = Config(
    # Document Processing
    CHUNK_SIZE=800,          # Balanced chunk size
    CHUNK_OVERLAP=100,       # Moderate overlap
    CHUNKER_TYPE=ChunkerType.TEXT,
    
    # Vector Store
    VECTOR_STORE_TYPE=VectorStoreType.HYBRID_ONLINE,
    FAISS_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2",
    
    # Retrieval
    TOP_K=8,                 # Optimal result count
    RETRIEVER_TYPE=RetrieverType.SIMPLE,  # Faster than hybrid
    
    # Performance
    BATCH_SIZE=64,           # Memory-efficient batch size
    ENABLE_CACHE=True,
    CACHE_DIR="/opt/kssrag/cache",
    
    # Memory Management
    MAX_DOCS_FOR_TESTING=None,  # Process all documents
)
```

### Use Case Specific Configurations

**Technical Documentation:**
```python
tech_docs_config = Config(
    CHUNK_SIZE=600,          # Smaller chunks for precision
    CHUNK_OVERLAP=75,        # Moderate overlap
    VECTOR_STORE_TYPE=VectorStoreType.BM25S,  # Keyword-focused
    TOP_K=6,                 # Fewer, more precise results
    BATCH_SIZE=32            # Conservative memory usage
)
```

**Research Papers:**
```python
research_config = Config(
    CHUNK_SIZE=1200,         # Larger chunks for context
    CHUNK_OVERLAP=150,       # Higher overlap for continuity
    VECTOR_STORE_TYPE=VectorStoreType.FAISS,  # Semantic search
    TOP_K=10,                # More comprehensive results
    BATCH_SIZE=48            # Balanced processing
)
```

**Enterprise Knowledge Base:**
```python
enterprise_config = Config(
    CHUNK_SIZE=1000,         # Balanced approach
    CHUNK_OVERLAP=100,       # Standard overlap
    VECTOR_STORE_TYPE=VectorStoreType.HYBRID_ONLINE,  # Best of both
    TOP_K=8,                 # Optimal result count
    BATCH_SIZE=64,           # Efficient processing
    ENABLE_CACHE=True,
    CACHE_DIR="/shared/cache"  # Shared cache for multiple instances
)
```

## Vector Store Optimization

### BM25/BMS25S Optimization

```python
# BM25-specific optimizations
class OptimizedBM25VectorStore(BM25VectorStore):
    def __init__(self, persist_path: Optional[str] = None):
        super().__init__(persist_path)
        self.optimization_settings = {
            'k1': 1.5,       # Term frequency saturation
            'b': 0.75,       # Document length normalization
            'epsilon': 0.25  # IDF smoothing
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """Optimized tokenization for BM25"""
        # Advanced preprocessing
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        tokens = text.split()
        
        # Filter short tokens and stop words
        tokens = [token for token in tokens if len(token) > 2]
        
        return tokens
```

### FAISS Optimization

```python
# FAISS performance tuning
class OptimizedFAISSVectorStore(FAISSVectorStore):
    def __init__(self, persist_path: Optional[str] = None, model_name: Optional[str] = None):
        super().__init__(persist_path, model_name)
        
        # Use more efficient index type
        self.index = faiss.IndexIVFFlat(
            faiss.IndexFlatL2(self.dimension),
            self.dimension,
            min(4096, max(100, len(documents) // 10)),  # Adaptive nlist
            faiss.METRIC_L2
        )
        
        # Enable GPU if available
        if faiss.get_num_gpus() > 0:
            self.res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index)
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Optimized document addition with training"""
        self.documents = documents
        self.doc_texts = [doc["content"] for doc in documents]
        
        # Generate embeddings in optimized batches
        embeddings = []
        optimal_batch_size = min(self.config.BATCH_SIZE, 128)  # Cap batch size
        
        for i in range(0, len(self.doc_texts), optimal_batch_size):
            batch_texts = self.doc_texts[i:i+optimal_batch_size]
            batch_embeddings = self.model.encode(
                batch_texts, 
                batch_size=optimal_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True  # Important for cosine similarity
            )
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings).astype('float32')
        
        # Train index for better performance
        if not self.index.is_trained:
            self.index.train(embeddings)
        
        self.index.add(embeddings)
```

### Hybrid Store Optimization

```python
# Optimized hybrid retrieval
class OptimizedHybridVectorStore(HybridVectorStore):
    def __init__(self, persist_path: Optional[str] = "optimized_hybrid_index"):
        super().__init__(persist_path)
        self.alpha = 0.6  # Weight BM25 higher for speed
        self.cache = {}    # Query result caching
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Optimized hybrid retrieval with caching"""
        
        # Check cache first
        cache_key = hash(query + str(top_k))
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Parallel retrieval from both stores
        with ThreadPoolExecutor(max_workers=2) as executor:
            bm25_future = executor.submit(self.bm25_store.retrieve, query, top_k * 3)
            faiss_future = executor.submit(self.faiss_store.retrieve, query, top_k * 3)
            
            bm25_results = bm25_future.result()
            faiss_results = faiss_future.result()
        
        # Efficient deduplication
        seen_contents = set()
        combined_results = []
        
        for doc in bm25_results + faiss_results:
            content_hash = hash(doc["content"])
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                combined_results.append(doc)
        
        # Smart scoring and ranking
        scored_results = self._score_and_rank(combined_results, query, top_k)
        
        # Cache results
        self.cache[cache_key] = scored_results[:top_k]
        
        return scored_results[:top_k]
```

## Memory Management

### Efficient Document Processing

```python
class MemoryEfficientProcessor:
    """Memory-optimized document processing"""
    
    def __init__(self, config: Config):
        self.config = config
        self.processed_chunks = 0
        
    def process_large_document(self, file_path: str, chunk_callback=None):
        """
        Process large documents with minimal memory usage.
        
        Args:
            file_path: Path to document
            chunk_callback: Function to call for each chunk
        """
        chunk_size = self.config.CHUNK_SIZE
        overlap = self.config.CHUNK_OVERLAP
        
        with open(file_path, 'r', encoding='utf-8') as file:
            buffer = ""
            
            while True:
                # Read in chunks to avoid loading entire file
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                    
                buffer += chunk
                
                # Process when we have enough content
                while len(buffer) >= chunk_size:
                    process_chunk = buffer[:chunk_size]
                    buffer = buffer[chunk_size - overlap:]
                    
                    # Process and yield chunk
                    processed = self.process_chunk(process_chunk)
                    self.processed_chunks += 1
                    
                    if chunk_callback:
                        chunk_callback(processed)
                    
                    # Optional: garbage collection
                    if self.processed_chunks % 100 == 0:
                        gc.collect()
```

### Cache Optimization

```python
class SmartCache:
    """Intelligent caching system for vector stores"""
    
    def __init__(self, max_size_mb: int = 1024, ttl_seconds: int = 3600):
        self.max_size_mb = max_size_mb
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.total_size = 0
        
    def get(self, key):
        """Get item from cache with LRU eviction"""
        if key in self.cache:
            # Update access time
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, key, value, size_mb: float):
        """Set item in cache with size awareness"""
        
        # Evict if necessary
        while self.total_size + size_mb > self.max_size_mb and self.cache:
            self._evict_oldest()
        
        self.cache[key] = value
        self.access_times[key] = time.time()
        self.total_size += size_mb
    
    def _evict_oldest(self):
        """Evict least recently used item"""
        if not self.access_times:
            return
            
        oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        oldest_size = self._estimate_size(self.cache[oldest_key])
        
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
        self.total_size -= oldest_size
    
    def _estimate_size(self, obj):
        """Estimate memory size of object"""
        return len(pickle.dumps(obj)) / (1024 * 1024)  # MB
```

## Query Optimization

### Query Preprocessing

```python
class QueryOptimizer:
    """Optimize queries for better retrieval performance"""
    
    def __init__(self):
        self.stop_words = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'
        ])
    
    def optimize_query(self, query: str) -> str:
        """
        Apply optimizations to improve query performance.
        
        Args:
            query: Original user query
            
        Returns:
            Optimized query string
        """
        # Basic cleaning
        query = query.lower().strip()
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query)
        
        # Expand abbreviations
        query = self._expand_abbreviations(query)
        
        # Remove stop words (context-dependent)
        if len(query.split()) > 4:  # Only for longer queries
            words = query.split()
            words = [w for w in words if w not in self.stop_words]
            query = ' '.join(words)
        
        return query
    
    def _expand_abbreviations(self, query: str) -> str:
        """Expand common abbreviations"""
        abbreviations = {
            'ai': 'artificial intelligence',
            'ml': 'machine learning',
            'api': 'application programming interface',
            'cpu': 'central processing unit',
            'gpu': 'graphics processing unit',
        }
        
        for abbr, full in abbreviations.items():
            query = re.sub(r'\b' + abbr + r'\b', full, query)
        
        return query
```

### Parallel Processing

```python
class ParallelRetriever:
    """Parallel document retrieval for improved performance"""
    
    def __init__(self, vector_stores: List[BaseVectorStore], max_workers: int = None):
        self.vector_stores = vector_stores
        self.max_workers = max_workers or min(len(vector_stores), 4)
    
    def retrieve_parallel(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve from multiple vector stores in parallel.
        
        Args:
            query: Search query
            top_k: Results per store
            
        Returns:
            Combined and ranked results
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit retrieval tasks
            future_to_store = {
                executor.submit(store.retrieve, query, top_k): store
                for store in self.vector_stores
            }
            
            # Collect results
            all_results = []
            for future in as_completed(future_to_store):
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    logging.error(f"Retrieval error: {e}")
        
        # Deduplicate and rank
        return self._rank_results(all_results, query, top_k)
    
    def _rank_results(self, results: List[Dict], query: str, top_k: int):
        """Rank results using multiple factors"""
        scored_results = []
        
        for doc in results:
            score = self._calculate_score(doc, query)
            scored_results.append((doc, score))
        
        # Sort by score and return top_k
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_results[:top_k]]
    
    def _calculate_score(self, doc: Dict, query: str) -> float:
        """Calculate comprehensive relevance score"""
        content = doc["content"].lower()
        query_terms = query.lower().split()
        
        # Term frequency score
        term_score = sum(content.count(term) for term in query_terms)
        
        # Position score (earlier mentions are better)
        first_position = min(
            [content.find(term) for term in query_terms if term in content] 
            or [len(content)]
        )
        position_score = 1.0 / (first_position + 1)
        
        # Length normalization
        length_penalty = min(1.0, 1000 / len(content))
        
        return term_score * position_score * length_penalty
```

## Monitoring and Profiling

### Performance Monitoring

```python
class PerformanceMonitor:
    """Comprehensive performance monitoring"""
    
    def __init__(self):
        self.metrics = {
            'query_times': [],
            'document_processing_times': [],
            'memory_usage': [],
            'cache_hit_rates': []
        }
        self.start_time = time.time()
    
    def record_query_time(self, duration: float):
        """Record query processing time"""
        self.metrics['query_times'].append(duration)
        
        # Keep only recent measurements
        if len(self.metrics['query_times']) > 1000:
            self.metrics['query_times'] = self.metrics['query_times'][-1000:]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        return {
            'uptime_seconds': time.time() - self.start_time,
            'query_performance': {
                'total_queries': len(self.metrics['query_times']),
                'average_time': np.mean(self.metrics['query_times']),
                'p95_time': np.percentile(self.metrics['query_times'], 95),
                'p99_time': np.percentile(self.metrics['query_times'], 99),
            },
            'memory_usage': {
                'current_mb': self._get_current_memory(),
                'peak_mb': max(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0,
            },
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        avg_query_time = np.mean(self.metrics['query_times'])
        if avg_query_time > 2.0:  # seconds
            recommendations.append("Consider reducing CHUNK_SIZE or TOP_K")
        
        current_memory = self._get_current_memory()
        if current_memory > 4000:  # MB
            recommendations.append("Consider reducing BATCH_SIZE or enabling more aggressive caching")
        
        return recommendations
    
    def _get_current_memory(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
```

### Profiling Tools

```python
def profile_rag_system():
    """Comprehensive profiling of RAG system"""
    
    # CPU profiling
    import cProfile
    import pstats
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run typical workload
    rag = KSSRAG()
    rag.load_document("large_document.pdf")
    
    for i in range(100):
        rag.query(f"test query {i}")
    
    profiler.disable()
    
    # Generate report
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
    
    # Memory profiling
    from memory_profiler import profile
    
    @profile
    def memory_intensive_operation():
        rag = KSSRAG()
        rag.load_document("large_document.pdf")
        return rag.query("complex query")
    
    memory_intensive_operation()
```

## Scaling Strategies

### Horizontal Scaling

```python
class DistributedRAG:
    """Distributed RAG system for horizontal scaling"""
    
    def __init__(self, instances: List[KSSRAG], load_balancer):
        self.instances = instances
        self.load_balancer = load_balancer
        self.instance_metrics = {i: [] for i in range(len(instances))}
    
    def query(self, question: str, top_k: int = 5) -> str:
        """Distributed query processing"""
        # Select instance using load balancer
        instance_id = self.load_balancer.select_instance()
        instance = self.instances[instance_id]
        
        start_time = time.time()
        
        try:
            result = instance.query(question, top_k)
            
            # Record success
            duration = time.time() - start_time
            self.instance_metrics[instance_id].append({
                'timestamp': time.time(),
                'duration': duration,
                'success': True
            })
            
            return result
            
        except Exception as e:
            # Record failure
            self.instance_metrics[instance_id].append({
                'timestamp': time.time(),
                'duration': time.time() - start_time,
                'success': False,
                'error': str(e)
            })
            raise
    
    def get_load_metrics(self) -> Dict[int, Dict]:
        """Get load metrics for each instance"""
        metrics = {}
        
        for instance_id, records in self.instance_metrics.items():
            if records:
                recent_records = [r for r in records if time.time() - r['timestamp'] < 300]  # Last 5 minutes
                
                metrics[instance_id] = {
                    'request_count': len(recent_records),
                    'success_rate': sum(1 for r in recent_records if r['success']) / len(recent_records),
                    'average_duration': np.mean([r['duration'] for r in recent_records]),
                    'current_load': len([r for r in recent_records if time.time() - r['timestamp'] < 60])  # Last minute
                }
        
        return metrics
```

This performance tuning guide provides comprehensive strategies for optimizing KSS RAG across various dimensions. Implement these techniques based on your specific workload and performance requirements.

---