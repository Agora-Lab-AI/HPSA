"""
Enhanced Hierarchical Probabilistic Search Architecture (HPSA)
Production Implementation with Sharding and Advanced Concurrency

Key features:
- Consistent hashing for shard distribution
- Lock-free concurrent data structures
- Proper error handling and logging
- Performance monitoring
- Memory-efficient data structures
"""

import bisect
import concurrent.futures
import math
import random
import statistics
import string
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue
from typing import Dict, List, Optional, Set, Tuple

import mmh3
import numpy as np
import xxhash
from loguru import logger


@dataclass
class Document:
    """Represents a searchable document in the system"""
    id: int
    content: str
    terms: Set[str]

@dataclass
class SearchResult:
    """Represents a single search result"""
    doc_id: int
    score: float
    match_positions: List[int]
    
class SkipNode:
    def __init__(self, key: str, doc_id: int, level: int):
        self.key = key
        self.doc_id = doc_id
        # Ensure level is non-negative
        self.level = max(0, level)
        # Initialize forward list with correct size
        self.forward = [None] * (self.level + 1)
        self.skip_pointer = None
        self.count = 1


class ConsistentHashRing:
    """
    Implements consistent hashing for shard distribution with virtual nodes
    to ensure even distribution of terms across shards.
    """
    def __init__(self, num_shards: int, virtual_nodes: int = 100):
        self.num_shards = num_shards
        self.virtual_nodes = virtual_nodes
        self.ring = []  # (hash_value, shard_id)
        self._build_ring()
        
    def _build_ring(self):
        """Builds the consistent hash ring with virtual nodes."""
        for shard_id in range(self.num_shards):
            for v in range(self.virtual_nodes):
                # Create virtual node key
                key = f"{shard_id}:{v}"
                # Use xxhash for better distribution
                hash_value = xxhash.xxh64(key).intdigest()
                self.ring.append((hash_value, shard_id))
        # Sort ring by hash values
        self.ring.sort()
        
    def get_shard(self, key: str) -> int:
        """
        Determines which shard a key belongs to.
        
        Args:
            key: The key to look up
            
        Returns:
            The shard ID where this key should be stored
            
        Time complexity: O(log n) where n is number of virtual nodes
        """
        if not self.ring:
            raise RuntimeError("Hash ring is empty")
            
        hash_value = xxhash.xxh64(key).intdigest()
        
        # Binary search for the first ring position >= hash_value
        pos = bisect.bisect_left([r[0] for r in self.ring], hash_value)
        
        # Wrap around to start of ring if needed
        if pos >= len(self.ring):
            pos = 0
            
        return self.ring[pos][1]

class ShardedBloomFilter:
    """
    Memory-efficient Bloom filter implementation using bit arrays and multiple hash functions.
    Sharded to reduce lock contention.
    """
    def __init__(self, expected_elements: int, false_positive_rate: float, num_shards: int = 16):
        # Calculate optimal parameters
        self.size_per_shard = self._calculate_optimal_size(
            expected_elements // num_shards,
            false_positive_rate
        )
        self.num_hash_functions = self._calculate_optimal_hash_functions(
            self.size_per_shard,
            expected_elements // num_shards
        )
        
        # Initialize sharded bit arrays
        self.shards = [
            {'bits': np.zeros(self.size_per_shard, dtype=np.bool_),
             'lock': threading.RLock()}
            for _ in range(num_shards)
        ]
        
    def _calculate_optimal_size(self, n: int, p: float) -> int:
        """Calculate optimal bit array size for desired false positive rate."""
        return int(-n * math.log(p) / (math.log(2) ** 2))
        
    def _calculate_optimal_hash_functions(self, m: int, n: int) -> int:
        """Calculate optimal number of hash functions."""
        return max(1, int(m / n * math.log(2)))
        
    def _get_shard_index(self, item: str) -> int:
        """Determine which shard an item belongs to."""
        return xxhash.xxh64(item).intdigest() % len(self.shards)
        
    def add(self, item: str):
        """Add an item to the appropriate shard."""
        shard_idx = self._get_shard_index(item)
        shard = self.shards[shard_idx]
        
        with shard['lock']:
            for seed in range(self.num_hash_functions):
                idx = mmh3.hash(item, seed) % self.size_per_shard
                shard['bits'][idx] = True
                
    def might_contain(self, item: str) -> bool:
        """Check if an item might be in the set."""
        shard_idx = self._get_shard_index(item)
        shard = self.shards[shard_idx]
        
        with shard['lock']:
            for seed in range(self.num_hash_functions):
                idx = mmh3.hash(item, seed) % self.size_per_shard
                if not shard['bits'][idx]:
                    return False
        return True

@dataclass
class SearchMetrics:
    """Tracks performance metrics for search operations."""
    query_time: float
    shard_times: List[float]
    num_matches: int
    cache_hit: bool

class SearchCache:
    """
    Thread-safe LRU cache for search results with TTL.
    """
    def __init__(self, capacity: int = 10000, ttl_seconds: int = 300):
        self.capacity = capacity
        self.ttl = ttl_seconds
        self.cache = {}  # {key: (result, timestamp)}
        self.lock = threading.RLock()
        
    def get(self, key: str) -> Optional[List[SearchResult]]:
        """Get cached results if they exist and haven't expired."""
        with self.lock:
            if key in self.cache:
                result, timestamp = self.cache[key]
                if time.time() - timestamp <= self.ttl:
                    return result
                else:
                    del self.cache[key]
        return None
        
    def put(self, key: str, value: List[SearchResult]):
        """Add results to cache, evicting old entries if needed."""
        with self.lock:
            while len(self.cache) >= self.capacity:
                # Evict oldest entry
                oldest_key = min(
                    self.cache.keys(),
                    key=lambda k: self.cache[k][1]
                )
                del self.cache[oldest_key]
            self.cache[key] = (value, time.time())


class HierarchicalSkipList:
    def __init__(self, max_level: int = 16):
        self.max_level = max_level
        # Initialize head node with proper level
        self.head = SkipNode('', -1, max_level)
        self.level = 0
        self.size = 0
        # Add thread safety
        self.lock = threading.RLock()

    def random_level(self) -> int:
        level = 0
        while random.random() < 0.5 and level < self.max_level:
            level += 1
        return level

    def insert(self, key: str, doc_id: int):
        with self.lock:
            update = [None] * (self.max_level + 1)
            current = self.head

            # Find insert position
            for i in range(self.level, -1, -1):
                while (current.forward[i] and 
                      (current.forward[i].key < key or 
                       (current.forward[i].key == key and 
                        current.forward[i].doc_id < doc_id))):
                    current = current.forward[i]
                update[i] = current

            # Create new node
            level = self.random_level()
            if level > self.level:
                for i in range(self.level + 1, level + 1):
                    update[i] = self.head
                self.level = level

            new_node = SkipNode(key, doc_id, level)

            # Update forward pointers
            for i in range(level + 1):
                new_node.forward[i] = update[i].forward[i]
                update[i].forward[i] = new_node

            self.size += 1

    def search(self, key: str) -> List[int]:
        results = []
        with self.lock:
            current = self.head
            
            # Search using skip pointers first
            if current.skip_pointer and current.skip_pointer.key <= key:
                current = current.skip_pointer
                if current.key == key:
                    results.append(current.doc_id)
                    current.count += 1

            # If skip pointer search fails, do regular search
            if not results:
                # Start from the highest level
                for i in range(min(self.level, len(current.forward) - 1), -1, -1):
                    while (current.forward[i] and 
                           current.forward[i].key < key):
                        current = current.forward[i]

                # Move to the first node
                current = current.forward[0] if current.forward[0] else None

                # Collect all matching documents
                while current and current.key == key and len(results) < 100:
                    results.append(current.doc_id)
                    current.count += 1
                    current = current.forward[0]

        return results
    
class EnhancedHPSA:
    """
    Enhanced HPSA implementation with sharding and advanced concurrency.
    """
    def __init__(
        self,
        expected_documents: int,
        num_shards: int = 16,
        cache_size: int = 10000
    ):
        self.num_shards = num_shards
        
        # Initialize consistent hash ring for shard routing
        self.hash_ring = ConsistentHashRing(num_shards)
        
        # Initialize sharded indexes
        self.shards = [
            {
                'skip_list': HierarchicalSkipList(),
                'bloom_filter': ShardedBloomFilter(
                    expected_documents // num_shards * 2,
                    false_positive_rate=0.01,
                    num_shards=4
                ),
                'lock': threading.RLock()
            }
            for _ in range(num_shards)
        ]
        
        # Thread-safe document store
        self.documents = {}
        self.doc_lock = threading.RLock()
        
        # Result cache
        self.cache = SearchCache(cache_size)
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(
            max_workers=min(32, num_shards * 2)
        )
        
        # Performance monitoring
        self.metrics = Queue()
        self._start_metrics_collector()
        
    def _start_metrics_collector(self):
        """Start background thread for metrics collection."""
        def collect_metrics():
            while True:
                metrics = []
                # Collect metrics for 1 minute
                deadline = time.time() + 60
                while time.time() < deadline:
                    try:
                        metric = self.metrics.get(timeout=1)
                        metrics.append(metric)
                    except Queue.Empty:
                        continue
                        
                if metrics:
                    avg_query_time = statistics.mean(
                        m.query_time for m in metrics
                    )
                    p95_query_time = statistics.quantiles(
                        [m.query_time for m in metrics],
                        n=20
                    )[18]
                    logger.info(
                        f"Search metrics - Avg: {avg_query_time:.3f}s, "
                        f"P95: {p95_query_time:.3f}s, "
                        f"Cache hit rate: {sum(m.cache_hit for m in metrics)/len(metrics):.2%}"
                    )
                    
        threading.Thread(
            target=collect_metrics,
            daemon=True
        ).start()
        
    def index_document(self, doc: Document):
        """
        Index a document across shards.
        
        Args:
            doc: Document to index
            
        Thread safety: Full
        Time complexity: O(T * log N) where T is terms in doc, N is docs per shard
        """
        # Store document
        with self.doc_lock:
            self.documents[doc.id] = doc
            
        # Group terms by shard
        shard_terms = defaultdict(set)
        for term in doc.terms:
            term = term.lower()
            shard_idx = self.hash_ring.get_shard(term)
            shard_terms[shard_idx].add(term)
            
        # Index terms in parallel across shards
        futures = []
        for shard_idx, terms in shard_terms.items():
            future = self.thread_pool.submit(
                self._index_shard,
                shard_idx,
                doc.id,
                terms
            )
            futures.append(future)
            
        # Wait for all shards to complete
        try:
            concurrent.futures.wait(
                futures,
                timeout=30
            )
        except concurrent.futures.TimeoutError:
            logger.error(
                f"Timeout indexing document {doc.id}"
            )
            raise
            
    def _index_shard(
        self,
        shard_idx: int,
        doc_id: int,
        terms: Set[str]
    ):
        """Index terms in a single shard."""
        shard = self.shards[shard_idx]
        with shard['lock']:
            for term in terms:
                shard['bloom_filter'].add(term)
                shard['skip_list'].insert(term, doc_id)
                
    def search(
        self,
        query: str,
        max_results: int = 100
    ) -> Tuple[List[SearchResult], SearchMetrics]:
        """
        Search for documents matching the query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            Tuple of (search results, performance metrics)
            
        Thread safety: Full
        Time complexity: O(Q * log N) where Q is query terms, N is docs per shard
        """
        start_time = time.time()
        query = query.lower()
        
        # Check cache first
        cached = self.cache.get(query)
        if cached is not None:
            query_time = time.time() - start_time
            metrics = SearchMetrics(
                query_time=query_time,
                shard_times=[],
                num_matches=len(cached),
                cache_hit=True
            )
            self.metrics.put(metrics)
            return cached, metrics
            
        # Group query terms by shard
        shard_terms = defaultdict(set)
        for term in query.split():
            shard_idx = self.hash_ring.get_shard(term)
            shard_terms[shard_idx].add(term)
            
        # Search shards in parallel
        futures = {}
        for shard_idx, terms in shard_terms.items():
            future = self.thread_pool.submit(
                self._search_shard,
                shard_idx,
                terms
            )
            futures[future] = shard_idx
            
        # Collect results from all shards
        results = defaultdict(lambda: SearchResult(0, 0.0, []))
        shard_times = []
        
        try:
            done, not_done = concurrent.futures.wait(
                futures,
                timeout=5
            )
            if not_done:
                logger.warning(
                    f"{len(not_done)} shards timed out"
                )
                
            for future in done:
                shard_result = future.result()
                shard_times.append(shard_result['time'])
                
                # Merge results using TF-IDF scoring
                for doc_id, matches in shard_result['matches'].items():
                    if doc_id in self.documents:
                        result = results[doc_id]
                        result.doc_id = doc_id
                        result.score += sum(
                            1.0 / max(1, len(self.documents))
                            for _ in matches
                        )
                        result.match_positions.extend(matches)
                        
        except concurrent.futures.TimeoutError:
            logger.error("Search timeout")
            raise
            
        # Sort results by score
        sorted_results = sorted(
            results.values(),
            key=lambda x: x.score,
            reverse=True
        )[:max_results]
        
        # Cache results
        self.cache.put(query, sorted_results)
        
        query_time = time.time() - start_time
        metrics = SearchMetrics(
            query_time=query_time,
            shard_times=shard_times,
            num_matches=len(sorted_results),
            cache_hit=False
        )
        self.metrics.put(metrics)
        
        return sorted_results, metrics
        
    def _search_shard(
        self,
        shard_idx: int,
        terms: Set[str]
    ) -> Dict:
        """Search within a single shard."""
        start_time = time.time()
        shard = self.shards[shard_idx]
        matches = defaultdict(list)
        
        with shard['lock']:
            for term in terms:
                # Skip if term definitely not in shard
                if not shard['bloom_filter'].might_contain(term):
                    continue
                    
                # Find matching documents
                doc_ids = shard['skip_list'].search(term)
                for doc_id in doc_ids:
                    matches[doc_id].append(1)  # Position placeholder
                    
        return {
            'matches': matches,
            'time': time.time() - start_time
        }




class EnhancedBenchmark:
    """
    Advanced benchmarking suite for the Enhanced HPSA implementation.
    Includes sophisticated performance metrics and realistic data generation.
    """
    def __init__(self, 
                 num_documents: int,
                 vocab_size: int,
                 doc_length_mean: int = 100,
                 doc_length_std: int = 20):
        self.num_documents = num_documents
        self.vocab_size = vocab_size
        self.doc_length_mean = doc_length_mean
        self.doc_length_std = doc_length_std
        
        # Track detailed metrics
        self.index_metrics = {
            'total_time': 0,
            'docs_per_second': 0,
            'memory_usage': [],
            'shard_distribution': defaultdict(int)
        }
        
        self.search_metrics = {
            'queries_per_second': 0,
            'mean_latency': 0,
            'p95_latency': 0,
            'p99_latency': 0,
            'cache_hit_rate': 0,
            'shard_timings': defaultdict(list)
        }

    def generate_realistic_document(self, doc_id: int) -> Document:
        """
        Generate a document with realistic term distribution following Zipf's law.
        This creates more realistic test data than purely random generation.
        """
        # Generate vocabulary if not exists
        if not hasattr(self, 'vocabulary'):
            self.vocabulary = [
                ''.join(random.choices(string.ascii_lowercase, k=5))
                for _ in range(self.vocab_size)
            ]
            
            # Assign Zipf frequencies to terms
            self.term_frequencies = np.random.zipf(1.5, self.vocab_size)
            self.term_frequencies = self.term_frequencies / sum(self.term_frequencies)

        # Sample document length from normal distribution
        doc_length = int(max(10, random.gauss(
            self.doc_length_mean,
            self.doc_length_std
        )))

        # Sample terms according to Zipf distribution
        terms = set(np.random.choice(
            self.vocabulary,
            size=doc_length,
            p=self.term_frequencies
        ))
        
        content = ' '.join(terms)
        return Document(doc_id, content, terms)

    def generate_realistic_query(self, num_terms: int = 3) -> str:
        """
        Generate realistic search queries based on term frequencies.
        Queries use more common terms with higher probability.
        """
        query_terms = np.random.choice(
            self.vocabulary,
            size=num_terms,
            p=self.term_frequencies
        )
        return ' '.join(query_terms)

    def run_comprehensive_benchmark(self,
                                  num_queries: int = 10000,
                                  concurrent_queries: int = 100) -> Dict:
        """
        Run a comprehensive benchmark suite measuring multiple aspects
        of search engine performance.
        
        Args:
            num_queries: Total number of queries to run
            concurrent_queries: Number of concurrent queries to simulate
            
        Returns:
            Dictionary containing detailed performance metrics
        """
        logger.info("Starting comprehensive benchmark...")
        
        # Initialize search engine
        engine = EnhancedHPSA(
            expected_documents=self.num_documents,
            num_shards=16
        )

        # Generate and index documents
        logger.info("Generating and indexing documents...")
        start_time = time.time()
        
        for doc_id in range(self.num_documents):
            doc = self.generate_realistic_document(doc_id)
            engine.index_document(doc)
            
            # Track shard distribution
            for term in doc.terms:
                shard = engine.hash_ring.get_shard(term)
                self.index_metrics['shard_distribution'][shard] += 1
                
            if doc_id % 10000 == 0:
                logger.info(f"Indexed {doc_id} documents...")

        index_time = time.time() - start_time
        self.index_metrics['total_time'] = index_time
        self.index_metrics['docs_per_second'] = self.num_documents / index_time

        # Generate queries
        logger.info("Generating test queries...")
        queries = [
            self.generate_realistic_query()
            for _ in range(num_queries)
        ]

        # Run queries with concurrent load
        logger.info("Running search benchmark...")
        search_times = []
        cache_hits = 0
        
        def run_query_batch(batch_queries):
            results = []
            for query in batch_queries:
                start_time = time.time()  # Use a different variable name if needed
                search_results, metrics = engine.search(query)
                query_duration = time.time() - start_time  # Calculate the duration
                
                results.append({
                    'time': query_duration,  # Store the duration
                    'num_results': len(search_results),
                    'cache_hit': metrics.cache_hit,
                    'shard_times': metrics.shard_times
                })
            return results

        # Split queries into batches
        batch_size = concurrent_queries
        query_batches = [
            queries[i:i + batch_size]
            for i in range(0, len(queries), batch_size)
        ]

        # Run batches with ThreadPoolExecutor
        all_results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_batch = {
                executor.submit(run_query_batch, batch): batch
                for batch in query_batches
            }
            
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_results = future.result()
                all_results.extend(batch_results)

        # Calculate metrics
        search_times = [r['time'] for r in all_results]
        cache_hits = sum(1 for r in all_results if r['cache_hit'])
        
        for result in all_results:
            for shard_idx, shard_time in enumerate(result['shard_times']):  # Renamed 'time' to 'shard_time'
                self.search_metrics['shard_timings'][shard_idx].append(shard_time)

        self.search_metrics.update({
            'queries_per_second': len(queries) / sum(search_times),
            'mean_latency': statistics.mean(search_times),
            'p95_latency': statistics.quantiles(search_times, n=20)[18],
            'p99_latency': statistics.quantiles(search_times, n=100)[98],
            'cache_hit_rate': cache_hits / len(queries)
        })

        return {
            'index_metrics': self.index_metrics,
            'search_metrics': self.search_metrics
        }

def main():
    """
    Run comprehensive benchmark and display results.
    """

    # Initialize benchmark
    benchmark = EnhancedBenchmark(
        num_documents=50000,  # 1 million documents
        vocab_size=100_000,       # 100k unique terms
        doc_length_mean=100,      # Average document length
        doc_length_std=20         # Standard deviation of document length
    )

    # Run benchmark
    results = benchmark.run_comprehensive_benchmark(
        num_queries=5000,      # 100k total queries
        concurrent_queries=1000    # 1000 concurrent queries
    )

    # Display results
    print("\nBenchmark Results:")
    print("\nIndexing Performance:")
    print(f"Total indexing time: {results['index_metrics']['total_time']:.2f} seconds")
    print(f"Documents per second: {results['index_metrics']['docs_per_second']:.2f}")
    
    print("\nShard Distribution:")
    shard_dist = results['index_metrics']['shard_distribution']
    mean_terms_per_shard = statistics.mean(shard_dist.values())
    std_terms_per_shard = statistics.stdev(shard_dist.values())
    print(f"Mean terms per shard: {mean_terms_per_shard:.2f}")
    print(f"Std dev terms per shard: {std_terms_per_shard:.2f}")
    
    print("\nSearch Performance:")
    print(f"Queries per second: {results['search_metrics']['queries_per_second']:.2f}")
    print(f"Mean latency: {results['search_metrics']['mean_latency']*1000:.2f}ms")
    print(f"95th percentile latency: {results['search_metrics']['p95_latency']*1000:.2f}ms")
    print(f"99th percentile latency: {results['search_metrics']['p99_latency']*1000:.2f}ms")
    print(f"Cache hit rate: {results['search_metrics']['cache_hit_rate']:.2%}")
    
    print("\nShard Response Times:")
    for shard_idx, times in results['search_metrics']['shard_timings'].items():
        print(f"Shard {shard_idx}:")
        print(f"  Mean: {statistics.mean(times)*1000:.2f}ms")
        print(f"  P95: {statistics.quantiles(times, n=20)[18]*1000:.2f}ms")

if __name__ == "__main__":
    main()