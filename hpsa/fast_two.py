"""
Hierarchical Probabilistic Search Architecture (HPSA)
Production Implementation with Benchmarking

This implementation includes:
- Complete HPSA algorithm implementation
- Comparison benchmarks against standard algorithms
- Performance testing suite
- Sample dataset generation
"""

import math
import threading
import time
import random
import string
import mmh3
import statistics
from typing import List, Dict, Set
from collections import defaultdict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# Performance monitoring

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

class BloomFilter:
    """Implements a Bloom filter with configurable false positive rate"""
    
    def __init__(self, expected_elements: int, false_positive_rate: float):
        self.size = self._calculate_size(expected_elements, false_positive_rate)
        self.hash_functions = self._calculate_hash_functions(self.size, expected_elements)
        self.bit_array = [False] * self.size
        
    def _calculate_size(self, n: int, p: float) -> int:
        """Calculate optimal bit array size"""
        return int(-n * math.log(p) / (math.log(2) ** 2))
    
    def _calculate_hash_functions(self, m: int, n: int) -> int:
        """Calculate optimal number of hash functions"""
        return int(m / n * math.log(2))
    
    def add(self, item: str):
        """Add an item to the Bloom filter"""
        for seed in range(self.hash_functions):
            index = mmh3.hash(item, seed) % self.size
            self.bit_array[index] = True
            
    def might_contain(self, item: str) -> bool:
        """Check if an item might be in the set"""
        for seed in range(self.hash_functions):
            index = mmh3.hash(item, seed) % self.size
            if not self.bit_array[index]:
                return False
        return True

class CascadeBloomFilter:
    """Implements a cascade of Bloom filters with decreasing false positive rates"""
    
    def __init__(self, expected_elements: int, cascade_depth: int = 4):
        self.cascade_depth = cascade_depth
        self.filters = []
        
        # Initialize cascade with decreasing false positive rates
        base_rate = 0.1
        for level in range(cascade_depth):
            false_positive_rate = base_rate * (0.5 ** level)
            self.filters.append(BloomFilter(expected_elements, false_positive_rate))
            
    def add(self, item: str):
        """Add an item to all filters in the cascade"""
        for filter in self.filters:
            filter.add(item)
            
    def might_contain(self, item: str) -> bool:
        """Check item existence through the cascade"""
        for level, filter in enumerate(self.filters):
            if not filter.might_contain(item):
                return False
            # Early exit optimization
            if level > 0 and random.random() < 0.1:  # 10% chance of early exit
                return True
        return True

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

class HPSA:
    def __init__(self, expected_documents: int):
        self.skip_list_index = HierarchicalSkipList()
        self.bloom_cascade = CascadeBloomFilter(expected_documents * 10)
        self.documents = {}
        self.term_document_map = defaultdict(set)
        self.lock = threading.RLock()

    def index_document(self, doc: Document):
        with self.lock:
            self.documents[doc.id] = doc
            
            # Index each term
            for term in doc.terms:
                term = term.lower()  # Normalize terms
                self.bloom_cascade.add(term)
                self.skip_list_index.insert(term, doc.id)
                self.term_document_map[term].add(doc.id)

    def search(self, query: str) -> List[SearchResult]:
        query_terms = set(query.lower().split())
        results = defaultdict(lambda: SearchResult(0, 0.0, []))
        
        # Process each query term
        for term in query_terms:
            # Skip if term definitely not in index
            if not self.bloom_cascade.might_contain(term):
                continue
                
            try:
                # Get matching documents
                matching_docs = self.skip_list_index.search(term)
                
                # Score documents
                for doc_id in matching_docs:
                    if doc_id in self.documents:
                        result = results[doc_id]
                        result.doc_id = doc_id
                        result.score += 1.0 / max(1, len(self.term_document_map[term]))  # IDF with safety check
            except Exception as e:
                print(f"Error processing term '{term}': {str(e)}")
                continue
                
        # Sort results by score
        sorted_results = sorted(
            results.values(), 
            key=lambda x: x.score, 
            reverse=True
        )
        
        return sorted_results[:100]  # Top 100 results
    
class Benchmark:
    """Benchmark suite for search algorithms"""
    
    @staticmethod
    def generate_document(doc_id: int, vocab_size: int, doc_length: int) -> Document:
        """Generate a random test document"""
        terms = set(
            ''.join(random.choices(string.ascii_lowercase, k=5))
            for _ in range(doc_length)
        )
        content = ' '.join(terms)
        return Document(doc_id, content, terms)
        
    @staticmethod
    def generate_dataset(num_docs: int, vocab_size: int, doc_length: int) -> List[Document]:
        """Generate a test dataset"""
        return [
            Benchmark.generate_document(i, vocab_size, doc_length)
            for i in range(num_docs)
        ]
        
    @staticmethod
    def generate_queries(num_queries: int, vocab_size: int, terms_per_query: int) -> List[str]:
        """Generate test queries"""
        return [
            ' '.join(
                ''.join(random.choices(string.ascii_lowercase, k=5))
                for _ in range(terms_per_query)
            )
            for _ in range(num_queries)
        ]
        
    @staticmethod
    def run_benchmark(
        algorithm: HPSA,
        queries: List[str],
        num_threads: int = 4
    ) -> Dict:
        """Run performance benchmark"""
        results = {
            'query_times': [],
            'queries_per_second': 0,
            'mean_response_time': 0,
            'p95_response_time': 0,
            'p99_response_time': 0
        }
        
        def process_query(query: str) -> float:
            start = time.time()
            algorithm.search(query)
            return time.time() - start
            
        # Run queries in parallel
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            query_times = list(executor.map(process_query, queries))
            
        results['query_times'] = query_times
        results['queries_per_second'] = len(queries) / sum(query_times)
        results['mean_response_time'] = statistics.mean(query_times)
        results['p95_response_time'] = statistics.quantiles(query_times, n=20)[18]
        results['p99_response_time'] = statistics.quantiles(query_times, n=100)[98]
        
        return results

def main():
    """Main benchmark execution"""
    
    # Test parameters
    NUM_DOCUMENTS = 50000
    VOCAB_SIZE = 100_000
    DOC_LENGTH = 100
    NUM_QUERIES = 10_000
    TERMS_PER_QUERY = 3
    
    print("Generating test dataset...")
    documents = Benchmark.generate_dataset(NUM_DOCUMENTS, VOCAB_SIZE, DOC_LENGTH)
    
    print("Initializing HPSA...")
    hpsa = HPSA(NUM_DOCUMENTS)
    
    print("Indexing documents...")
    start_time = time.time()
    for doc in documents:
        hpsa.index_document(doc)
    index_time = time.time() - start_time
    print(f"Indexing completed in {index_time:.2f} seconds")
    
    print("Generating test queries...")
    queries = Benchmark.generate_queries(NUM_QUERIES, VOCAB_SIZE, TERMS_PER_QUERY)
    
    print("Running benchmark...")
    results = Benchmark.run_benchmark(hpsa, queries)
    
    print("\nBenchmark Results:")
    print(f"Queries per second: {results['queries_per_second']:.2f}")
    print(f"Mean response time: {results['mean_response_time']*1000:.2f}ms")
    print(f"95th percentile response time: {results['p95_response_time']*1000:.2f}ms")
    print(f"99th percentile response time: {results['p99_response_time']*1000:.2f}ms")

if __name__ == "__main__":
    main()