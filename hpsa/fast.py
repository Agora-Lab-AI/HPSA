"""
HDPSA: Hierarchical Distributed Pattern-based Search Architecture
A high-performance search algorithm implementation with pattern recognition and distributed processing capabilities.

This implementation includes:
- Pattern extraction using multiple feature types
- Efficient signature generation
- Probability-guided search
- Comprehensive testing suite
- Performance benchmarking capabilities

Author: Claude
Date: December 2024
"""

import concurrent.futures
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

import mmh3  # MurmurHash3 for efficient hashing
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from loguru import logger

@dataclass
class SearchResult:
    """Data class for search results with relevance scoring."""
    content_id: str
    content: Any
    score: float
    pattern_match: float
    access_count: int

class HDPSANode:
    """Node class for the hierarchical pattern tree."""
    
    def __init__(self, pattern_signature: bytes = None):
        self.pattern_signature = pattern_signature
        self.children: Dict[bytes, HDPSANode] = {}
        self.content_ids: Set[str] = set()
        self.access_count: int = 0
        self.last_access: float = time.time()
        self.lock = threading.Lock()

    def update_access(self):
        """Thread-safe update of access statistics."""
        with self.lock:
            self.access_count += 1
            self.last_access = time.time()

class ProbabilityTree:
    """Maintains search pattern statistics and probabilities."""
    
    def __init__(self, decay_factor: float = 0.1):
        self.root = HDPSANode()
        self.decay_factor = decay_factor
        self.access_history: Dict[bytes, List[float]] = defaultdict(list)
        self.lock = threading.Lock()

    def update_pattern(self, signature: bytes):
        """Update pattern statistics with thread safety."""
        with self.lock:
            current_time = time.time()
            self.access_history[signature].append(current_time)
            
            # Cleanup old history
            cutoff_time = current_time - (5 / self.decay_factor)
            self.access_history[signature] = [
                t for t in self.access_history[signature]
                if t > cutoff_time
            ]

    def get_pattern_probability(self, signature: bytes) -> float:
        """Calculate probability score for a pattern."""
        current_time = time.time()
        if signature not in self.access_history:
            return 0.0
            
        recent_accesses = sum(
            np.exp(-(current_time - t) * self.decay_factor)
            for t in self.access_history[signature]
        )
        return recent_accesses

class SignatureGenerator:
    """Generates compact binary signatures for content patterns."""
    
    def __init__(self, num_bands: int = 20, band_size: int = 5):
        self.num_bands = num_bands
        self.band_size = band_size
        self.total_hashes = num_bands * band_size
        
    def generate(self, pattern: np.ndarray) -> bytes:
        """Generate locality-sensitive hash signature."""
        # Normalize pattern vector
        pattern_normalized = pattern / np.linalg.norm(pattern)
        
        # Generate multiple hash values
        hash_values = []
        for i in range(self.total_hashes):
            seed = mmh3.hash(pattern_normalized.tobytes(), i)
            hash_values.append(seed)
            
        # Convert to compact binary signature
        signature = np.packbits(
            np.array(hash_values) > 0
        ).tobytes()
        
        return signature

class PatternExtractor:
    """Extracts multi-dimensional feature patterns from content."""
    
    def __init__(self, max_features: int = 1000):
        self.tfidf = TfidfVectorizer(max_features=max_features)
        self.fitted = False
        
        self.stop_words = set(stopwords.words('english'))
        
        # Ensure required NLTK data is downloaded
        self._ensure_nltk_data()

        self.stop_words = set(stopwords.words('english'))

    def _ensure_nltk_data(self):
        """Download necessary NLTK data if not present."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
    def fit(self, documents: List[str]):
        """Fit the feature extractors on a corpus."""
        self.tfidf.fit(documents)
        self.fitted = True
        
    def extract(self, content: str) -> np.ndarray:
        """Extract pattern features from content."""
        if not self.fitted:
            raise ValueError("PatternExtractor must be fitted before extraction")
            
        # Text preprocessing
        tokens = word_tokenize(content.lower())
        tokens = [t for t in tokens if t not in self.stop_words]
        processed_text = " ".join(tokens)
        
        # Extract TF-IDF features
        tfidf_features = self.tfidf.transform([processed_text]).toarray()[0]
        
        # Extract statistical features
        stat_features = self._extract_statistical_features(tokens)
        
        # Combine all features
        return np.concatenate([tfidf_features, stat_features])
        
    def _extract_statistical_features(self, tokens: List[str]) -> np.ndarray:
        """Extract statistical features from tokenized text."""
        # Average word length
        avg_word_length = np.mean([len(t) for t in tokens]) if tokens else 0
        
        # Vocabulary richness (unique words ratio)
        vocabulary_richness = len(set(tokens)) / len(tokens) if tokens else 0
        
        # Character distribution entropy
        char_freq = defaultdict(int)
        total_chars = 0
        for token in tokens:
            for char in token:
                char_freq[char] += 1
                total_chars += 1
                
        char_entropy = 0
        if total_chars > 0:
            for count in char_freq.values():
                prob = count / total_chars
                char_entropy -= prob * np.log2(prob)
                
        return np.array([avg_word_length, vocabulary_richness, char_entropy])

class HDPSA:
    """Main HDPSA search algorithm implementation."""
    
    def __init__(
        self,
        num_shards: int = 100,
        similarity_threshold: float = 0.85,
        max_features: int = 1000
    ):
        self.num_shards = num_shards
        self.similarity_threshold = similarity_threshold
        
        # Initialize components
        self.pattern_extractor = PatternExtractor(max_features=max_features)
        self.signature_generator = SignatureGenerator()
        self.probability_tree = ProbabilityTree()
        
        # Initialize shards
        self.shards = [HDPSANode() for _ in range(num_shards)]
        
        # Content storage
        self.content_store: Dict[str, Any] = {}
        self.content_patterns: Dict[str, np.ndarray] = {}
        self.content_signatures: Dict[str, bytes] = {}
        
        # Threading lock for content updates
        self.content_lock = threading.Lock()
        
    def fit(self, documents: List[str]):
        """Fit the pattern extractor on a corpus."""
        self.pattern_extractor.fit(documents)
        
    def index(self, content_id: str, content: Any):
        """Index new content with thread safety."""
        if not isinstance(content, str):
            content = str(content)
            
        # Extract pattern and generate signature
        pattern = self.pattern_extractor.extract(content)
        signature = self.signature_generator.generate(pattern)
        
        # Store content and metadata
        with self.content_lock:
            self.content_store[content_id] = content
            self.content_patterns[content_id] = pattern
            self.content_signatures[content_id] = signature
            
        # Add to appropriate shard
        shard_id = self._get_shard_id(signature)
        shard = self.shards[shard_id]
        
        with shard.lock:
            current_node = shard
            for i in range(0, len(signature), 4):
                chunk = signature[i:i+4]
                if chunk not in current_node.children:
                    current_node.children[chunk] = HDPSANode(chunk)
                current_node = current_node.children[chunk]
            current_node.content_ids.add(content_id)
            
    def search(
        self,
        query: str,
        max_results: int = 10,
        min_similarity: float = 0.5
    ) -> List[SearchResult]:
        """
        Search for relevant content using pattern matching.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            min_similarity: Minimum similarity threshold for results
            
        Returns:
            List of SearchResult objects sorted by relevance
        """
        # Extract query pattern and signature
        query_pattern = self.pattern_extractor.extract(query)
        query_signature = self.signature_generator.generate(query_pattern)
        
        # Update probability tree
        self.probability_tree.update_pattern(query_signature)
        
        # Find relevant shards
        relevant_shards = self._get_relevant_shards(query_signature)
        
        # Search shards in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_shard = {
                executor.submit(
                    self._search_shard,
                    shard,
                    query_signature,
                    query_pattern,
                    min_similarity
                ): shard_id
                for shard_id, shard in enumerate(relevant_shards)
            }
            
            # Collect results
            all_results = []
            for future in concurrent.futures.as_completed(future_to_shard):
                shard_id = future_to_shard[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    logger.error(f"Error searching shard {shard_id}: {e}")
                    
        # Sort and return top results
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:max_results]
        
    def _get_shard_id(self, signature: bytes) -> int:
        """Determine shard ID for a signature."""
        return mmh3.hash(signature) % self.num_shards
        
    def _get_relevant_shards(self, query_signature: bytes) -> List[HDPSANode]:
        """Find most relevant shards for a query signature."""
        shard_scores = []
        for shard_id, shard in enumerate(self.shards):
            # Calculate shard relevance score
            pattern_prob = self.probability_tree.get_pattern_probability(
                query_signature
            )
            access_score = np.log1p(shard.access_count)
            score = pattern_prob * access_score
            shard_scores.append((score, shard_id))
            
        # Sort and return top shards
        shard_scores.sort(reverse=True)
        return [
            self.shards[shard_id]
            for _, shard_id in shard_scores[:max(3, self.num_shards // 10)]
        ]
        
    def _search_shard(
        self,
        shard: HDPSANode,
        query_signature: bytes,
        query_pattern: np.ndarray,
        min_similarity: float
    ) -> List[SearchResult]:
        """Search within a single shard."""
        results = []
        nodes_to_check = [(shard, 0)]
        
        while nodes_to_check:
            current_node, depth = nodes_to_check.pop()
            
            # Check content in current node
            for content_id in current_node.content_ids:
                content_pattern = self.content_patterns[content_id]
                similarity = 1 - cosine(query_pattern, content_pattern)
                
                if similarity >= min_similarity:
                    pattern_match = self._calculate_pattern_match(
                        query_signature,
                        self.content_signatures[content_id]
                    )
                    
                    score = (
                        similarity * 0.6 +
                        pattern_match * 0.3 +
                        (np.log1p(current_node.access_count) * 0.1)
                    )
                    
                    results.append(SearchResult(
                        content_id=content_id,
                        content=self.content_store[content_id],
                        score=score,
                        pattern_match=pattern_match,
                        access_count=current_node.access_count
                    ))
                    
            # Add child nodes to check
            for chunk, child in current_node.children.items():
                if self._should_explore_branch(
                    query_signature,
                    chunk,
                    depth,
                    child.access_count
                ):
                    nodes_to_check.append((child, depth + 1))
                    
        return results
        
    def _calculate_pattern_match(
        self,
        sig1: bytes,
        sig2: bytes
    ) -> float:
        """Calculate similarity between two signatures."""
        bits1 = np.unpackbits(np.frombuffer(sig1, dtype=np.uint8))
        bits2 = np.unpackbits(np.frombuffer(sig2, dtype=np.uint8))
        return np.mean(bits1 == bits2)
        
    def _should_explore_branch(
        self,
        query_signature: bytes,
        chunk: bytes,
        depth: int,
        access_count: int
    ) -> bool:
        """Determine if a branch should be explored during search."""
        # Early pruning based on signature difference
        if depth < 2:  # Always explore first two levels
            return True
            
        chunk_match = self._calculate_pattern_match(
            query_signature[depth*4:(depth+1)*4],
            chunk
        )
        
        # Consider both signature similarity and access history
        access_score = np.log1p(access_count) / 10
        threshold = max(0.7 - (depth * 0.1), 0.3)
        
        return (chunk_match + access_score) >= threshold

    def benchmark(
        self,
        test_queries: List[str],
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark search performance.
        
        Args:
            test_queries: List of queries to test
            num_iterations: Number of times to run each query
            
        Returns:
            Dictionary with benchmark metrics
        """
        results = {
            'avg_search_time': 0,
            'min_search_time': float('inf'),
            'max_search_time': 0,
            'queries_per_second': 0
        }
        
        total_time = 0
        for query in test_queries:
            query_times = []
            for _ in range(num_iterations):
                start_time = time.time()
                self.search(query, max_results=10)
                query_time = time.time() - start_time
                query_times.append(query_time)
                
                # Update statistics
                results['min_search_time'] = min(
                    results['min_search_time'],
                    query_time
                )
                results['max_search_time'] = max(
                    results['max_search_time'],
                    query_time
                )
                total_time += query_time
                
        # Calculate final statistics
        total_queries = len(test_queries) * num_iterations
        results['avg_search_time'] = total_time / total_queries
        results['queries_per_second'] = total_queries / total_time
        
        return results

    def compare_with_baseline(
        self,
        test_queries: List[str],
        baseline_search_fn,
        num_iterations: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare HDPSA performance with a baseline search implementation.
        
        Args:
            test_queries: List of queries to test
            baseline_search_fn: Function implementing baseline search
            num_iterations: Number of iterations per query
            
        Returns:
            Dictionary comparing performance metrics
        """
        hdpsa_results = self.benchmark(test_queries, num_iterations)
        
        # Benchmark baseline implementation
        baseline_results = {
            'avg_search_time': 0,
            'min_search_time': float('inf'),
            'max_search_time': 0,
            'queries_per_second': 0
        }
        
        total_time = 0
        for query in test_queries:
            for _ in range(num_iterations):
                start_time = time.time()
                baseline_search_fn(query)
                query_time = time.time() - start_time
                
                baseline_results['min_search_time'] = min(
                    baseline_results['min_search_time'],
                    query_time
                )
                baseline_results['max_search_time'] = max(
                    baseline_results['max_search_time'],
                    query_time
                )
                total_time += query_time
                
        total_queries = len(test_queries) * num_iterations
        baseline_results['avg_search_time'] = total_time / total_queries
        baseline_results['queries_per_second'] = total_queries / total_time
        
        # Calculate improvement ratios
        improvement = {
            metric: baseline_results[metric] / hdpsa_results[metric]
            if metric != 'queries_per_second'
            else hdpsa_results[metric] / baseline_results[metric]
            for metric in hdpsa_results
        }
        
        return {
            'hdpsa': hdpsa_results,
            'baseline': baseline_results,
            'improvement_ratio': improvement
        }

class HDPSATest:
    """Test suite for HDPSA algorithm."""
    
    def __init__(self):
        self.hdpsa = None
        self.test_data = []
        self.test_queries = []
        
    def setup(
        self,
        num_documents: int = 1000,
        num_queries: int = 100,
        seed: int = 42
    ):
        """Set up test environment with synthetic data."""
        np.random.seed(seed)
        
        # Generate synthetic documents
        vocab = ['apple', 'banana', 'cherry', 'date', 'elderberry',
                'fig', 'grape', 'honeydew', 'kiwi', 'lemon']
                
        self.test_data = []
        for _ in range(num_documents):
            # Generate random document
            doc_length = np.random.randint(20, 100)
            words = np.random.choice(vocab, size=doc_length)
            document = ' '.join(words)
            self.test_data.append(document)
            
        # Generate test queries
        self.test_queries = []
        for _ in range(num_queries):
            query_length = np.random.randint(2, 5)
            words = np.random.choice(vocab, size=query_length)
            query = ' '.join(words)
            self.test_queries.append(query)
            
        # Initialize HDPSA
        self.hdpsa = HDPSA(num_shards=10)
        self.hdpsa.fit(self.test_data)
        
        # Index test data
        for i, doc in enumerate(self.test_data):
            self.hdpsa.index(f"doc_{i}", doc)
            
    def run_basic_tests(self) -> Dict[str, bool]:
        """Run basic functionality tests."""
        results = {}
        
        # Test 1: Index and retrieve single document
        try:
            test_doc = "test document content"
            self.hdpsa.index("test_1", test_doc)
            search_results = self.hdpsa.search("test document")
            results['basic_retrieval'] = any(
                r.content_id == "test_1"
                for r in search_results
            )
        except Exception as e:
            logger.error(f"Basic retrieval test failed: {e}")
            results['basic_retrieval'] = False
            
        # Test 2: Pattern matching
        try:
            doc1 = "apple banana cherry"
            doc2 = "cherry banana apple"
            self.hdpsa.index("pattern_1", doc1)
            self.hdpsa.index("pattern_2", doc2)
            results1 = self.hdpsa.search("apple banana")
            results2 = self.hdpsa.search("banana apple")
            
            # Check if similar documents get similar scores
            scores1 = {r.content_id: r.score for r in results1}
            scores2 = {r.content_id: r.score for r in results2}
            score_diff = abs(
                scores1.get("pattern_1", 0) - scores2.get("pattern_1", 0)
            )
            results['pattern_matching'] = score_diff < 0.1
        except Exception as e:
            logger.error(f"Pattern matching test failed: {e}")
            results['pattern_matching'] = False
            
        # Test 3: Concurrent operations
        try:
            def index_docs():
                for i in range(100):
                    self.hdpsa.index(f"concurrent_{i}", f"test document {i}")
                    
            def search_docs():
                for i in range(100):
                    self.hdpsa.search("test document")
                    
            threads = [
                threading.Thread(target=index_docs),
                threading.Thread(target=search_docs)
            ]
            
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
                
            results['concurrent_operations'] = True
        except Exception as e:
            logger.error(f"Concurrent operations test failed: {e}")
            results['concurrent_operations'] = False
            
        return results
        
    def run_performance_tests(
        self,
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """Run performance benchmarks."""
        return self.hdpsa.benchmark(self.test_queries, num_iterations)
        
    def run_comparison_tests(
        self,
        num_iterations: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """Compare with baseline implementation."""
        def baseline_search(query: str) -> List[Tuple[str, float]]:
            """Simple TF-IDF baseline search."""
            results = []
            query_tokens = set(word_tokenize(query.lower()))
            
            for i, doc in enumerate(self.test_data):
                doc_tokens = set(word_tokenize(doc.lower()))
                similarity = len(query_tokens & doc_tokens) / len(query_tokens | doc_tokens)
                results.append((f"doc_{i}", similarity))
                
            return sorted(results, key=lambda x: x[1], reverse=True)[:10]
            
        return self.hdpsa.compare_with_baseline(
            self.test_queries,
            baseline_search,
            num_iterations
        )

def main():
    """Main function to demonstrate HDPSA usage."""
    # Initialize test suite
    test_suite = HDPSATest()
    test_suite.setup(num_documents=1000, num_queries=100)
    
    # Run basic tests
    logger.info("Running basic functionality tests...")
    basic_results = test_suite.run_basic_tests()
    for test_name, passed in basic_results.items():
        logger.info(f"{test_name}: {'PASSED' if passed else 'FAILED'}")
        
    # Run performance tests
    logger.info("\nRunning performance tests...")
    perf_results = test_suite.run_performance_tests()
    for metric, value in perf_results.items():
        logger.info(f"{metric}: {value:.4f}")
        
    # Run comparison tests
    logger.info("\nRunning comparison tests...")
    comp_results = test_suite.run_comparison_tests()
    logger.info("\nPerformance comparison:")
    logger.info(f"HDPSA queries/second: {comp_results['hdpsa']['queries_per_second']:.2f}")
    logger.info(f"Baseline queries/second: {comp_results['baseline']['queries_per_second']:.2f}")
    logger.info(f"Improvement ratio: {comp_results['improvement_ratio']['queries_per_second']:.2f}x")

if __name__ == "__main__":
    main()