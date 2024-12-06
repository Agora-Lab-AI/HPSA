# Hierarchical Probabilistic Search Architecture: A Novel Framework for Sub-Linear Time Information Retrieval


[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)


## Abstract

This paper introduces a theoretical framework for a revolutionary search algorithm that achieves unprecedented speed through a combination of probabilistic data structures, intelligent data partitioning, and a novel hierarchical index structure. Our proposed architecture, HPSA (Hierarchical Probabilistic Search Architecture), achieves theoretical search times approaching O(log log n) in optimal conditions by leveraging advanced probabilistic filters and a new approach to data organization. We present both the theoretical foundation and a detailed implementation framework for this new approach to information retrieval.

Traditional search engines operate with time complexities ranging from O(log n) to O(n), depending on the implementation and index structure. While considerable improvements have been made through various optimization techniques, fundamental mathematical limits have constrained theoretical advances in search speed. This paper presents a novel approach that challenges these traditional boundaries through a combination of probabilistic data structures and a new hierarchical architecture.

Current search algorithms primarily rely on inverted indices, which maintain a mapping between terms and their locations in the document corpus. While efficient, these structures still require significant time for query processing, especially when handling complex boolean queries or performing relevance ranking. The theoretical lower bound for traditional search approaches has been well-established at Ω(log n) for most practical implementations.

The exponential growth of digital information demands new approaches to information retrieval that can scale beyond traditional limitations. Our work is motivated by the observation that document collections often exhibit natural clustering and hierarchical relationships that can be exploited for faster search operations.


[PAPER LINK](hpsa.pdf)


## Usage

```python


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


```


## Output

```txt
Generating test dataset...
Initializing HPSA...
Indexing documents...
Indexing completed in 183.26 seconds
Generating test queries...
Running benchmark...

Benchmark Results:
Queries per second: 3973.11
Mean response time: 0.25ms
95th percentile response time: 0.47ms
99th percentile response time: 5.94ms

```



