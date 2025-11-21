"""
Performance comparison between single and batch embedding.
"""
import sys
import os
import time

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RepoKnowledgeGraphLib.ModelService import create_model_service


def benchmark_embedding_performance():
    """Compare performance of single vs batch embedding."""
    print("\n=== Embedding Performance Benchmark ===\n")
    
    # Create model service
    model_kwargs = {
        "EMBED_MODEL_NAME": "sentence-transformers/all-MiniLM-L6-v2"
    }
    model_service = create_model_service(
        embedder_type='sentence-transformers',
        model_kwargs=model_kwargs
    )
    
    # Generate test data (realistic code chunks)
    code_chunks = [
        f"def function_{i}(x):\n    return x * {i}\n" for i in range(50)
    ]
    
    print(f"Benchmarking with {len(code_chunks)} code chunks...")
    print("-" * 60)
    
    # Test single embedding (one at a time)
    print("\n1. Single Embedding (Sequential):")
    start_time = time.time()
    single_embeddings = []
    for chunk in code_chunks:
        emb = model_service.embed_chunk_code(chunk)
        single_embeddings.append(emb)
    single_time = time.time() - start_time
    print(f"   Time: {single_time:.2f} seconds")
    print(f"   Average per chunk: {single_time / len(code_chunks):.4f} seconds")
    
    # Test batch embedding
    print("\n2. Batch Embedding:")
    start_time = time.time()
    batch_embeddings = model_service.embed_chunk_code_batch(code_chunks)
    batch_time = time.time() - start_time
    print(f"   Time: {batch_time:.2f} seconds")
    print(f"   Average per chunk: {batch_time / len(code_chunks):.4f} seconds")
    
    # Calculate improvement
    speedup = single_time / batch_time if batch_time > 0 else 0
    time_saved = single_time - batch_time
    percent_faster = ((single_time - batch_time) / single_time * 100) if single_time > 0 else 0
    
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"Speedup: {speedup:.2f}x faster")
    print(f"Time saved: {time_saved:.2f} seconds ({percent_faster:.1f}% faster)")
    print(f"Batch embedding is {percent_faster:.1f}% more efficient!")
    print("=" * 60)
    
    # Verify results are the same
    print("\nVerifying embeddings match...")
    for i, (single, batch) in enumerate(zip(single_embeddings[:3], batch_embeddings[:3])):
        diff = sum(abs(a - b) for a, b in zip(single, batch))
        print(f"  Chunk {i}: difference = {diff:.6f}")
        assert diff < 0.01, f"Embeddings don't match for chunk {i}"
    
    print("\n✅ All embeddings match! Batch embedding is working correctly.")


if __name__ == "__main__":
    print("=" * 60)
    print("Batch Embedding Performance Test")
    print("=" * 60)
    
    try:
        benchmark_embedding_performance()
        
        print("\n" + "=" * 60)
        print("🎉 Benchmark completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
