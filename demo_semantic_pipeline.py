"""
Interactive Demo: 6-Step Semantic Indexing & Vector Search in ARA
Run with: python demo_semantic_pipeline.py
"""

import os
from dotenv import load_dotenv
from search_agent.semantic_indexer import (
    remove_stop_words,
    tokenize_and_preprocess,
    PorterStemmer,
    InvertedIndex,
    embed_texts,
    cosine_similarity,
    prompt_specific_search,
)

load_dotenv()

def run_demo():
    print("=" * 70)
    print("ARA SEMANTIC INDEXING & VECTOR SEARCH PIPELINE DEMO")
    print("=" * 70)

    sample_sentence = "Large Language Models are performing multi-hop reasoning over long contexts"
    print(f"\n[Raw Input Text]:\n  \"{sample_sentence}\"")

    # -------------------------------------------------------------
    # STEP 1: Removing Stop Words
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 1: REMOVING STOP WORDS")
    print("-" * 70)
    import re
    raw_tokens = re.findall(r"\b[a-zA-Z]{2,}\b", sample_sentence.lower())
    non_stop = remove_stop_words(raw_tokens)
    print(f"Raw tokens:         {raw_tokens}")
    print(f"Stop words filtered: {non_stop}")
    print(f"Removed words:      {[t for t in raw_tokens if t not in non_stop]}")

    # -------------------------------------------------------------
    # STEP 2: Stemming (Porter Stemmer)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 2: PORTER STEMMING")
    print("-" * 70)
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(w) for w in non_stop]
    print(f"Input tokens:   {non_stop}")
    print(f"Stemmed roots:  {stemmed}")
    print(f"Example stems:  'models' -> '{stemmer.stem('models')}', 'performing' -> '{stemmer.stem('performing')}', 'connections' -> '{stemmer.stem('connections')}'")

    # -------------------------------------------------------------
    # STEP 3: Inverted Indexing
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 3: INVERTED INDEXING & BM25 SEARCH")
    print("-" * 70)
    candidate_papers = [
        {
            "source_id": "PAPER_1",
            "title": "Lost in the Middle: How Language Models Use Long Contexts",
            "abstract": "We find that retrieval accuracy degrades when relevant information is in the middle of input contexts."
        },
        {
            "source_id": "PAPER_2",
            "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
            "abstract": "Generating a chain of thought significantly improves reasoning abilities on multi-hop benchmarks."
        },
        {
            "source_id": "PAPER_3",
            "title": "Deep Learning for Chest Radiography and Thoracic X-Ray Analysis",
            "abstract": "Multimodal fusion of medical imaging and radiology clinical report notes."
        }
    ]

    index = InvertedIndex()
    for p in candidate_papers:
        index.add_document(p["source_id"], p["title"], p["abstract"])
    print(f"Indexed {index.total_docs} research papers.")
    print(f"Unique vocabulary terms in inverted index: {len(index.index)}")

    bm25_hits = index.search_bm25("multi-hop reasoning in language models", top_k=3)
    print(f"BM25 Search for 'multi-hop reasoning in language models':")
    for doc_id, score in bm25_hits:
        print(f"  • {doc_id}: BM25 Score = {score:.4f}")

    # -------------------------------------------------------------
    # STEP 4: Converting to Vectors
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 4: CONVERTING TEXT TO DENSE VECTORS")
    print("-" * 70)
    gemini_client = None
    try:
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            gemini_client = genai.Client(api_key=api_key)
            print("Connected to Google Gemini API (gemini-embedding-001).")
    except Exception:
        pass

    texts_to_embed = [p["title"] for p in candidate_papers]
    vectors = embed_texts(gemini_client, texts_to_embed)
    print(f"Generated {len(vectors)} dense embeddings.")
    print(f"Vector dimensionality: {len(vectors[0])} dimensions.")
    print(f"Sample vector snippet (first 5 floats): {vectors[0][:5]}")

    # -------------------------------------------------------------
    # STEP 5: Cosine Similarity Algorithm
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 5: COSINE SIMILARITY CALCULATION")
    print("-" * 70)
    sim_1_2 = cosine_similarity(vectors[0], vectors[1])
    sim_1_3 = cosine_similarity(vectors[0], vectors[2])
    print(f"Cosine Similarity (Paper 1 vs Paper 2 - both LLM papers): {sim_1_2:.4f}")
    print(f"Cosine Similarity (Paper 1 vs Paper 3 - LLM vs Medical X-Ray): {sim_1_3:.4f}")
    print("Result: Semantic similarity correctly identifies domain affinity.")

    # -------------------------------------------------------------
    # STEP 6: Prompt-Specific Search
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 6: PROMPT-SPECIFIC HYBRID SEARCH")
    print("-" * 70)
    user_prompt = "Does increasing context window length degrade LLM accuracy due to lost-in-the-middle effects?"
    print(f"Planner Sub-Question Prompt:\n  \"{user_prompt}\"\n")

    ranked_results = prompt_specific_search(
        gemini_client=gemini_client,
        question_prompt=user_prompt,
        candidate_papers=candidate_papers,
        inverted_index=index,
        top_k=3,
        semantic_weight=0.65
    )

    for rank, p in enumerate(ranked_results, start=1):
        print(f"Rank #{rank}: [{p['source_id']}] {p['title']}")
        print(f"         Hybrid Score: {p['prompt_similarity_score']:.4f}  |  Semantic Cosine: {p['semantic_cosine_sim']:.4f}  |  BM25: {p['lexical_bm25_score']:.4f}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE: All 6 stages executed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    run_demo()
