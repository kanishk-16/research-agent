import pytest
from search_agent.semantic_indexer import (
    STOP_WORDS,
    remove_stop_words,
    PorterStemmer,
    stem_word,
    tokenize_and_preprocess,
    InvertedIndex,
    embed_texts,
    cosine_similarity,
    prompt_specific_search,
)


def test_step_1_remove_stop_words():
    tokens = ["the", "transformer", "model", "is", "reasoning", "about", "context"]
    filtered = remove_stop_words(tokens)
    assert "the" not in filtered
    assert "is" not in filtered
    assert "about" not in filtered
    assert "transformer" in filtered
    assert "model" in filtered
    assert "reasoning" in filtered
    assert "context" in filtered


def test_step_2_porter_stemming():
    stemmer = PorterStemmer()
    assert stemmer.stem("connects") == "connect"
    assert stemmer.stem("connecting") == "connect"
    assert stemmer.stem("connections") == "connect"
    assert stemmer.stem("troubles") == "troubl"
    assert stemmer.stem("troubled") == "troubl"
    assert stemmer.stem("operating") == "oper"
    assert stemmer.stem("operation") == "oper"
    
    # Combined tokenize and preprocess
    processed = tokenize_and_preprocess("Language Models are performing multi-hop reasoning")
    assert "model" in processed  # models -> model
    assert "reason" in processed # reasoning -> reason
    assert "are" not in processed # stop word removed


def test_step_3_inverted_indexing_and_bm25():
    idx = InvertedIndex()
    idx.add_document("doc1", "Deep Learning for Vision", "Convolutional neural networks analyze images.")
    idx.add_document("doc2", "Large Language Models", "Transformers and attention mechanisms for reasoning.")
    idx.add_document("doc3", "Reinforcement Learning", "Policy gradients and reward models for agents.")
    
    assert idx.total_docs == 3
    # Search for language models
    results = idx.search_bm25("transformers and language models", top_k=2)
    assert len(results) > 0
    top_doc_id, top_score = results[0]
    assert top_doc_id == "doc2"
    assert top_score > 0.0


def test_step_4_converting_to_vectors():
    # Test fallback deterministic embedding generation without network
    vectors = embed_texts(gemini_client=None, texts=["Attention is all you need", "Medical radiology reports"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 128
    assert len(vectors[1]) == 128
    # Test normalization
    norm = sum(x * x for x in vectors[0])
    assert abs(norm - 1.0) < 1e-4


def test_step_5_cosine_similarity():
    vec1 = [1.0, 0.0, 1.0]
    vec2 = [1.0, 0.0, 1.0]
    vec3 = [-1.0, 0.0, -1.0]
    vec4 = [0.0, 1.0, 0.0]

    # Identical vectors -> 1.0
    assert abs(cosine_similarity(vec1, vec2) - 1.0) < 1e-5
    # Opposite vectors -> -1.0
    assert abs(cosine_similarity(vec1, vec3) - (-1.0)) < 1e-5
    # Orthogonal vectors -> 0.0
    assert abs(cosine_similarity(vec1, vec4) - 0.0) < 1e-5
    # Empty vectors
    assert cosine_similarity([], [1.0]) == 0.0


def test_step_6_prompt_specific_search():
    candidate_papers = [
        {
            "source_id": "P_LOST",
            "title": "Lost in the Middle: How Language Models Use Long Contexts",
            "abstract": "Performance degrades when key information is located in the middle of long contexts."
        },
        {
            "source_id": "P_MEDICAL",
            "title": "Deep Learning for Chest Radiography and X-ray Analysis",
            "abstract": "Convolutional networks detect thoracic abnormalities from MIMIC-CXR images."
        },
        {
            "source_id": "P_SOLAR",
            "title": "Photovoltaic Solar Power Forecasting with Neural Networks",
            "abstract": "Predicting solar energy generation using weather sensor time series."
        }
    ]

    question = "Does increasing LLM context length degrade reasoning performance due to lost in the middle effects?"
    ranked = prompt_specific_search(
        gemini_client=None,  # Exercises offline deterministic vector + BM25 hybrid path
        question_prompt=question,
        candidate_papers=candidate_papers,
        top_k=3
    )

    assert len(ranked) == 3
    # P_LOST should clearly rank #1
    assert ranked[0]["source_id"] == "P_LOST"
    assert ranked[0]["prompt_similarity_score"] > ranked[1]["prompt_similarity_score"]
    assert "semantic_cosine_sim" in ranked[0]
    assert "lexical_bm25_score" in ranked[0]
