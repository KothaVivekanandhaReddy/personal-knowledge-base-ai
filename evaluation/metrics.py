def recall_at_k(results, relevant_documents, k=5):
    """
    Returns 1.0 if any of the top k retrieved document names 
    are inside the relevant_documents list, else 0.0.
    """
    # Slice the top k results
    top_k_results = results[:k]
    
    # Extract just the clean document names from your vector store dictionaries
    retrieved_names = [doc["document_name"].strip() for doc in top_k_results]
    
    # Clean the expected names list
    target_names = [doc.strip() for doc in relevant_documents]
    
    # Check for intersection
    for name in retrieved_names:
        if name in target_names:
            return 1.0
            
    return 0.0


def reciprocal_rank(results, relevant_documents, k=5):
    """
    Calculates MRR contribution based on the rank of the first 
    matching document found inside the relevant_documents list.
    """
    top_k_results = results[:k]
    target_names = [doc.strip() for doc in relevant_documents]
    
    for rank, doc in enumerate(top_k_results, start=1):
        if doc["document_name"].strip() in target_names:
            return 1.0 / rank
            
    return 0.0


def mean(scores):
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
