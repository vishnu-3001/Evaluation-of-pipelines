from bert_score import score

def evaluate_hallucination(pred_answer: str,
                           ground_truth: str,
                           retrieved_docs: list,
                           gold_passage: str,
                           sim_threshold=0.7,
                           grounded_threshold=0.6):
    """
    Evaluate hallucination for a single prediction.

    Args:
        pred_answer (str): Generated answer by the model
        ground_truth (str): Reference correct answer
        retrieved_docs (list[str]): List of retrieved context docs (e.g., top-10 chunks)
        gold_passage (str): The gold paragraph known to contain the true answer
        sim_threshold (float): Threshold for semantic similarity
        grounded_threshold (float): Threshold for groundedness similarity

    Returns:
        dict: {
            "semantic_similarity": float,
            "max_groundedness": float,
            "grounded_doc_idx": int,
            "gold_passage_similarity": float,
            "hallucinated": bool
        }
    """

    # BERTScore between prediction and ground truth
    _, _, f1_semantic = score([pred_answer], [ground_truth], lang="en", model_type="roberta-base", verbose=False)
    semantic_score = f1_semantic.item()

    # BERTScore between prediction and each retrieved doc
    grounded_scores = []
    for doc in retrieved_docs:
        _, _, f1 = score([pred_answer], [doc], lang="en", model_type="roberta-base", verbose=False)
        grounded_scores.append(f1.item())

    max_groundedness = max(grounded_scores)
    grounded_doc_idx = grounded_scores.index(max_groundedness)

    # BERTScore with gold passage
    _, _, f1_gold = score([pred_answer], [gold_passage], lang="en", model_type="roberta-base", verbose=False)
    gold_score = f1_gold.item()

    hallucinated = semantic_score < sim_threshold or max_groundedness < grounded_threshold

    return {
        "semantic_similarity": round(semantic_score, 4),
        "max_groundedness": round(max_groundedness, 4),
        "grounded_doc_idx": grounded_doc_idx,
        "gold_passage_similarity": round(gold_score, 4),
        "hallucinated": hallucinated
    }
