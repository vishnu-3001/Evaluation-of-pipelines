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
    _, _, f1_semantic = score([pred_answer], [ground_truth], lang="en", model_type="roberta-base", verbose=False)
    semantic_score = f1_semantic.item()
    grounded_scores = []
    for doc in retrieved_docs:
        _, _, f1 = score([pred_answer], [doc], lang="en", model_type="roberta-base", verbose=False)
        grounded_scores.append(f1.item())

    max_groundedness = max(grounded_scores)
    grounded_doc_idx = grounded_scores.index(max_groundedness)
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

def evaluate_hallucination_llm(pred_answer: str,
                               ground_truth: str,
                               context: str,
                               sim_threshold=0.7,
                               grounded_threshold=0.6):
    """
    Evaluate hallucination for a direct LLM call where you have the original (true) answer and passage context.

    Args:
        pred_answer (str): Model-generated answer
        ground_truth (str): Reference / true answer
        context (str): Passage or context given to the LLM
        sim_threshold (float): Threshold for semantic similarity to gold
        grounded_threshold (float): Threshold for similarity to context

    Returns:
        dict: {
            "semantic_similarity": float,
            "groundedness": float,
            "hallucinated": bool
        }
    """

    # Semantic similarity (pred vs ground truth)
    _, _, f1_sem = score([pred_answer], [ground_truth], lang="en", model_type="roberta-base", verbose=False)
    semantic_score = f1_sem.item()

    # Groundedness (pred vs context)
    _, _, f1_ground = score([pred_answer], [context], lang="en", model_type="roberta-base", verbose=False)
    grounded_score = f1_ground.item()

    hallucinated = semantic_score < sim_threshold or grounded_score < grounded_threshold

    return {
        "semantic_similarity": round(semantic_score, 4),
        "groundedness": round(grounded_score, 4),
        "hallucinated": hallucinated
    }