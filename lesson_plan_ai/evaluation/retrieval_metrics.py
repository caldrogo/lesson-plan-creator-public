import math


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return sum(item in relevant for item in retrieved[:k]) / k if k else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return sum(item in relevant for item in retrieved[:k]) / len(relevant) if relevant else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    return next((1 / index for index, item in enumerate(retrieved, 1) if item in relevant), 0.0)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum((1 / math.log2(index + 2)) for index, item in enumerate(retrieved[:k]) if item in relevant)
    ideal = sum(1 / math.log2(index + 2) for index in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0
