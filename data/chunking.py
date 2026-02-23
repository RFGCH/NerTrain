from typing import Dict, List
from transformers import PreTrainedTokenizerBase


from typing import Dict, List

def chunk_gliner_example(
    example: Dict,
    max_length: int = 512
) -> List[Dict]:
    """
    Divide un ejemplo GLiNER en chunks de max_length tokens (palabras).
    Ajusta los spans (en índices de token).
    """
    tokens = example["tokenized_text"]
    spans = example["ner"]

    chunks = []
    token_start = 0
    n_tokens = len(tokens)

    while token_start < n_tokens:
        token_end = min(token_start + max_length, n_tokens)

        chunk_tokens = tokens[token_start:token_end]
        chunk_spans = []

        for start, end, label in spans:
            # span completamente dentro del chunk
            if start >= token_start and end <= token_end:
                chunk_spans.append([
                    start - token_start,
                    end - token_start,
                    label
                ])

        if chunk_spans:
            chunks.append({
                "tokenized_text": chunk_tokens,
                "ner": chunk_spans
            })

        token_start = token_end

    return chunks



def chunk_gliner_dataset(
    dataset: List[Dict],
    max_length: int = 512
) -> List[Dict]:
    """
    Aplica chunking a un dataset GLiNER ya tokenizado.
    """
    chunked_dataset = []

    for example in dataset:
        chunked_dataset.extend(
            chunk_gliner_example(
                example=example,
                max_length=max_length
            )
        )

    return chunked_dataset