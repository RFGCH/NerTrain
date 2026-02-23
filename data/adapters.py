from typing import List, Dict
from data.uniner_utils import (
    extract_text_from_conversation,
    extract_entities_by_label,
    build_spans_from_entities,
    char_to_token_span,
)

def tokenize_text(text: str, ner: List[str], tokenizer) -> List[str]:
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False
    )

    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    offsets = encoding["offset_mapping"]

    new_ner = []

    for char_start, char_end, label in ner:
        token_start, token_end = char_to_token_span(char_start, char_end, offsets)

        if token_start is not None and token_end is not None:
            new_ner.append([token_start, token_end, label])
    
    gliner_example = {
        "tokenized_text": tokens,
        "ner": new_ner
    }

    return gliner_example

def uniner_to_gliner(dataset, tokenizer) -> List[Dict]:
    """
    Convierte un split de UniNER (HF Dataset) a ejemplos GLiNER.
    """
    gliner_examples = []

    for example in dataset:
        text = extract_text_from_conversation(example["conversations"])
        entities_by_label = extract_entities_by_label(example["conversations"])
        spans = build_spans_from_entities(text, entities_by_label)
        sample = tokenize_text(text, spans, tokenizer)

        if spans:
            gliner_examples.append(sample)

    return gliner_examples