from datasets import load_dataset
import ast
from typing import Dict, List, Any


def load_uniner_dataset(split: str = "train"):
    return load_dataset("Universal-NER/Pile-NER-type")[split]

def extract_text_from_conversation(
    conversation: List[Dict[str, Any]]
) -> str:
    raw_text = conversation[0]["value"]
    return raw_text.replace("Text: ", "").strip()

def extract_entities_by_label(
    conversation: List[Dict[str, Any]]
) -> Dict[str, List[str]]:
    entities_by_label = {}

    for i in range(len(conversation) - 1):
        turn = conversation[i]

        if (
            turn.get("from") == "human"
            and turn.get("value", "").startswith("What describes")
        ):
            label = (
                turn["value"]
                .split("What describes ")[1]
                .split(" in")[0]
                .upper()
            )

            answer = conversation[i + 1]["value"]

            if answer != "[]":
                entities_by_label[label] = ast.literal_eval(answer)

    return entities_by_label


def build_spans_from_entities(
    text: str,
    entities_by_label: Dict[str, List[str]]
) -> List[List]:
    spans = []
    used_spans = set()

    for label, entities in entities_by_label.items():
        for entity in entities:
            start = 0
            while True:
                idx = text.find(entity, start)
                if idx == -1:
                    break

                span = (idx, idx + len(entity), label)

                if span not in used_spans:
                    spans.append(list(span))
                    used_spans.add(span)

                start = idx + len(entity)

    return spans

def char_to_token_span(char_start, char_end, offsets):
    token_start = None
    token_end = None

    for i, (s, e) in enumerate(offsets):
        if s <= char_start < e:
            token_start = i
        if s < char_end <= e:
            token_end = i

    return token_start, token_end