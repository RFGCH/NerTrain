from gliner import GLiNER
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_gliner_model(
    model_name: str = "urchade/gliner_small-v2.1",
    max_length: int = 512
) -> GLiNER:
    """
    Carga un modelo GLiNER preentrenado.
    """
    model = GLiNER.from_pretrained(
        model_name,
        max_length=max_length
    ).to(device)
    return model

def load_nuner_model(
    model_name: str = "numind/NuNER-v2.0",
    max_length: int = 512
) -> AutoModelForTokenClassification:
    """
    Carga un modelo NuNER preentrenado.
    """
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        max_length=max_length,
        num_labels=10,
        ignore_mismatched_sizes=True
    ).to(device)
    return model

def load_tokenizer_gliner(
    model_name: GLiNER
):
    return model_name.data_processor.transformer_tokenizer

def load_tokenizer_nuner(
    model_name: str = "numind/NuNER-v2.0"
) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name)


def train_gliner_model(
    model: GLiNER,
    train_data: List[Dict],
    eval_data: List[Dict],
    output_dir: str,
    max_steps: int = 1000,
    learning_rate: float = 3e-5,
    per_device_train_batch_size: int = 16
):
    """
    Entrena un modelo GLiNER.
    """
    trainer = model.train_model(
        train_dataset=train_data,
        eval_dataset=eval_data,
        output_dir=output_dir,
        max_steps=max_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size
    )

    trainer.save_model(trainer.args.output_dir)