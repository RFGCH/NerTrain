from typing_extensions import runtime
from contextlib import redirect_stdout
import torch, time

from config import *
from gliner import GLiNER
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer
from seqeval.metrics import precision_score, recall_score, f1_score

import os
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_gliner_model(
    model_name: str = GLINER_MODEL,
    max_length: int = 512
) -> GLiNER:
    """
    Carga un modelo GLiNER preentrenado.
    """
    model = GLiNER.from_pretrained(
        model_name,
        max_length=max_length,
    ).to(device)

    return model

def load_nuner_model(
    model_name: str = NUNER_MODEL,
    id2label = None,
    label2id = None
) -> AutoModelForTokenClassification:
    """
    Carga un modelo NuNER preentrenado.
    """
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    ).to(device)
    
    return model

def load_tokenizer_gliner(
    model_name: GLiNER
):
    return model_name.data_processor.transformer_tokenizer

def load_tokenizer_nuner(
    model_name: str = NUNER_MODEL
) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        add_prefix_space=True
    )
    return tokenizer


def train_gliner_model(
    model: GLiNER,
    train_data: List[Dict],
    eval_data: List[Dict],
    output_dir: str,
    max_steps: int = 10000
):

    with open(os.devnull, "w") as f, redirect_stdout(f):

        # Tomamos tiempo
        start = time.time()

        trainer = model.train_model( 
            train_dataset=train_data, 
            eval_dataset=eval_data, 
            output_dir=output_dir, 
            max_steps=max_steps,
            learning_rate=LR, 
            per_device_train_batch_size=BATCH_SIZE,
            
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
    # Si hay GPU involucrada, sincronizamos
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.time()
    elapsed = end - start
    print(f"Train Time: {elapsed:.2f}s")
    trainer.save_model(trainer.args.output_dir)

def evaluate_finetune_gliner_model(
    model: GLiNER,
    domain: str,
    test_data: List[Dict],
    train_data: List[Dict],
    eval_data: List[Dict],
):
    max_steps = len(train_data) * EPOCHS // BATCH_SIZE

    # 1 Eval base model

    model.eval()
    with torch.no_grad():
        metrics_base = model.evaluate(
            test_data,
            batch_size=BATCH_SIZE
        )

    # 2 Fine tune model

    train_gliner_model(
        model=model,
        train_data=train_data,
        eval_data=eval_data,
        output_dir=f"./models/gliner_crossner_{domain}",
        max_steps=max_steps
    )

    # 3 Eval fine tuned model

    model = load_gliner_model(f"./models/gliner_crossner_{domain}")
    model.eval()

    with torch.no_grad():
        metrics_tunned = model.evaluate(
            test_data,
            batch_size=BATCH_SIZE
        )

    print(metrics_base[0].rstrip(), "Base Model")
    print(metrics_tunned[0].rstrip(), "Fine Tuned Model")

def eval_nuner_model(
        model = None,
        test_tok = None,
        tokenizer = None,
        compute_metrics = None
):

    # Args
    collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True
    )
    args = TrainingArguments(
        output_dir="tmp_eval",         
        per_device_eval_batch_size=BATCH_SIZE,
        report_to="none",
        do_train=False,
        do_eval=True,
        save_strategy="no",       
        logging_strategy="no"
    )
    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=test_tok,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # 4️⃣ evaluar
    with open(os.devnull, "w") as f, redirect_stdout(f):
        metrics = trainer.evaluate()

    return metrics

def train_nuner_model(
        model = None,
        train_tok = None,
        dev_tok = None,
        tokenizer = None,
        compute_metrics = None,
        output_dir = None
):
    
    collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True
    )
    args = TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,

        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    with open(os.devnull, "w") as f, redirect_stdout(f):
        trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return trainer

def evaluate_finetune_nuner_model(
        model = None,
        domain = None,
        id2label = None,
        label2id = None,
        test_tok = None,
        train_tok = None,
        dev_tok = None,
        tokenizer = None
):

    def compute_metrics(p):

        preds = np.argmax(p.predictions, axis=-1)
        labels = p.label_ids

        true_preds=[]
        true_labels=[]

        for pred,lab in zip(preds,labels):

            cur_preds=[]
            cur_labels=[]

            for p_,l_ in zip(pred,lab):

                if l_==-100:
                    continue

                cur_preds.append(id2label[p_])
                cur_labels.append(id2label[l_])

            true_preds.append(cur_preds)
            true_labels.append(cur_labels)

        return {
            "precision": precision_score(true_labels,true_preds),
            "recall": recall_score(true_labels,true_preds),
            "f1": f1_score(true_labels,true_preds),
        }

    # 1 Eval base model

    metrics_base = eval_nuner_model(
        model=model,
        test_tok=test_tok,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # 2 Fine tune model
    train_nuner_model(
        model = model,
        train_tok = train_tok,
        dev_tok = dev_tok,
        tokenizer = tokenizer,
        compute_metrics = compute_metrics,
        output_dir = f"models/nuner_crossner_{domain}"
    )

    # 3 Eval fine tuned model

    model = load_nuner_model(
        f"models/nuner_crossner_{domain}", 
        id2label=id2label, 
        label2id=label2id
    )
    tokenizer = load_tokenizer_nuner(
        f"models/nuner_crossner_{domain}"
    )

    metrics_tunned = eval_nuner_model(
        model=model,
        test_tok=test_tok,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("P:", str(metrics_base["eval_precision"]*100)[:5], "%\t", "R:", str(metrics_base["eval_recall"]*100)[:5], "%\t", "F1:", str(metrics_base["eval_f1"]*100)[:5], "Base Model"  )
    print("P:", str(metrics_tunned["eval_precision"]*100)[:5], "%\t", "R:", str(metrics_tunned["eval_recall"]*100)[:5], "%\t", "F1:", str(metrics_tunned["eval_f1"]*100)[:5], "Tunned Model"  )