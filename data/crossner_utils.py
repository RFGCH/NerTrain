from datasets import Dataset
from datasets import disable_progress_bar

def read_bio(path):

    sents=[]
    labels=[]
    t=[]
    l=[]

    with open(path,encoding="utf8") as f:
        for line in f:

            line=line.strip()

            if not line:
                if t:
                    sents.append(t)
                    labels.append(l)
                    t,l=[],[]
                continue

            tok,lab=line.split()
            t.append(tok)
            l.append(lab)

    return sents,labels


def load_crossner(base,domain,split):

    return read_bio(f"{base}/{domain}/{split}.txt")

def bio_to_spans(labels):

    spans=[]
    start=None
    ent=None

    for i,l in enumerate(labels):

        if l.startswith("B-"):
            if start is not None:
                spans.append((start,i-1,ent))

            start=i
            ent=l[2:]

        elif l.startswith("I-"):
            continue

        else:
            if start is not None:
                spans.append((start,i-1,ent))
                start=None

    if start is not None:
        spans.append((start,len(labels)-1,ent))

    return spans

def build_gliner_dataset(tokens_list, labels_list):

    data=[]

    for tokens,labels in zip(tokens_list,labels_list):

        spans=bio_to_spans(labels)

        data.append({
            "tokenized_text":tokens,
            "ner":spans
        })

    return data

def label_map(labels):
    
    # label mapping
    unique=set(
        l
        for seq in labels
        for l in seq
    )

    label_list=sorted(unique)

    label2id={l:i for i,l in enumerate(label_list)}
    id2label={i:l for l,i in label2id.items()}

    return label2id, id2label

def prepare_tokenized_dataset(tokens, labels, tokenizer, label2id):
    """
    Refactored function to prepare tokenized dataset.
    
    Args:
        tokens: List of token sequences
        labels: List of label sequences
        tokenizer: HuggingFace tokenizer
        label2id: Mapping from label strings to IDs
    
    Returns:
        Tokenized dataset with aligned labels
    """

    disable_progress_bar()

    # Create dataset from raw tokens and labels
    ds = Dataset.from_dict({
        "tokens": tokens,
        "ner_tags": [[label2id[l] for l in seq] for seq in labels]
    })
    
    # Define tokenization and alignment function
    def tokenize_and_align(ex):
        tok = tokenizer(
            ex["tokens"],
            truncation=True,
            max_length=512,
            is_split_into_words=True
        )
        
        word_ids = tok.word_ids()
        aligned_labels = []
        prev = None
        
        for w in word_ids:
            if w is None:
                aligned_labels.append(-100)
            elif w != prev:
                aligned_labels.append(ex["ner_tags"][w])
            else:
                aligned_labels.append(-100)
            prev = w
        
        tok["labels"] = aligned_labels
        return tok
    
    # Apply tokenization and alignment
    tokenized_ds = ds.map(
        tokenize_and_align,
        remove_columns=ds.column_names
    )
    
    return tokenized_ds
