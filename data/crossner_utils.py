
def read_bio_file(path):

    tokens = []
    labels = []

    cur_tokens = []
    cur_labels = []

    with open(path, encoding="utf8") as f:
        for line in f:

            line = line.strip()

            if not line:
                if cur_tokens:
                    tokens.append(cur_tokens)
                    labels.append(cur_labels)
                    cur_tokens, cur_labels = [], []
                continue

            tok, lab = line.split()
            cur_tokens.append(tok)
            cur_labels.append(lab)

    return tokens, labels