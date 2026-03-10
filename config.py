import warnings
import os

os.environ["DISABLE_TQDM"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

warnings.filterwarnings("ignore")

CROSSNER_PATH = "CrossNER/ner_data"
DOMAIN = "ai"

NUNER_MODEL = "numind/NuNER-v2.0"
GLINER_MODEL = "urchade/gliner_small-v2.1"

BATCH_SIZE = 8
LR = 3e-5
EPOCHS = 5

max_span_width=12