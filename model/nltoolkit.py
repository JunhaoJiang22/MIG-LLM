import nltk
from filelock import FileLock
from huggingface_hub import is_offline_mode


def init_nltk():
    r"""Initialize nltk data files"""
    
    try:
        nltk.data.find("tokenizers/punkt")
    except (LookupError, OSError):
        if is_offline_mode():
            raise LookupError("Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files")
        with FileLock(".lock") as lock:
            nltk.download("punkt", quiet=True)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rouge score expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels
