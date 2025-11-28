import random

def text_typo_noise(text, prob=0.03):
    new = ""
    for ch in text:
        if random.random() < prob:
            new += random.choice("abcdefghijklmnopqrstuvwxyz")
        else:
            new += ch
    return new

def word_dropout(text, drop_prob=0.1):
    words = text.split()
    kept = [w for w in words if random.random() > drop_prob]
    return " ".join(kept) if kept else text

def lowercase(text):
    return text.lower()
