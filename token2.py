import json
import nltk
from nltk.corpus import wordnet
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from sklearn.metrics import f1_score
import random
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pickle

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
with open("intent.json", "r") as file:
    commands_data = json.load(file)
formatted_data = []


def get_synonyms(word):
    """Retrieve a list of unique synonyms for a given word using WordNet.

    Args:
        word (str): The target word for which to find synonyms.

    Returns:
        List[str]: A list of unique synonym strings derived from WordNet synsets,
        with underscores and hyphens replaced by spaces."""
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").replace("-", " ")
            synonyms.append(synonym)
    return list(set(synonyms))


def word_insertion(tokens, insert_prob=0.1):
    """Randomly inserts duplicate tokens into a list of tokens with a given probability.

    Args:
        tokens (list): A list of tokens (strings) to augment.
        insert_prob (float, optional): Probability of inserting a duplicate token at each token position. Defaults to 0.1.

    Returns:
        list: A new list of tokens with some tokens randomly duplicated and inserted at random positions."""
    augmented_tokens = tokens.copy()
    for i in range(len(tokens)):
        if random.random() < insert_prob:
            augmented_tokens.insert(i, random.choice(augmented_tokens))
    return augmented_tokens


def word_deletion(tokens, delete_prob=0.1):
    """Randomly deletes tokens from the input list based on a specified probability.

    Args:
        tokens (list of str): The list of tokens to be augmented.
        delete_prob (float, optional): Probability of deleting each token. Defaults to 0.1.

    Returns:
        list of str: A new list of tokens with some tokens randomly removed according to delete_prob."""
    augmented_tokens = tokens.copy()
    to_delete = [
        i for i, token in enumerate(augmented_tokens) if random.random() < delete_prob
    ]
    for i in sorted(to_delete, reverse=True):
        del augmented_tokens[i]
    return augmented_tokens


def text_normalization(text):
    """Normalizes the input text by converting it to lowercase and removing all punctuation.

    Args:
        text (str): The input string to be normalized.

    Returns:
        str: The normalized string with all characters in lowercase and punctuation removed."""
    text = text.lower()
    import string

    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


stemmer = PorterStemmer()


def stem_text(text):
    """Stems each word in the input text and returns the stemmed text as a single string.

    Args:
        text (str): A string containing one or more words to be stemmed.

    Returns:
        str: A string with each word stemmed and joined by spaces.

    Note:
        Assumes that a predefined `stemmer` object with a `stem` method is available in the scope."""
    tokens = text.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)


for intent, variations in commands_data.items():
    for variation in variations:
        tokens = variation.split()
        stemmed_tokens = [text_normalization(token) for token in tokens]
        augmented_tokens = []
        for token in stemmed_tokens:
            if random.random() < 0.4:
                synonyms = get_synonyms(token)
                if synonyms:
                    augmented_tokens.append(random.choice(synonyms))
                else:
                    augmented_tokens.append(token)
            else:
                augmented_tokens.append(token)
        augmented_variation = " ".join(augmented_tokens)
        formatted_data.append((augmented_variation.split(), intent))
        augmented_tokens = word_insertion(stemmed_tokens)
        augmented_variation = " ".join(augmented_tokens)
        formatted_data.append((augmented_variation.split(), intent))
intent_labels = list(set(intent for _, intent in formatted_data))
intent_label_map = {intent: idx for idx, intent in enumerate(intent_labels)}
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(intent_labels)
)
max_len = max(len(tokens) for tokens, _ in formatted_data)
input_ids = []
attention_masks = []
labels = []
for tokens, intent in formatted_data:
    encoded_dict = tokenizer.encode_plus(
        tokens,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids.append(encoded_dict["input_ids"])
    attention_masks.append(encoded_dict["attention_mask"])
    labels.append(intent_label_map[intent])
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)
(train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels) = (
    train_test_split(input_ids, attention_masks, labels, test_size=0.2, random_state=42)
)
batch_size = 32
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
optimizer = AdamW(model.parameters(), lr=3e-05, weight_decay=0.01)
epochs = 15
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        inputs, masks, labels = batch
        outputs = model(inputs, attention_mask=masks, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
model.eval()
val_preds = []
val_labels_list = []
with torch.no_grad():
    for batch in val_loader:
        inputs, masks, labels = batch
        outputs = model(inputs, attention_mask=masks)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        val_preds.extend(preds.tolist())
        val_labels_list.extend(labels.tolist())
val_f1 = f1_score(val_labels_list, val_preds, average="weighted")
print(f"Validation F1-score: {val_f1}")
save_directory = "intent_classification_model"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
with open(f"{save_directory}/intent_labels.pkl", "wb") as f:
    pickle.dump(intent_label_map, f)
