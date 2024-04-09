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
# Download the required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load JSON data
with open('intent.json', 'r') as file:
    commands_data = json.load(file)

# Initialize list to store formatted data
formatted_data = []

# Function to get synonyms for a word
def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ').replace('-', ' ')
            synonyms.append(synonym)
    return list(set(synonyms))

# Function to perform word insertion
def word_insertion(tokens, insert_prob=0.1):
    augmented_tokens = tokens.copy()
    for i in range(len(tokens)):
        if random.random() < insert_prob:
            augmented_tokens.insert(i, random.choice(augmented_tokens))
    return augmented_tokens

# Function to perform word deletion
def word_deletion(tokens, delete_prob=0.1):
    augmented_tokens = tokens.copy()
    to_delete = [i for i, token in enumerate(augmented_tokens) if random.random() < delete_prob]
    for i in sorted(to_delete, reverse=True):
        del augmented_tokens[i]
    return augmented_tokens

# Function to perform text normalization
def text_normalization(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    import string
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    return text

# Function to perform stemming
stemmer = PorterStemmer()
def stem_text(text):
    tokens = text.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

# Function to perform lemmatization

# Iterate through JSON data and perform data augmentation
for intent, variations in commands_data.items():
    # Iterate through variations for each intent
    for variation in variations:
        tokens = variation.split()
        
        # Text normalization
        stemmed_tokens = [text_normalization(token) for token in tokens]

        
        # Synonym replacement
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
        augmented_variation = ' '.join(augmented_tokens)
        formatted_data.append((augmented_variation.split(), intent))
        
        # Word insertion
        augmented_tokens = word_insertion(stemmed_tokens)
        augmented_variation = ' '.join(augmented_tokens)
        formatted_data.append((augmented_variation.split(), intent))
        

intent_labels = list(set(intent for _, intent in formatted_data))
intent_label_map = {intent: idx for idx, intent in enumerate(intent_labels)}

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(intent_labels))

# Convert formatted data to input tensors
max_len = max(len(tokens) for tokens, _ in formatted_data)
input_ids = []
attention_masks = []
labels = []

for tokens, intent in formatted_data:
    encoded_dict = tokenizer.encode_plus(tokens, add_special_tokens=True, max_length=max_len, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])
    labels.append(intent_label_map[intent])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Split data into train and validation sets
train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = train_test_split(input_ids, attention_masks, labels, test_size=0.2, random_state=42)

# Create data loaders
batch_size = 32
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Fine-tune the model
optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
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

# Evaluate the model
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

val_f1 = f1_score(val_labels_list, val_preds, average='weighted')
print(f"Validation F1-score: {val_f1}")
save_directory = "intent_classification_model"

# Save the model itself
model.save_pretrained(save_directory)

# Save the tokenizer
tokenizer.save_pretrained(save_directory)

# Save the intent label mapping
with open(f"{save_directory}/intent_labels.pkl", "wb") as f:
    pickle.dump(intent_label_map, f)