import json
import pickle
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from torch.utils.data import DataLoader, TensorDataset

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Use double backslashes or raw string for file path
file_path = r'D:\Speecher\commands.json'

# Load JSON data
with open(file_path, 'r') as f:
    data = json.load(f)

tokenized_data = []

# Tokenize voice commands and prepare tokenized data
for example in data:
    voice_command = example['voice_command']

    # Tokenize voice command
    tokens = tokenizer.tokenize(voice_command)
    
    # Convert tokens to input IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # Truncate or pad input sequence to fixed length
    max_length = 32  # Example maximum sequence length
    input_ids = input_ids[:max_length] + [tokenizer.pad_token_id] * (max_length - len(input_ids))
    
    tokenized_data.append({
        'input_ids': input_ids,
        'autohotkey_command': example['autohotkey_command']
    })

# Split the tokenized data into training and testing sets
train_data, test_data = train_test_split(tokenized_data, test_size=0.3, random_state=42)


# Extract AutoHotkey commands from the training data
autohotkey_commands = [example['autohotkey_command'] for example in train_data]

# Find the index of the unseen AutoHotkey command in the test data
unseen_command_index = None
for i, example in enumerate(test_data):
    if example['autohotkey_command'] == '^d:: ; Press Ctrl + D to deselect text\nSend, ^d\nreturn':
        unseen_command_index = i
        break

if unseen_command_index is not None:
    print(f"Found unseen command at index {unseen_command_index}")
    unseen_command = test_data[unseen_command_index]['autohotkey_command']
    autohotkey_commands.append(unseen_command)
    print(f"Added unseen command: {unseen_command}")

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit label encoder and transform AutoHotkey commands into numerical labels
encoded_labels = label_encoder.fit_transform(autohotkey_commands)

# Create a dictionary to map numerical labels back to AutoHotkey commands
label_to_command_map = {label: command for label, command in zip(encoded_labels, autohotkey_commands)}

# Print encoded labels and corresponding AutoHotkey commands
for label, command in label_to_command_map.items():
    print(f'Encoded Label: {label}, AutoHotkey Command: {command}')

# Directory to save the processed data
save_dir = 'data'

# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)

# Save the processed data
with open(os.path.join(save_dir, 'train_data.pkl'), 'wb') as f:
    pickle.dump(train_data, f)

with open(os.path.join(save_dir, 'test_data.pkl'), 'wb') as f:
    pickle.dump(test_data, f)

with open(os.path.join(save_dir, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(label_encoder, f)

# Define the default label index for unseen labels
default_label_index = 0

# Encode test labels, handling unseen labels with the default_label_index
test_labels = []
for example in test_data:
    try:
        label_index = label_encoder.transform([example['autohotkey_command']])[0]
    except KeyError:
        label_index = default_label_index
    test_labels.append(label_index)

# Convert the list of label indices to a tensor
test_labels = torch.tensor(test_labels)

# Define the BERT-based model architecture for sequence classification
class BertSequenceClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BertSequenceClassifier, self).__init__()
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
        # Add a classification layer
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        # Get BERT output
        outputs = self.bert(input_ids, attention_mask)
        # Extract pooled output (CLS token)
        pooled_output = outputs.pooler_output
        # Pass pooled output through classification layer
        logits = self.fc(pooled_output)
        return logits

# Prepare data loaders for training and testing
train_inputs = torch.tensor([example['input_ids'] for example in train_data])
train_masks = torch.tensor([[1] * len(example['input_ids']) for example in train_data])  # Assuming all tokens are valid
train_labels = torch.tensor(encoded_labels)

train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_inputs = torch.tensor([example['input_ids'] for example in test_data])
test_masks = torch.tensor([[1] * len(example['input_ids']) for example in test_data])  # Assuming all tokens are valid

test_dataset = TensorDataset(test_inputs, test_masks, test_labels)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define model and move it to GPU if available
model = BertSequenceClassifier(num_classes=len(label_to_command_map))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)  # Example learning rate

# Training loop
num_epochs = 3  # Example number of epochs
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(train_loader)
    print(f'Training - Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}')