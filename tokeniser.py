import json
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Load processed data from JSON file
with open('processed_data.json', 'r') as f:
    processed_data = json.load(f)

encoded_labels = processed_data['encoded_labels']
label_to_command_map = processed_data['label_to_command_map']

# Convert encoded labels to PyTorch tensor
labels_tensor = torch.tensor(encoded_labels)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load JSON data
with open('commands.json', 'r') as f:
    data = json.load(f)

# Extract voice commands
voice_commands = [example['voice_command'] for example in data]

# Tokenize voice commands for training set
train_tokenized_commands = [tokenizer.tokenize(command) for command in voice_commands]

# Encode tokenized commands into input IDs for training set
train_input_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in train_tokenized_commands]

# Pad or truncate input sequences to a fixed length for training set
max_length = 32  # Example maximum sequence length
train_input_ids = [ids[:max_length] + [tokenizer.pad_token_id] * (max_length - len(ids)) for ids in train_input_ids]

# Split the dataset into training and testing sets
train_input_ids, test_input_ids, train_labels, test_labels = train_test_split(train_input_ids, labels_tensor, test_size=0.2, random_state=42)

# Convert input_ids to PyTorch tensors for training and testing sets
train_input_ids = torch.tensor(train_input_ids)
test_input_ids = torch.tensor(test_input_ids)

# Define batch size
batch_size = 16

# Create DataLoader for training set
train_dataset = TensorDataset(train_input_ids, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Save training data as a JSON file
train_data = {'input_ids': train_input_ids.tolist(), 'labels': train_labels.tolist()}
with open('train_data.json', 'w') as f:
    json.dump(train_data, f)

# Create DataLoader for testing set
test_dataset = TensorDataset(test_input_ids, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Save testing data as a JSON file
test_data = {'input_ids': test_input_ids.tolist(), 'labels': test_labels.tolist()}
with open('test_data.json', 'w') as f:
    json.dump(test_data, f)

# Define the number of classes
num_classes = len(label_to_command_map)  # Number of classes obtained from your processed data

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
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Extract pooled output (CLS token)
        pooled_output = outputs.pooler_output
        # Pass pooled output through classification layer
        logits = self.fc(pooled_output)
        return logits

# Initialize the model
model = BertSequenceClassifier(num_classes)

# Set model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 3  # Example number of epochs
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        input_ids, labels = batch
        
        # Create attention mask
        attention_mask = (input_ids != tokenizer.pad_token_id).type(torch.float)
        
        # Move tensors to the GPU if available
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
    
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'bert_model.pth')
