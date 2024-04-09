import json
import torch
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
from transformers import BertModel

# Load processed data
with open('processed_data.json', 'r') as f:
    processed_data = json.load(f)

encoded_labels = processed_data['encoded_labels']
label_to_command_map = processed_data['label_to_command_map']
num_classes = len(label_to_command_map)
max_length = 32  # Example maximum sequence length

# Define the BERT-based model architecture for sequence classification
class BertSequenceClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BertSequenceClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.fc(pooled_output)
        return logits

# Load the trained model
model = BertSequenceClassifier(num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('bert_model.pth', map_location=device))
model.eval()

# Load the test data from JSON file
with open('test_data.json', 'r') as f:
    test_data = json.load(f)

# Extract input IDs and labels from the test data
input_ids = torch.tensor(test_data['input_ids'])
labels = torch.tensor(test_data['labels'])

# Tokenize the input sequences
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
test_texts = [tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids)) for ids in test_data['input_ids']]
# Tokenize the input sequences
test_tokenized = tokenizer.batch_encode_plus(test_texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

# Move tensors to the appropriate device
test_input_ids = test_tokenized['input_ids'].to(device)
test_attention_masks = test_tokenized['attention_mask'].to(device)

# Perform inference
with torch.no_grad():
    logits = model(test_input_ids, attention_mask=test_attention_masks)

# Compute predicted labels
predicted_labels = torch.argmax(logits, dim=1)

# Move predicted labels to CPU and convert to numpy array
predicted_labels_np = predicted_labels.cpu().numpy()

# Move labels to CPU and convert to numpy array
labels_np = labels.numpy()

# Calculate accuracy
accuracy = accuracy_score(labels_np, predicted_labels_np)
print(f"Accuracy: {accuracy}")
