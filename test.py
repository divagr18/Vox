import json
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load JSON data
with open('commands.json', 'r') as f:
    data = json.load(f)

# Extract voice commands and AutoHotkey commands
voice_commands = [example['voice_command'] for example in data]
autohotkey_commands = [example['autohotkey_command'] for example in data]

# Tokenize voice commands
tokenized_voice_commands = [tokenizer.tokenize(command) for command in voice_commands]

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode AutoHotkey commands into numerical labels
encoded_labels = label_encoder.fit_transform(autohotkey_commands)

# Convert encoded labels to a Python list
encoded_labels = encoded_labels.tolist()

# Create a dictionary to map numerical labels back to AutoHotkey commands
label_to_command_map = {label: command for label, command in zip(encoded_labels, autohotkey_commands)}

# Optionally, save the processed data
processed_data = {
    'tokenized_voice_commands': tokenized_voice_commands,
    'encoded_labels': encoded_labels,
    'label_to_command_map': label_to_command_map
}

with open('processed_data.json', 'w') as f:
    json.dump(processed_data, f)

print("Processing complete. Saved processed data to 'processed_data.json'.")

import json

# Load processed data from JSON file
with open('processed_data.json', 'r') as f:
    processed_data = json.load(f)

tokenized_voice_commands = processed_data['tokenized_voice_commands']
encoded_labels = processed_data['encoded_labels']
label_to_command_map = processed_data['label_to_command_map']
