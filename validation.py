import json
import torch
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
from transformers import BertModel

with open("processed_data.json", "r") as f:
    processed_data = json.load(f)
encoded_labels = processed_data["encoded_labels"]
label_to_command_map = processed_data["label_to_command_map"]
num_classes = len(label_to_command_map)
max_length = 32


class BertSequenceClassifier(nn.Module):
    def __init__(self, num_classes):
        """Initializes the BertSequenceClassifier model.

        Loads a pretrained BERT base model with frozen parameters and sets up a linear
        layer to classify the BERT output into the specified number of classes.

        Args:
            num_classes (int): The number of target classes for classification."""
        super(BertSequenceClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """Performs a forward pass through the BERT model and a fully connected layer to obtain logits.

        Args:
            input_ids (torch.Tensor): Tensor of input token IDs with shape (batch_size, sequence_length).
            attention_mask (torch.Tensor): Tensor indicating which tokens to attend to (1) or mask (0),
                with shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Logits output from the fully connected layer, typically used for classification."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.fc(pooled_output)
        return logits


model = BertSequenceClassifier(num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("bert_model.pth", map_location=device))
model.eval()
with open("test_data.json", "r") as f:
    test_data = json.load(f)
input_ids = torch.tensor(test_data["input_ids"])
labels = torch.tensor(test_data["labels"])
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
test_texts = [
    tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids))
    for ids in test_data["input_ids"]
]
test_tokenized = tokenizer.batch_encode_plus(
    test_texts,
    padding="max_length",
    truncation=True,
    max_length=max_length,
    return_tensors="pt",
)
test_input_ids = test_tokenized["input_ids"].to(device)
test_attention_masks = test_tokenized["attention_mask"].to(device)
with torch.no_grad():
    logits = model(test_input_ids, attention_mask=test_attention_masks)
predicted_labels = torch.argmax(logits, dim=1)
predicted_labels_np = predicted_labels.cpu().numpy()
labels_np = labels.numpy()
accuracy = accuracy_score(labels_np, predicted_labels_np)
print(f"Accuracy: {accuracy}")
