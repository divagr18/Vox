# Use a pipeline as a high-level helper
from transformers import pipeline
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
pipe = pipeline("zero-shot-classification", model="MoritzLaurer/bge-m3-zeroshot-v2.0")
input_text = "is this fact true"
labels = ["check mail", "show time", "check weather", "enable bluetooth",
          "disable bluetooth", "quit", "refresh", "save", "copy", "paste", "delete", "search", "reload", "shut down", "verify this fact"]

results = []
for label in labels:
    result = pipe(input_text, label)  # Capture the full result
    results.append({
        'label': label,  
        'score': result['scores'][0]  # Access the first (and often only) score 
    })


# Sort results by descending score
results.sort(key=lambda x: x['score'], reverse=True)

# Top Label is now the most likely one, order preserved
top_label = results[0]['label']  
print(top_label) 
