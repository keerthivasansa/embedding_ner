from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Input text and the target word
f = open("data/train-medical.txt")
text = f.read()
f.close()

# Tokenize the input text
inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
input_ids = inputs['input_ids']

# Get the tokenized words to match against
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

print("Tokenized Words:", tokens)

def get_embedding(word):
    target_word_tokens = tokenizer.tokenize(word)
    target_word_indices = [i for i, token in enumerate(tokens) if token in target_word_tokens]

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state

    word_embeddings = last_hidden_states[0, target_word_indices, :]

    if word_embeddings.size(0) > 1:
        word_embedding = word_embeddings.mean(dim=0)
    else:
        word_embedding = word_embeddings.squeeze()

    return word_embedding

# Output the embedding
e1 = get_embedding("legs")
e2 = get_embedding("insulin")
e3 = get_embedding("statin")

# c1 = torch.cosine_similarity(e1, e2).item()
# c2 = torch.cosine_similarity(e2, e3).item()

print(e1)
print(e2)
print(e3)