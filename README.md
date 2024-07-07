# BERT Pipeline for Classification

This project involves building a BERT pipeline for classification using a pretrained BERT model for feature extraction and comparing the results with classical feature transformation techniques such as TF-IDF.

## Table of Contents

- [Data Preparation](#data-preparation)
- [BERT Tokenization and Encoding](#bert-tokenization-and-encoding)
- [Classical Feature Transformation](#classical-feature-transformation)
- [T-SNE Visualization](#t-sne-visualization)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)

## Data Preparation

Download and prepare the dataset:

```python
import pandas as pd

# Download the dataset
!gdown --id 1NdvIddoyYy2idsAWxJ8lodKfD-PZhmyL

# Read into a pandas dataframe
df = pd.read_csv("in_domain_train.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])[['label', 'sentence']]
sentences = df.sentence.values
sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
labels = df.label.values
```

## BERT Tokenization and Encoding

Apply BERT tokenizer and encode the sentences:

```python
from pytorch_pretrained_bert import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import TensorDataset, DataLoader

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the sentences
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

# Pad the input tokens
MAX_LEN = 128
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# Create attention masks
attention_masks = [[float(i>0) for i in seq] for seq in input_ids]

# Convert to tensors
batch_size = 8
input_tensor = torch.tensor(input_ids)
masks_tensor = torch.tensor(attention_masks)
train_data = TensorDataset(input_tensor, masks_tensor)
dataloader = DataLoader(train_data, batch_size=batch_size)

# Initialize BERT model and encode sentences
model = BertModel.from_pretrained("bert-base-uncased").to('cuda')
model.eval()
outputs = []

for input, masks in dataloader:
    input = input.to('cuda')
    masks = masks.to('cuda')
    output = model(input, output_all_encoded_layers=False, attention_mask=masks)[0]
    outputs.append(output.cpu().detach().numpy())

outputs = [x for y in outputs for x in y]
```

## Classical Feature Transformation

Apply TF-IDF transformation:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenizer.tokenize)
tfidf_features = tfidf_vectorizer.fit_transform(sentences)
```

## T-SNE Visualization

Plot T-SNE for both representations:

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_tsne(data, title):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_data = tsne.fit_transform(data)
    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=labels, cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.colorbar()
    plt.show()

plot_tsne(np.mean(outputs, axis=1), "T-SNE Plot for Aggregated Representation")
plot_tsne(tfidf_features.toarray(), "T-SNE Plot for TF-IDF Features")
```

## Model Training and Evaluation

Train and evaluate an ML model:

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(np.mean(outputs, axis=1), labels, test_size=0.2, random_state=42)
lr_model_agg = LogisticRegression(max_iter=1000)
lr_model_agg.fit(X_train, y_train)
y_pred_agg = lr_model_agg.predict(X_test)
accuracy_agg = accuracy_score(y_test, y_pred_agg)
print("Accuracy using aggregated representation:", accuracy_agg)

X_train_tfidf, X_test_tfidf, _, _ = train_test_split(tfidf_features, labels, test_size=0.2, random_state=42)
lr_model_tfidf = LogisticRegression(max_iter=1000)
lr_model_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = lr_model_tfidf.predict(X_test_tfidf)
accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)
print("Accuracy using TF-IDF features:", accuracy_tfidf)
```

## Results

The results are obtained by comparing the accuracy of a Logistic Regression model on aggregated BERT representations which was 77% and TF-IDF features which was 72%. 
T-SNE plots are also provided for visual comparison.
