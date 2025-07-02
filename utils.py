import numpy as np
import torch
from typing import Optional
from typing import List, Union
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer
from sentence_transformers import SentenceTransformer
from transformers import AutoModel
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import string
from nltk.corpus import stopwords as stop_words
import warnings
from collections import Counter
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, Dataset
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import os
import string
from sklearn.model_selection import train_test_split


def train_test_split_df(df, text_col='text', label_col='label', test_size=0.2, random_state=42):
    """
    Splits a DataFrame into train and test sets with stratification.

    Args:
        df (pd.DataFrame): The input dataframe.
        text_col (str): Name of the text column.
        label_col (str): Name of the label column.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for reproducibility.

    Returns:
        train_df, test_df (pd.DataFrame): Split dataframes.
    """
    X = df[text_col]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    train_df = pd.DataFrame({text_col: X_train, label_col: y_train})
    test_df = pd.DataFrame({text_col: X_test, label_col: y_test})

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def bert_mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # shape: (batch, seq_len, hidden)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask

def compute_transformer_mean_embeddings(
    texts,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch_size=32,
    max_length=128,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    model = model.to(device)
    model.eval()

    dataset = TextDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []

    with torch.no_grad():
        for batch_texts in dataloader:
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            # Move tensors to device
            encoded = {k: v.to(device) for k, v in encoded.items() if isinstance(v, torch.Tensor)}

            outputs = model(**encoded)
            embeddings = bert_mean_pooling(outputs, encoded["attention_mask"])
            all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)  # shape: (num_docs, hidden_size)



def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output #First element of model_output contains all token embeddings
    #print('token embedding shape', token_embeddings.shape)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    #print('input_mask_expanded shape', input_mask_expanded)
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def compute_bert_label_embedding(keywords_list, model, tokenizer):
  label_vec = []
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  for keywords in keywords_list:
    encoded_input = tokenizer(keywords, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
      model.to(device)
      model_output = model(**encoded_input.to(device))
      hidden = model_output.last_hidden_state #hidden_states[12]+model_output.hidden_states[11]+model_output.hidden_states[10]+model_output.hidden_states[9]
      sbert = mean_pooling(hidden, encoded_input['attention_mask'])
      label_vec.append(sbert.detach().cpu())
  return label_vec


def compute_PLM_label_embeddings(model, keywords_list, tokenizer = None):
  if isinstance(model, SentenceTransformer):
    label_embeddings = [model.encode(keyword) for keyword in keywords_list]
  elif isinstance(model, PreTrainedModel) and tokenizer is not None:
    label_embeddings = compute_bert_label_embedding(keywords_list, model, tokenizer)
  else:
    raise ValueError("Unsupported model type or missing tokenizer.")
  return label_embeddings



def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    """
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

    :param batch: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float
    """
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()

def p_distribution(batch, cluster_centers):
    alpha = 1
    inner = torch.Tensor(batch).unsqueeze(1) - cluster_centers
    #inner = F.normalize(inner, dim=-1)
    norm_squared = torch.sum(inner ** 2, 2)
    numerator = 1.0 / (1.0 + (norm_squared / alpha))
    power = float(alpha + 1) / 2
    numerator = numerator ** power
    return numerator / torch.sum(numerator, dim=1, keepdim=True)

def p_cos_distribution(batch, cluster_centers):
    batch = torch.tensor(batch)
    cluster_centers = torch.tensor(cluster_centers)
    cluster_centers = F.normalize(cluster_centers, dim=-1)
    sim = torch.matmul(batch, cluster_centers.t()) 
    p_cosine = F.softmax(sim, dim=-1)
    return p_cosine

def centroid(vectors: List[np.array]) -> np.array:
    '''
    Returns the centroid vector for a given list of vectors.

    Parameters
    ----------
    vectors : List[`np.array`_]
            The list of vectors to calculate the centroid from

    Returns
    -------
    centroid : `np.array`_
        The centroid vector
    '''
    # stack vectors and calculate mean as document embedding tensor
    centroid = np.stack(vectors).mean(axis=0)
    return centroid



def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a.astype('float32')) #list 

    if not isinstance(b, torch.Tensor): #vector
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    dist = torch.mm(a_norm, b_norm.transpose(0, 1))
    return dist




def top_similar_vectors(key_vector: np.array, candidate_vectors: List[np.array]) -> List[tuple]:
    '''
     Calculates the cosines similarities of a given key vector to a list of candidate vectors.

     Parameters
     ----------
     key_vector : `np.array`_
             The key embedding vector

     candidate_vectors : List[`np.array`_]
             A list of candidate embedding vectors
     Returns
     -------
     top_results : List[tuples]
          A descending sorted of tuples of (cos_similarity, list_idx) by cosine similarities for each candidate vector in the list
     '''
    cos_scores = cos_sim(key_vector, candidate_vectors)[0]
    top_results = torch.topk(cos_scores, k=len(candidate_vectors))
    top_cos_scores = top_results[0].detach().cpu().numpy()
    top_indices = top_results[1].detach().cpu().numpy()
    return list(zip(top_cos_scores, top_indices))

