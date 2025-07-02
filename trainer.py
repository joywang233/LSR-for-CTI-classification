import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader, default_collate
from typing import Tuple, Callable, Optional, Union
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from typing import List, Union
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from utils import target_distribution, p_distribution, p_cos_distribution, centroid, top_similar_vectors, cos_sim
from model import LSR_clus, AutoEncoder

class LSRTrainer:
  def __init__(self,
      cluster_number:int,
      input_dim:int, #768
      hidden_dims: list, #[100,100]
      hidden_dimension: int,
      epochs: int,
      batch_size: int,
      lr:int,
      collate_fn=default_collate,
      sampler: Optional[torch.utils.data.sampler.Sampler] = None,
      clean_outliers: bool = False,
      #silent: bool = False,
      min_num_docs: int = 1,
      update_freq: int = 10
      ):
      self.cluster_number = cluster_number
      self.model = LSR_clus(cluster_number = cluster_number,
        encoder=AutoEncoder(input_dim = input_dim, hidden_dims = hidden_dims),
        hidden_dimension = hidden_dimension)
      self.epochs = epochs
      self.collate_fn = collate_fn
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.batch_size = batch_size
      self.lr = lr
      self.update_freq = update_freq
      self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
      self.sampler = sampler
      self.min_num_docs = min_num_docs
      self.clean_outliers = clean_outliers
      self.silent = False
      self.labels = None
      self.documents = None
      self.similarity_threshold = None
      self.similarity_threshold_offset = 0.05 #dynamic threshold, using the max_similarity -0.05 to refined the keyword embeddings
      # self.cluster_weight = cluster_weight #temperature number 

  def pretrain(self,train_dataset, pretrain_epoch = 20, pretrain_lr=0.001):
      model = self.model.encoder
      eps = 1e-10
      dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
      criterion = nn.MSELoss(reduction="mean")
      optim = torch.optim.Adam(model.parameters(), lr=pretrain_lr)
      model.train()
      model = model.to(self.device)
      loss_lst = []
      for epoch in range(pretrain_epoch):
        per_epoch_loss = 0
        for batch_ix, batch in enumerate(dataloader):
            x = batch[0].to(self.device)
            labels = batch[1].to(self.device)
            optim.zero_grad()
            x_rec, _, sim_mat = model.forward(x)
            rec_loss = criterion(x, x_rec)
            loss = rec_loss
            per_epoch_loss += loss
            loss.backward()
            optim.step()
        print('per epoch loss is', per_epoch_loss)
        loss_lst.append(per_epoch_loss.cpu().detach().numpy())
      
        if loss < 0.01:
          break
          #torch.save(model.ae.state_dict(), pretrained_path)
      plt.figure(figsize=(10, 10))
      plt.plot(loss_lst) 
      plt.show()
      plt.close()

      
  def train(self, train_dataset: torch.utils.data.Dataset, centroids_lsa, evaluate_every:int = 30):
      """
      Train the DEC model given a dataset, a model instance and various configuration parameters.

      :param dataset: instance of Dataset to use for training
      :param model: instance of DEC model to train
      :param epochs: number of training epochs
      :param batch_size: size of the batch to train with
      :param optimizer: instance of optimizer to use
      :param stopping_delta: label delta as a proportion to use for stopping, None to disable, default None
      :param collate_fn: function to merge a list of samples into mini-batch
      :param cuda: whether to use CUDA, defaults to True
      :param sampler: optional sampler to use in the DataLoader, defaults to None
      :param silent: set to True to prevent printing out summary statistics, defaults to False
      :param update_freq: frequency of batches with which to update counter, None disables, default 10
      :param evaluate_batch_size: batch size for evaluation stage, default 1024
      :param update_callback: optional function of accuracy and loss to update, default None
      :param epoch_callback: optional function of epoch and model, default None
      :return: None
      """
      static_dataloader = DataLoader(
          train_dataset,
          batch_size=self.batch_size,
          collate_fn=self.collate_fn,
          pin_memory=False,
          sampler=self.sampler,
          shuffle=False,
      )
      train_dataloader = DataLoader(
          train_dataset,
          batch_size=self.batch_size,
          collate_fn=self.collate_fn,
          sampler=self.sampler,
          shuffle=True,
      )
      data_iterator = tqdm(
          static_dataloader,
          leave=True,
          unit="batch",
          disable=False,
      )
      #clustering init 
      kmeans = KMeans(n_clusters=self.cluster_number, n_init=100)
      model = self.model
      model = model.to(self.device)
      model.eval()
      centroids_lsa = torch.tensor(centroids_lsa).to(self.device) #fixed
      features = []
      for index, batch in enumerate(data_iterator):
        x = batch[0].to(self.device)
        x_rec, z, _ = model.encoder(x)
        features.append(z.detach().cpu())
      y_pred_last = kmeans.fit_predict(torch.cat(features).detach().cpu().numpy())
      centroids = kmeans.cluster_centers_
      cluster_centers = torch.tensor(
          centroids, dtype=torch.float, requires_grad=True
      ).to(self.device)
      model.cluster_centers.data = cluster_centers
      #model.cluster_centers = nn.Parameter(cluster_centers)

      loss_function = nn.KLDivLoss(size_average=False)
      clus_loss1_lst = []
      clus_loss2_lst = []
      all_loss_lst = []
      collect_nmi = []
      collect_ari = []
      eps = 1e-10

      for epoch in range(self.epochs):
          per_epoch_cluser_loss1 = 0
          per_epoch_cluser_loss2 = 0

          per_epoch_entropy_loss = 0
          per_epoch_loss = 0

          data_iterator = tqdm(
              train_dataloader,
              leave=True,
              unit="batch",
              disable=self.silent,
          )
          model.train()
          for index, batch in enumerate(data_iterator):
              x = batch[0].to(self.device) #x_contextualized 
              labels = batch[1].to(self.device) 
              x_1 = batch[2].to(self.device) #x_lsa
              p, p_cosine, sim = model(x) #here is p
              p_lsa = p_distribution(x_1, centroids_lsa)
              p_lsa_cos = p_cos_distribution(x_1, centroids_lsa)
              target_lsa = target_distribution(p_lsa_cos).detach()
              cluster_loss1 = loss_function(p_cosine.log(), target_lsa) / p.shape[0] #orangee
              loss = cluster_loss1
              self.optimizer.zero_grad()
              loss.backward()
              self.optimizer.step(closure=None)
              per_epoch_loss += loss.item()
              per_epoch_cluser_loss1 += cluster_loss1.item()

          all_loss_lst.append(per_epoch_loss)
          clus_loss1_lst.append(per_epoch_cluser_loss1)
          clus_loss2_lst.append(per_epoch_cluser_loss2)


          # if (epoch+1) % evaluate_every == 0:
          #     latent_embedding = self.inference(train_dataset)
          #     y_pred = latent_embedding.max(1)[1]
          #     nmi = np.round(normalized_mutual_info_score(y_pred_last, y_pred.clone().detach().cpu().numpy()), 5)
          #     ari = np.round(adjusted_rand_score(y_pred_last, y_pred.clone().detach().cpu().numpy()), 5)
          #     collect_nmi.append(nmi)
          #     collect_ari.append(ari)
          #     print(f'intrinsic evaluations: nmi:{nmi}, ari:{ari}')
          #     y_pred_last = y_pred.detach().clone().cpu().numpy()

              # tsne = TSNE(n_components=2, random_state=0)
              # word_vectors_tsne = tsne.fit_transform(np.array(latent_embedding))
              # plt.figure(figsize=(10, 10))
              # plt.plot(word_vectors_tsne[:, 0], word_vectors_tsne[:, 1], '.', alpha=0.5) #movie 
              # plt.show()
              # plt.close()
              
      plt.figure(figsize=(10, 10))
      plt.plot(clus_loss1_lst, '--', marker ="o", label = 'Clustering loss', color='red') #lsa based 
      plt.xlabel('Number of epoch')
      plt.ylabel('Loss')
      plt.legend(loc="upper right")
      #plt.savefig("training.png", dpi=100)
      plt.show()
      plt.close()
      # latent_embedding = self.inference(train_dataset)
      # return latent_embedding

  def inference(self, train_dataset, return_actual = False):
      """
      Predict clusters for a dataset given a DEC model instance and various configuration parameters.

      :param dataset: instance of Dataset to use for training
      :param model: instance of DEC model to predict
      :param batch_size: size of the batch to predict with, default 1024
      :param collate_fn: function to merge a list of samples into mini-batch
      :param cuda: whether CUDA is used, defaults to True
      :param silent: set to True to prevent printing out summary statistics, defaults to False
      :param return_actual: return actual values, if present in the Dataset
      :return: tuple of prediction and actual if return_actual is True otherwise prediction
      """
      dataloader = DataLoader(
          train_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=False
      )
      data_iterator = tqdm(dataloader, leave=True, unit="batch", disable=self.silent,)
      model = self.model
      features = []
      actual = []
      model = model.to(self.device)
      model.eval()
      for batch in data_iterator:
          x = batch[0].to(self.device)
          x_rec, z, _ = model.encoder(x)
          features.append(z.detach().cpu())
      if return_actual:
          return torch.cat(features).max(1)[1], torch.cat(actual).long()
      else:
          return torch.cat(features)

  def classification(self, train_dataset, keywords_list, label_embeddings, refine_label=True):
    doc_embedding = self.inference(train_dataset).clone().detach().cpu().numpy()
    label_names = list(range(len(keywords_list)))
    self.labels = pd.DataFrame(zip(label_names, keywords_list), columns=['label_name', 'description_keywords'])
    label_vec = [self.inference(torch.Tensor(vec[None, ...])) for vec in label_embeddings]
    self.labels['keyword_vectors'] = label_vec
    self.labels['mean_keyword_vector'] = self.labels['keyword_vectors'].apply(centroid)
    self.documents = pd.DataFrame()
    self.documents['doc_vec'] = list(doc_embedding)
    self.labels['doc_vectors'] = self.labels['mean_keyword_vector'].apply(
        lambda vec: self._get_similar_documents(vec, list(self.documents['doc_vec'])))

    # Label refinement
    label_vec_col = 'label_vector_from_docs' if refine_label else 'mean_keyword_vector'
    if refine_label:
        print('how many docs to refine labels?', len(self.labels['doc_vectors']))
        self.labels[label_vec_col] = self.labels['doc_vectors'].apply(lambda docs: centroid(docs))
    print('what is label vec col?', label_vec_col)
    # Similarity and classification
    label_vecs = list(self.labels[label_vec_col])
    doc_vecs = list(self.documents['doc_vec'])

    sim_matrix = cosine_similarity(doc_vecs, label_vecs)
    print('sim_matrix_dimension:', sim_matrix.shape)
    df = pd.DataFrame(sim_matrix)
    df['doc_key'] = self.documents.index
    df['pred_label'] = df[self.labels['label_name']].idxmax(axis=1)
    df['highest_similarity_score'] = df[self.labels['label_name']].max(axis=1)
    return df[['doc_key', 'pred_label', 'highest_similarity_score'] + list(self.labels['label_name'])]
    

  def _get_similar_documents(self, label_vec, doc_vecs):
      max_docs = len(doc_vecs)
      results = top_similar_vectors(label_vec, doc_vecs[:max_docs])
      if not results:
        return []
      scores, indices = zip(*results)

      df = pd.DataFrame({'score': scores, 'idx': indices}).head(max_docs)
      threshold = df['score'].iloc[0] - self.similarity_threshold_offset if self.similarity_threshold is None else self.similarity_threshold

      if self.min_num_docs and (df[df['score'] > threshold].shape[0] < self.min_num_docs):
          print(f"[WARN] Using fallback: only {df[df['score'] > threshold].shape[0]} docs passed threshold, forcing top-{self.min_num_docs}")
          df = df.head(self.min_num_docs)
      else:
          df = df[df['score'] > threshold]
      return [doc_vecs[i] for i in df['idx']]
  