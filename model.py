import torch
import torch.nn as nn
#from ptdec.cluster import ClusterAssignment
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.utils.data import TensorDataset
#activations: ReLU, softplus, Tanh

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(AutoEncoder, self).__init__()
        self.encoder_layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            if i < len(dims) - 2:
                layer = nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.Softplus())
            else:
                layer = nn.Linear(dims[i], dims[i+1])
            self.encoder_layers.append(layer)
        self.encoder = nn.Sequential(*self.encoder_layers)

        self.decoder_layers = []
        hidden_dims.reverse()
        dims = hidden_dims + [input_dim]
        for i in range(len(dims) - 1):
            if i < len(dims) - 2:
                layer = nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.Softplus())
            else:
                layer = nn.Linear(dims[i], dims[i+1])
            self.decoder_layers.append(layer)
        self.decoder = nn.Sequential(*self.decoder_layers)
        self.batch_norm = nn.BatchNorm1d(hidden_dims[-1], affine=False)
        self.dropout_enc = nn.Dropout(p=0.2)

    def forward(self, x):
        z = self.encoder(x)
        z = self.dropout_enc(z)
        # z = F.normalize(z, dim=-1)
        sim = self.sim_mat(z)
        z = self.batch_norm(z)
        x_rec = self.decoder(z)
        return x_rec, z, sim

    def sim_mat(self, z):
        z_norm = F.normalize(z, p=2, dim=1) #L2 norm?
        sim = torch.matmul(z_norm, z_norm.transpose(0, -1))
        return sim

    def decode(self, z):
        z = F.normalize(z, dim=-1)
        return self.decoder(z)



class LSR_clus(nn.Module):
    def __init__(
        self,
        cluster_number: int,
        encoder: torch.nn.Module,
        hidden_dimension: int,
        alpha: float = 1.0,
    ):
        """
        Module which holds all the moving parts of the DEC algorithm, as described in
        Xie/Girshick/Farhadi; this includes the AutoEncoder stage and the ClusterAssignment stage.

        :param cluster_number: number of clusters
        :param hidden_dimension: hidden dimension, output of the encoder
        :param encoder: encoder to use
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        """
        super(LSR_clus, self).__init__()
        self.cluster_number = cluster_number
        self.encoder = encoder
        self.hidden_dimension = hidden_dimension
        self.alpha = alpha
        self.cluster_centers = Parameter(torch.Tensor(cluster_number, hidden_dimension))
        torch.nn.init.xavier_normal_(self.cluster_centers.data)


    def sim_mat(self, z):
        z_norm = F.normalize(z, p=2, dim=1) #L2 norm?
        sim = torch.matmul(z_norm, z_norm.transpose(0, -1))
        return sim

    def cluster_assign(self, z):
        inner = z.unsqueeze(1) - self.cluster_centers
        norm_squared = torch.sum(inner ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        p = numerator / torch.sum(numerator, dim=1, keepdim=True)
        return p

    def cluster_assign_cosine(self, z):
        self.cluster_centers.data = F.normalize(self.cluster_centers.data, dim=-1)
        sim = torch.matmul(z, self.cluster_centers.t()) 
        p_cosine = F.softmax(sim, dim=-1)
        return p_cosine

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the cluster assignment using the ClusterAssignment after running the batch
        through the encoder part of the associated AutoEncoder module.

        :param batch: [batch size, embedding dimension] FloatTensor
        :return: [batch size, number of clusters] FloatTensor
        """
        x_rec, z, _ = self.encoder(batch)
        sim = self.sim_mat(z)
        p = self.cluster_assign(z)
        p_cosine = self.cluster_assign_cosine(z)
        return p, p_cosine, sim


