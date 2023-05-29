from scipy.stats import multivariate_t

import torch
from torch.utils.data import Dataset
import numpy as np


class MultivariateTDataset(Dataset):
    def __init__(self, mean, cov, num_examples, num_dim):
        #self.data = torch.zeros((num_examples, num_dim))
        rv = multivariate_t(mean, cov, df=1)
        self.gt = rv.rvs(size=num_examples)
        self.data = self.gt.T
        self.data = self.data.astype(np.double)
        self.gt = self.data

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        return self.data[:,idx]


class MultivariateNormalDataset(Dataset):
    def __init__(self, mean, cov, num_examples, num_dim):
        #self.data = torch.zeros((num_examples, num_dim))
        self.gt = np.random.multivariate_normal(mean, cov, (num_examples))
        self.data = self.gt.T
        self.data = self.data.astype(np.double)
        self.gt = self.data

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        return self.data[:,idx]


class RadenmacherDataset(Dataset):
    def __init__(self, num_examples, num_dim):
        self.gt = np.zeros((num_examples, num_dim))
        for i in range(num_examples):
            for j in range(num_dim):
                self.gt[i, j] = np.random.choice([-1, 1])
        self.data = self.gt.T
        self.data = self.data.astype(np.double)
        self.gt = self.data
 
    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        return self.data[:,idx]


class GaussianDataset(Dataset):
    def __init__(self, num_examples, num_dim):
        #self.data = torch.zeros((num_examples, num_dim))
        self.gt = torch.normal(mean=0, std=1, size=(num_examples, num_dim))
        self.gt = self.gt.numpy()
        self.data = self.gt.T
        self.data = self.data.astype(np.double)
        self.gt = self.data

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        return self.data[:,idx]


class LogNormalDataset(Dataset):
    def __init__(self, num_examples, num_dim):
        self.gt = torch.zeros((num_examples, num_dim))
        self.gt.log_normal_(mean=0, std=1)
        self.gt = self.gt.numpy()
        self.data = self.gt.T
        self.data = self.data.astype(np.double)
        self.gt = self.data

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        return self.data[:,idx]


class PointSphereDataset(Dataset):
    def __init__(self, num_examples, num_points, num_dim, num_coeffs=5):
        self.gt = self._sample_sphere(num_dim, num_points).astype(np.float32)
        self.coeffs = self._sample_coefficients(num_points, num_examples, num_coeffs).astype(np.float32)
        self.data = np.einsum('ij,jk->ik', self.gt, self.coeffs).astype(np.float32)
    
    def _sample_sphere(self, num_dim, num_points):
        vec = np.random.randn(num_dim, num_points)
        vec /= np.linalg.norm(vec, axis=0)
        return vec

    def _sample_coefficients(self, num_points, num_examples, num_coeffs):
        # TODO: Add option for Bernoulli data
        prob_active = num_coeffs / num_points
        binary_rv = np.random.choice([0, 1], size=(num_points, num_examples), p=[1-prob_active, prob_active])
        scale = np.random.uniform(0, 1, (num_points, num_examples))
        avg_num_features = np.average(np.sum(binary_rv, axis=0))
        print(f"Avg num features: {avg_num_features}")
        return scale * binary_rv

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        return self.data[:,idx]