import matplotlib.pyplot as plt
import numpy as np
import torch

from pandas.io.parsers import read_csv
from torch.utils.data import Dataset

import time
import sys

def plot_image(x, y, y_pred=None):
    '''
    Plota uma imagem com os keypoints

    x: imagem
    y: keypoints
    '''
    plt.imshow(x, cmap='gray')
    plt.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, c='cyan',
            marker='.', s=10)

    if y_pred is not None:
        plt.scatter(y_pred[0::2] * 48 + 48, y_pred[1::2] * 48 + 48,
                c='red', marker='.', s=10)

class ToTensor(object):
    """
    Converte amostra de ndarrays para Tensors.
    """
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']

        # imagem numpy: H x W x C
        # imagem torch: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(keypoints)}

class Normalize(object):
    """
    Normaliza amostra.
    """
    def __call__(self, sample, image_size=96, channels=1):
        """
        sample: amostra para normalização.
        image_size: tamanho da imagem.
        channels: canais para conversão.
        """
        image, keypoints = sample['image'], sample['keypoints']
        image = np.vstack(image) / 255.
        image = image.astype(np.float32)
        image = image.reshape(image_size, image_size, channels)

        # aplica transformação somente no conjunto de treino
        if keypoints.shape[0] > 1:
            keypoints = (keypoints - 48) / 48  # coloca os dados entre [-1, 1]
            keypoints = keypoints.astype(np.float32)

        return {'image': image, 'keypoints': keypoints}

class Channel(object):
    """
    Repete canais.
    """
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']

        # repete canais para simular RGB
        return {'image': image.repeat(3, axis=2),
                'keypoints': keypoints}

class FacialKeypointsDataset(Dataset):
    """Facial Keypoints dataset."""

    def __init__(self, csv_file, train=True, split=0.8, transform=None):
        """
        csv_file (string): caminho para o csv com as anotações e imagens.
        transform (callable, optional): Transformação opcional para aplicar
                                        na amostra.
        """
        # cria dataframe descartando amostras com dados faltando
        self.facial_keypoints = read_csv(csv_file)
        # descarta linhas com dados faltando
        self.facial_keypoints = self.facial_keypoints.dropna()
        # recria os índices
        self.facial_keypoints = self.facial_keypoints.reset_index(drop=True)
        self.transform = transform

        if train:
            self.facial_keypoints = self.facial_keypoints[:int(
                np.round(self.facial_keypoints.shape[0] * split))]
            # recria os índices
            self.facial_keypoints = self.facial_keypoints.reset_index(
                    drop=True)
        else:
            self.facial_keypoints = self.facial_keypoints[int(
                np.round(self.facial_keypoints.shape[0] * split)):]
            # recria os índices
            self.facial_keypoints = self.facial_keypoints.reset_index(
                    drop=True)

    def __len__(self):
        return len(self.facial_keypoints)

    def __getitem__(self, idx):
        # carrega imagem
        image = self.facial_keypoints['Image'][idx]
        # converte string para imagem
        image = np.fromstring(image, sep=' ')
        # carrega keypoints como matriz
        keypoints = self.facial_keypoints.iloc[idx, :-1].as_matrix().astype(
            'float')
        # dicionário com a imagen e os keypoints
        sample = {'image': image, 'keypoints': keypoints}

        # aplica transformação
        if self.transform:
            sample = self.transform(sample)

        return sample

