import os
import pickle
import time
import matplotlib
import sys
import numpy as np
from torch.hub import download_url_to_file
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 7) # size of window
plt.style.use('dark_background')

LEARNING_RATE = 1e-3
BATCH_SIZE = 1
TRAIN_TEST_SPLIT = 0.7


class Dataset:
    def __init__(self):
        super().__init__()
        path_dataset = '../data/cardekho_india_dataset.pkl'
        if not os.path.exists(path_dataset):
            os.makedirs('../data', exist_ok=True)
            download_url_to_file(
                'http://share.yellowrobot.xyz/1630528570-intro-course-2021-q4/cardekho_india_dataset.pkl',
                path_dataset,
                progress=True
            )
        with open(f'{path_dataset}', 'rb') as fp:
            self.X, self.Y, self.labels = pickle.load(fp)

        self.X = np.array(self.X)
        self.X_scalar = self.X[:, 4:]
        self.X_cat = self.X[:, :4]

        self.X_mean = np.mean(self.X_scalar, axis=0)
        self.X_std = np.std(self.X_scalar, axis=0)
        self.X_scalar = (self.X_scalar - self.X_mean) / self.X_std  # normalizing data

        self.Y = np.array(self.Y)
        self.Y_mean = np.mean(self.Y, axis=0)
        self.Y_std = np.std(self.Y, axis=0)
        self.Y = (self.Y - self.Y_mean) / self.Y_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X_scalar[idx], self.X_cat[idx], self.Y[idx]


class DataLoader:
    def __init__(
            self,
            dataset,
            idx_start, idx_end,
            batch_size
    ):
        super().__init__()
        self.dataset = dataset
        self.idx_start = idx_start
        self.idx_end = idx_end
        self.batch_size = batch_size
        self.idx_batch = 0

    def __len__(self):
        return (self.idx_end - self.idx_start - self.batch_size) // self.batch_size

    def __iter__(self):
        self.idx_batch = 0
        return self

    def __next__(self):
        if self.idx_batch > len(self):
            raise StopIteration()
        idx_start = self.idx_batch * self.batch_size + self.idx_start
        idx_end = idx_start + self.batch_size
        self.idx_batch += 1
        return self.dataset[idx_start:idx_end]

dataset_full = Dataset()
train_test_split = int(len(dataset_full) * TRAIN_TEST_SPLIT)

dataloader_train = DataLoader(
    dataset_full,
    idx_start=0,
    idx_end=train_test_split,
    batch_size=BATCH_SIZE
)
dataloader_test = DataLoader(
    dataset_full,
    idx_start=train_test_split,
    idx_end=len(dataset_full),
    batch_size=BATCH_SIZE
)


class Variable:
    def __init__(self, value, grad=None):
        self.value: np.ndarray = value
        self.grad: np.ndarray = np.zeros_like(value)
        if grad is not None:
            self.grad = grad


class LayerLinear:
    def __init__(self, in_features: int, out_features: int):
        self.W: Variable = Variable(
            value=np.random.uniform(low=-1, high=1, size=(in_features, out_features)),
            grad=np.zeros(shape=(BATCH_SIZE, in_features, out_features))
        )
        self.b: Variable = Variable(
            value=np.zeros(shape=(out_features, )),
            grad=np.zeros(shape=(BATCH_SIZE, out_features))
        )
        self.x: Variable = None
        self.output: Variable = None

    def forward(self, x: Variable):
        self.x = x
        self.output = Variable(
            (self.W.value.T @ self.x.value[:, :, np.newaxis])[:, :, 0] + self.b.value
        )
        return self.output

    def backward(self):
        self.b.grad += self.output.grad
        self.W.grad += self.x.value[:, :, np.newaxis] @ self.output.grad[:, np.newaxis, :]
        self.x.grad += (self.W.value @ self.output.grad[:, :, np.newaxis])[:, :, 0]

class LayerSigmoid():
    def __init__(self):
        self.x = None
        self.output = None

    def forward(self, x: Variable):
        self.x = x
        self.output = Variable(1. / (1. + np.exp(-x.value)))
        return self.output

    def backward(self):
        self.x.grad += self.output.value * (1. - self.output.value) * self.output.grad


class LossMAE():
    def __init__(self):
        self.y = None
        self.y_prim = None

    def forward(self, y: Variable, y_prim: Variable):
        self.y = y
        self.y_prim = y_prim
        loss = np.mean(np.abs(y.value - y_prim.value))
        return loss

    def backward(self):
        self.y_prim.grad += -(self.y.value - self.y_prim.value) / (np.abs(self.y.value - self.y_prim.value) + 1e-8)


class Model:
    def __init__(self):
        self.layers = [
            LayerLinear(in_features=2, out_features=4),
            LayerSigmoid(),
            LayerLinear(in_features=4, out_features=5),
            LayerSigmoid(),
            LayerLinear(in_features=5, out_features=2)
        ]

        self.embs = []

        for x_cat_labels in dataset_full.labels:
            print(x_cat_labels)
            self.embs.append(LayerEmbedding(
                num_embeddings=len(x_cat_labels),
                embedding_dim=2
            ))

    def forward(self, x_scalar, x_cat):
        self.x_cat_embs = []
        for idx, emb_layer in enumerate(self.embs):
            self.x_cat_embs.append(
                emb_layer.forward(Variable(x_cat[:, idx:idx+1]))
            )
        self.x_cat = Variable(np.concatenate([it.value for it in self.x_cat_embs] + [x_scalar.value], axis=-1))
        out = self.x_cat
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self):
        for layer in reversed(self.layers):
            layer.backward()

    def parameters(self):
        variables = []
        for layer in self.layers:
            if isinstance(layer, LayerLinear):
                variables.append(layer.W)
                variables.append(layer.b)
        return variables

class OptimizerSGD:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for param in self.parameters:
            param.value -= np.mean(param.grad, axis=0) * self.learning_rate

    def zero_grad(self):
        for param in self.parameters:
            param.grad = np.zeros_like(param.grad)

# For later
class LayerEmbedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.x_indexes = None
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.emb_m = Variable(np.random.random((num_embeddings, embedding_dim)))
        self.output: Variable = None

    def forward(self, x: Variable):
        self.x_indexes = x.value.squeeze().astype(np.int)
        self.output = Variable(np.array(self.emb_m.value[self.x_indexes, :]))
        return self.output

    def backward(self):
        self.emb_m.grad[self.x_indexes, :] += self.output.grad

model = Model()
optimizer = OptimizerSGD(
    model.parameters(),
    learning_rate=LEARNING_RATE
)
loss_fn = LossMAE()


loss_plot_train = []
loss_plot_test = []
for epoch in range(1, 1000):

    for dataloader in [dataloader_train, dataloader_test]:
        losses = []
        for x_scalar, x_cat, y in dataloader:

            y_prim = model.forward(Variable(value=x_scalar))
            loss = loss_fn.forward(Variable(value=y), y_prim)
            losses.append(loss)

            if dataloader == dataloader_train:
                loss_fn.backward()
                model.backward()

                optimizer.step()
                optimizer.zero_grad()

        if dataloader == dataloader_train:
            loss_plot_train.append(np.mean(losses))
        else:
            loss_plot_test.append(np.mean(losses))

    print(f'epoch: {epoch} loss_train: {loss_plot_train[-1]} loss_test: {loss_plot_test[-1]}')

    if epoch % 10 == 0:
        fig, ax1 = plt.subplots()
        ax1.plot(loss_plot_train, 'r-', label='train')
        ax2 = ax1.twinx()
        ax2.plot(loss_plot_test, 'c-', label='test')
        ax1.legend()
        ax2.legend(loc='upper left')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        plt.show()
