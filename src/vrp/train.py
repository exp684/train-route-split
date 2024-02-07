import torch

from Trainer import Trainer

graph_size = 50
n_epochs = 50
batch_size = 250
nb_train_samples = 10000
nb_val_samples = 1000
n_layers = 3
n_heads = 8
embedding_dim = 128
dim_feedforward = 512
C = 10
dropout = 0.1
learning_rate = 1e-5
seed = 1234

if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(seed)
    trainer = Trainer(graph_size, n_epochs, batch_size, nb_train_samples, nb_val_samples,
                      n_layers, n_heads, embedding_dim, dim_feedforward, C,
                      dropout, learning_rate)
    trainer.train()
