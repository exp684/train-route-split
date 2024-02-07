from math import sqrt

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.distributions import Categorical

from Encoder import MultiHeadAttention


class Decoder(nn.Module):
    """
        This class contains the decoder that will be used to compute the probability distribution from which we will sample
        which city to visit next.
    """

    def __init__(self, n_head, embedding_dim, decode_mode="sample", C=10):
        super(Decoder, self).__init__()
        self.scale = sqrt(embedding_dim)
        self.decode_mode = decode_mode
        self.C = C

        self.vl = nn.Parameter(
            torch.FloatTensor(size=[1, 1, embedding_dim]).uniform_(-1. / embedding_dim, 1. / embedding_dim),
            requires_grad=True)
        self.vf = nn.Parameter(
            torch.FloatTensor(size=[1, 1, embedding_dim]).uniform_(-1. / embedding_dim, 1. / embedding_dim),
            requires_grad=True)

        self.glimpse = MultiHeadAttention(n_head, embedding_dim, 3 * embedding_dim, embedding_dim, embedding_dim)
        self.project_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

        self.accelerate = Accelerator()
        self.glimpse, self.project_k, self.cross_entropy = self.accelerate.prepare(self.glimpse, self.project_k, self.cross_entropy)

    def forward(self, inputs):
        """

        :param inputs: (encoded_inputs, demands, capacities) ([batch_size, seq_len, embedding_dim],[batch_size, seq_len],[batch_size])
        :return: log_prob, solutions
        """
        encoded_inputs, demands, capacities = inputs

        batch_size, seq_len, embedding_dim = encoded_inputs.size()  # sel_len = nb_clients + nb_depot (1)

        h_hat = encoded_inputs.mean(-2, keepdim=True)

        city_index = None

        # case of vrp : mask[:, 0] is the depot
        mask = torch.zeros([batch_size, seq_len], device=self.accelerate.device).bool()

        solution = torch.zeros([batch_size, 1], dtype=torch.long, device=self.accelerate.device)  # first node is depot
        mask[:, 0] = True  # vehicle is in the depot location

        log_probabilities = torch.zeros(batch_size, dtype=torch.float32, device=self.accelerate.device)

        last = self.vl.repeat(batch_size, 1, 1)  # batch_size, 1, embedding_dim

        first = self.vf.repeat(batch_size, 1, 1)  # batch_size, 1, embedding_dim

        raw_logits = torch.tensor([], device=self.accelerate.device)
        t = 0  # time steps

        # for t in range(seq_len):
        while torch.sum(mask) < batch_size * seq_len:
            t += 1
            h_c = torch.cat((h_hat, last, first), dim=-1)  # [batch_size, 1, 3 * embedding_size]

            context = self.glimpse(h_c, encoded_inputs, encoded_inputs,
                                   mask.unsqueeze(1).unsqueeze(1))

            k = self.project_k(encoded_inputs)

            u = torch.tanh(torch.matmul(context, k.clone().transpose(-2, -1)) / self.scale) * self.C

            raw_logits = torch.cat((raw_logits, u), dim=1)

            u = u.masked_fill(mask.unsqueeze(1), float('-inf'))

            probas = nn.functional.softmax(u.squeeze(1), dim=-1)

            one_hot = torch.zeros([seq_len], device=self.accelerate.device)
            one_hot[0] = 1

            if self.decode_mode == "greedy":
                proba, city_index = self.greedy_decoding(probas)
            elif self.decode_mode == "sample":
                proba, city_index = self.sample_decoding(probas)

            log_probabilities += self.cross_entropy(u.squeeze(1), city_index.view(-1))

            solution = torch.cat((solution, city_index), dim=1)

            # next node for the query
            last = encoded_inputs[[i for i in range(batch_size)], city_index.view(-1), :].unsqueeze(1)

            # update mask
            mask = mask.scatter(1, city_index, True)

            if t == 1:
                first = last
        return raw_logits, log_probabilities, solution

    @staticmethod
    def greedy_decoding(probas):
        """
        :param probas: [batch_size, seq_len]
        :return: probas : [batch_size],  city_index: [batch_size,1]
        """
        probas, city_index = torch.max(probas, dim=1)

        return probas, city_index.view(-1, 1)

    @staticmethod
    def sample_decoding(probas):
        """

        :param probas: [ batch_size, seq_len]
        :return: probas : [batch_size],  city_index: [batch_size,1]
        """
        batch_size = probas.size(0)
        m = Categorical(probas)
        city_index = m.sample()
        probas = probas[[i for i in range(batch_size)], city_index]

        return probas, city_index.view(-1, 1)
