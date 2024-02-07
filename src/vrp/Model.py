import time

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from Decoder import Decoder
from Encoder import TransformerEncoder


class AttentionModel(nn.Module):
    def __init__(self, embedding_dim, n_layers, n_head, dim_feedforward, C, dropout=0.1):
        super(AttentionModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.n_head = n_head
        self.dim_feedforward = dim_feedforward
        self.decode_mode = "greedy"
        self.dropout = dropout
        self.C = C
        self.input_dim = 2

        self.demand_embedding = nn.Linear(1, self.embedding_dim)

        self.city_embedding = nn.Linear(self.input_dim, self.embedding_dim)

        self.encoder = TransformerEncoder(self.n_layers, self.n_head, self.embedding_dim,
                                          self.dim_feedforward, self.dropout)
        self.decoder = Decoder(self.n_head, self.embedding_dim, self.decode_mode, self.C)

        self.accelerator = Accelerator()
        self.encoder, self.decoder = self.accelerator.prepare(self.encoder, self.decoder)

    def forward(self, inputs):
        """

        :param inputs : (locations, demands,capacities)
               (locations : [batch_size, seq_len, input_dim],
                demands : [batch_size, seq_len, 1],
                capacities : [batch_size])

        :return: raw_logits : [batch_size, seq_len, seq_len],
                 log_prob : [batch_size],
                 solutions : [batch_size, seq_len]
        """

        inputs, demands, capacities = inputs
        dem = demands.unsqueeze(-1)

        data = self.encoder(self.city_embedding(inputs) + self.demand_embedding(dem))

        raw_logits, log_prob, solution = self.decoder((data, demands, capacities))

        return raw_logits, log_prob, solution

    def set_decode_mode(self, mode):
        self.decode_mode = mode
        self.decoder.decode_mode = mode

    def test(self, data: DataLoader, decode_mode="greedy"):
        tour_lengths = torch.tensor([])
        self.eval()
        self.set_decode_mode(decode_mode)
        cpu = time.time()

        for batch_id, batch in enumerate(tqdm(data)):
            locations, demands, capacities = batch
            inputs = (locations, demands, capacities.float())
            _, _, solution = self(inputs)

            btl = self.compute(inputs, solution)
            tour_lengths = torch.cat((tour_lengths, btl), dim=0)

        cpu = time.time() - cpu
        return {
            "tour_lengths": tour_lengths,
            "avg_tl": tour_lengths.mean().item(),
            "cpu": cpu
        }

    def compute(self, instance_data, route):

        batch_size = route.size(0)
        length = torch.FloatTensor(torch.zeros(batch_size, 1, device=self.accelerator.device))
        locations, demands, capacities = instance_data
        for batch in range(batch_size):
            # Ref : Thibaut Vidal, Split algorithm in O(n) for the capacitated vehicle routing problem
            # https://arxiv.org/pdf/1508.02759.pdf
            nb_nodes = len(route[batch]) - 1
            distance_to_depot = torch.tensor(
                [torch.norm(locations[batch, int(route[batch, route[batch, i]])] - locations[batch, 0]) for i in
                 range(nb_nodes + 1)]
            , device=self.accelerator.device)
            distance_to_next = torch.tensor([
                torch.sqrt(torch.sum((locations[batch, route[batch, i]] - locations[batch, route[batch, i + 1]]) ** 2))

                if i < nb_nodes else -1
                for i in range(nb_nodes + 1)], device=self.accelerator.device)
            potential = torch.tensor([0.0] + [float('inf')] * nb_nodes, device=self.accelerator.device)
            pred = torch.tensor([-1] * (nb_nodes + 1), device=self.accelerator.device)
            sum_distance = torch.zeros(nb_nodes + 1, device=self.accelerator.device)
            sum_load = torch.zeros(nb_nodes + 1, device=self.accelerator.device)

            for i in range(1, nb_nodes + 1):
                sum_load[i] = sum_load[i - 1] + demands[batch, i]
                sum_distance[i] = sum_distance[i - 1] + distance_to_next[i - 1]

            queue = [0]

            for i in range(1, nb_nodes + 1):
                potential[i] = potential[queue[0]] + sum_distance[i] - sum_distance[queue[0] + 1] + distance_to_depot[
                    queue[0] + 1] + distance_to_depot[i]
                pred[i] = queue[0]

                if i < nb_nodes:
                    if (sum_load[queue[-1]] != sum_load[i]) or \
                            (potential[i] + distance_to_depot[i + 1] <= potential[queue[-1]]
                             + distance_to_depot[queue[-1] + 1]
                             + sum_distance[i + 1]
                             - sum_distance[queue[-1] + 1]):

                        while len(queue) > 0 and \
                                (potential[i] + distance_to_depot[i + 1] <
                                 potential[queue[-1]]
                                 + distance_to_depot[queue[-1] + 1]
                                 + sum_distance[i + 1]
                                 - sum_distance[queue[-1] + 1]):
                            queue.pop()
                        queue.append(i)
                    while sum_load[i + 1] - sum_load[queue[0]] > capacities[batch]:
                        queue.pop(0)

            length[batch] = potential[nb_nodes]

        return length.view(-1)
