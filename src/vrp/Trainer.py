import csv
import time

import numpy as np
import torch
from scipy.stats import ttest_rel
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import plot
from Baseline import Baseline
from Model import AttentionModel
from data.VRPDataset import VRPDataset
from accelerate import Accelerator


class Trainer:
    def __init__(self, graph_size, n_epochs, batch_size, nb_train_samples,
                 nb_val_samples, n_layers, n_heads, embedding_dim,
                 dim_feedforward, C, dropout, learning_rate
                 ):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.graph_size = graph_size
        self.nb_train_samples = nb_train_samples
        self.nb_val_samples = nb_val_samples
        self.accelerator = Accelerator()

        self.model = AttentionModel(embedding_dim, n_layers, n_heads,
                                    dim_feedforward,
                                    C, dropout)  # embedding, encoder, decoder

        # This ia a rollout baseline

        baseline_model = AttentionModel(embedding_dim, n_layers, n_heads,
                                        dim_feedforward,
                                        C, dropout)
        self.baseline = Baseline(baseline_model, graph_size, nb_val_samples)

        self.baseline.load_state_dict(self.model.state_dict())

        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

        self.baseline, self.optimizer = self.accelerator.prepare(self.baseline, self.optimizer)

        log_file_name = "{}-{}-logs.csv".format("vrp", graph_size)

        f = open(log_file_name, 'w', newline='')
        self.log_file = csv.writer(f, delimiter=",")

        header = ["epoch", "losses_per_batch", "avg_tl_batch_train", "avg_tl_epoch_train", "avg_tl_epoch_val"]
        self.log_file.writerow(header)

    def train(self):
        validation_dataset = VRPDataset(size=self.graph_size, num_samples=self.nb_val_samples)
        print("Validation dataset created with {} samples".format(len(validation_dataset)))
        validation_dataloader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0,
                                           pin_memory=True)
        losses = []
        avg_tour_length_batch = []
        avg_tour_length_epoch = []
        avg_tl_epoch_val = []
        for epoch in range(self.n_epochs):
            cpu = time.time()

            all_tour_lengths = torch.tensor([], dtype=torch.float32, device=self.accelerator.device)

            # Put model in train mode!
            self.model.set_decode_mode("sample")
            self.model, self.optimizer, validation_dataloader = self.accelerator.prepare(self.model, self.optimizer,
                                                                                         validation_dataloader)
            self.baseline.model = self.accelerator.prepare(self.baseline.model)
            self.model.train()

            # Generate new training data for each epoch
            train_dataset = VRPDataset(size=self.graph_size, num_samples=self.nb_train_samples)

            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0,
                                          pin_memory=True)
            nb_batches = len(validation_dataloader)

            for batch_id, batch in enumerate(tqdm(train_dataloader)):
                locations, demands, capacities = batch

                inputs = (locations.to(self.accelerator.device), demands.to(self.accelerator.device),
                          capacities.float().to(self.accelerator.device))

                _, log_prob, solution = self.model(inputs)

                with torch.no_grad():
                    tour_lengths = self.model.compute(inputs, solution)
                    baseline_tour_lengths = self.baseline.evaluate(inputs, False)

                    advantage = tour_lengths - baseline_tour_lengths[0:len(tour_lengths)]

                loss = advantage * (-log_prob)
                loss = loss.mean()

                self.optimizer.zero_grad(set_to_none=True)
                # loss.backward()
                self.accelerator.backward(loss)
                self.optimizer.step()

                # save data for plot
                losses.append(loss.item())
                avg_tour_length_batch.append(tour_lengths.mean().item())
                all_tour_lengths = torch.cat((all_tour_lengths, tour_lengths), dim=0)

            avg_tour_length_epoch.append(all_tour_lengths.mean().item())

            print(
                "\nEpoch: {}\t\nAverage tour length model : {}\nAverage tour length baseline : {}\n".format(
                    epoch + 1, tour_lengths.mean(), baseline_tour_lengths.mean()
                ))

            print("Validation and rollout update check\n")
            # t-test :
            self.model.set_decode_mode("greedy")
            self.baseline.set_decode_mode("greedy")
            self.model.eval()
            self.baseline.eval()
            with torch.no_grad():
                rollout_tl = torch.tensor([], dtype=torch.float32)
                policy_tl = torch.tensor([], dtype=torch.float32)
                for batch_id, batch in enumerate(tqdm(validation_dataloader)):
                    locations, demands, capacities = batch

                    inputs = (locations, demands, capacities.float())

                    _, _, solution = self.model(inputs)

                    tour_lengths = self.model.compute(inputs, solution)
                    baseline_tour_lengths = self.baseline.evaluate(inputs, False)

                    rollout_tl = torch.cat((rollout_tl, baseline_tour_lengths.view(-1).cpu()), dim=0)
                    policy_tl = torch.cat((policy_tl, tour_lengths.view(-1).cpu()), dim=0)

                rollout_tl = rollout_tl.cpu().numpy()
                policy_tl = policy_tl.cpu().numpy()

                avg_ptl = np.mean(policy_tl)
                avg_rtl = np.mean(rollout_tl)

                avg_tl_epoch_val.append(avg_ptl.item())

                cpu = time.time() - cpu
                print(
                    "CPU: {}\n"
                    "Loss: {}\n"
                    "Average tour length by policy: {}\n"
                    "Average tour length by rollout: {}\n".format(cpu, loss, avg_ptl, avg_rtl))

                self.log_file.writerow([epoch, losses[-nb_batches:],
                                        avg_tour_length_batch[-nb_batches:],
                                        avg_tour_length_epoch[-1],
                                        avg_ptl.item()
                                        ])

                if (avg_ptl - avg_rtl) < 0:
                    # t-test
                    _, pvalue = ttest_rel(policy_tl, rollout_tl)
                    pvalue = pvalue / 2  # one-sided ttest [refer to the original implementation]
                    if pvalue < 0.05:
                        print("Rollout network update...\n")
                        self.baseline.load_state_dict(self.model.state_dict())
                        self.baseline.reset()
                        print("Generate new validation dataset\n")

                        validation_dataset = VRPDataset(size=self.graph_size, num_samples=self.nb_val_samples)

                        validation_dataloader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=True,
                                                           num_workers=0, pin_memory=True)

            model_name = "RL_{}{}_Epoch_{}.pt".format("vrp", self.graph_size, epoch + 1)
            torch.save({
                "epoch": epoch,
                "model": self.model.state_dict(),
                "baseline": self.baseline.state_dict(),
                "optimizer": self.optimizer.state_dict()
            }, model_name)

        plot.plot_stats(losses, "{}-RL-Losses per batch {}".format("vrp", self.graph_size), "Batch", "Loss")
        plot.plot_stats(avg_tour_length_epoch,
                        "{} Average tour length per epoch train {}".format("vrp", self.graph_size),
                        "Epoch", "Average tour length")
        plot.plot_stats(avg_tour_length_batch,
                        "{} Average tour length per batch train {}".format("vrp", self.graph_size),
                        "Batch", "Average tour length")
        plot.plot_stats(avg_tl_epoch_val,
                        "{} Average tour length per epoch validation {}".format("vrp", self.graph_size),
                        "Epoch", "Average tour length")
