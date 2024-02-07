import os

import torch
from torch.utils.data import DataLoader

import plot
import train
from Model import AttentionModel
from data.VRPDataset import VRPDataset
from data.VRPLiteratureDataset import VRPLiteratureDataset
from data.VRPXML100Dataset import VRPXML100Dataset

if __name__ == "__main__":
    graph_size = 100
    batch_size = 100
    nb_test_samples = 1000
    test_samples = "xml100"  # "xml100", "random", "literature"
    n_layers = train.n_layers
    n_heads = train.n_heads
    embedding_dim = train.embedding_dim
    dim_feedforward = train.dim_feedforward
    decode_mode = "greedy"
    C = train.C
    dropout = train.dropout
    seed = 1234
    torch.manual_seed(seed)

    print("Generating {} samples of type {} and size {}".format(nb_test_samples, test_samples, graph_size))
    if test_samples == "xml100":
        test_dataset = VRPXML100Dataset(size=graph_size, num_samples=nb_test_samples, seed=seed)
    elif test_samples == "literature":
        # FIXME : allow to load instances with different sizes
        test_dataset = VRPLiteratureDataset(low=graph_size - 1, high=graph_size - 1)
    else:
        test_dataset = VRPDataset(size=graph_size, num_samples=nb_test_samples)

    print("Number of test samples : ", len(test_dataset))

    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=1)

    random_model = AttentionModel(embedding_dim, n_layers, n_heads, dim_feedforward, C, dropout)

    tour_lengths = {
        "random": random_model.test(test_dataloader)
    }

    datasets = os.listdir("../../pretrained/" + "vrp" + str(graph_size) + "/")
    for dataset in sorted(datasets, key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.endswith('.pt') else 0):
        if not dataset.endswith('.pt'): continue
        print("Testing model : ", dataset)
        data = torch.load("../../pretrained/vrp" + str(graph_size) + "/" + dataset)

        try:
            model = AttentionModel(embedding_dim, n_layers, n_heads, dim_feedforward, C, dropout)
            model.load_state_dict(data["model"])
            results = model.test(test_dataloader)
            tour_lengths[int(dataset.split('_')[-1].split('.')[0])] = results
            print('{} : {} in cpu = {}'.format(dataset, results["avg_tl"], results["cpu"]))
        except Exception as e:
            print(e)
            continue

    plot.plot_stats([results["avg_tl"] for dataset, results in tour_lengths.items()],
                    "{} Average tour length per epoch evaluation {}".format("vrp", graph_size),
                    "Epoch", "Average tour length")

    sorted_tour_lengths = sorted(tour_lengths.items(), key=lambda x: x[1]["avg_tl"])

    print('Sorted tour lengths per model')
    for model_name, results in sorted_tour_lengths:
        print('{} : {} in cpu = {}'.format(model_name, results["avg_tl"], results["cpu"]))
