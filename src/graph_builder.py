import torch
from torch_geometric.data import Data

def build_music_graph(node_features, similarity_threshold=0.7):
    num_nodes = node_features.shape[0]
    edge_index = []
    edge_attr = []

    norm_features = node_features / (torch.norm(node_features, dim=1, keepdim=True) + 1e-8)
    sim_matrix = torch.mm(norm_features, norm_features.T)

    for i in range(num_nodes):
        if i > 0:
            edge_index.append([i, i - 1])
            edge_attr.append(1.0)
        if i < num_nodes - 1:
            edge_index.append([i, i + 1])
            edge_attr.append(1.0)
            
        for j in range(num_nodes):
            if i != j and sim_matrix[i, j].item() > similarity_threshold:
                edge_index.append([i, j])
                edge_attr.append(sim_matrix[i, j].item())

    if len(edge_index) == 0:
        edge_index = [[i, i] for i in range(num_nodes)]
        edge_attr = [1.0] * num_nodes

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32).unsqueeze(1)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)