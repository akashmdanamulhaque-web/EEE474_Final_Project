import torch
import torch.nn as nn
from transformers import BertModel
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNEncoder(nn.Module):
    def __init__(self, in_dim=140, hidden_dim=256, out_dim=256):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)  # in_dim is 140 to match your node features
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch=None):
        x = self.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        return x

class GNNBertClassifier(nn.Module):
    def __init__(self, num_classes=188, gnn_in_dim=140, gnn_out_dim=256, bert_model_name='bert-base-uncased'):
        super().__init__()
        self.gnn = GNNEncoder(in_dim=gnn_in_dim, hidden_dim=256, out_dim=gnn_out_dim)
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
            
        combined_dim = gnn_out_dim + self.bert.config.hidden_size  # 256 + 768 = 1024
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, batch_graphs, text_inputs):
        if isinstance(batch_graphs, (list, tuple)):
            batch_graphs = Batch.from_data_list(list(batch_graphs))

        if hasattr(batch_graphs, 'batch') and batch_graphs.batch is not None:
            gnn_out = self.gnn(batch_graphs.x, batch_graphs.edge_index, batch_graphs.batch)
        else:
            gnn_out = self.gnn(batch_graphs.x, batch_graphs.edge_index)

        bert_out = self.bert(**text_inputs).last_hidden_state[:, 0, :]  # CLS token representation

        combined = torch.cat([gnn_out, bert_out], dim=1)
        logits = self.classifier(combined)
        return logits