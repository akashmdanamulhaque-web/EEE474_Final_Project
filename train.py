import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import random_split
from torch_geometric.data import Batch
from transformers import AutoTokenizer

from src.dataloader import get_dataloader
from src.gnn_model import GNNBertClassifier

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    batch_size = 32
    epochs = 15
    learning_rate = 1e-4

    # 1. Initialize Tokenizer & Full Dataset
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Load entire dataset
    full_dataset = get_dataloader(
        processed_dir=r'.\data\processed',
        csv_file=r'.\data\annotations_final.csv',
        batch_size=batch_size,
        shuffle=True
    )

    # 2. Initialize Model, Loss Function, Optimizer
    model = GNNBertClassifier(num_classes=188).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    os.makedirs(r'.\models', exist_ok=True)
    best_loss = float('inf')

    print("Starting full training run...")

    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (graphs, texts, targets) in enumerate(full_dataset):
            optimizer.zero_grad()

            if isinstance(graphs, (list, tuple)):
                batch_graphs = Batch.from_data_list(list(graphs)).to(device)
            else:
                batch_graphs = graphs.to(device)

            targets = targets.to(device)

            text_inputs = tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors='pt'
            ).to(device)

            logits = model(batch_graphs, text_inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(full_dataset):
                print(f"Epoch [{epoch}/{epochs}] | Batch [{batch_idx + 1}/{len(full_dataset)}] | Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(full_dataset)
        print(f"--- Epoch [{epoch}/{epochs}] Finished | Average Loss: {epoch_loss:.4f} ---")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), r'.\models\best_model.pt')
            print(f"Saved best model checkpoint (Loss: {best_loss:.4f})")

if __name__ == '__main__':
    train()