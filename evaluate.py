import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Batch
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score

from src.dataloader import get_dataloader
from src.gnn_model import GNNBertClassifier

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Tag Names from Annotations CSV
    df = pd.read_csv(r'.\data\annotations_final.csv')
    tag_columns = [col for col in df.columns if col not in ['path', 'track_id', 'text', 'caption']]

    # 2. Dataloader & Model Setup
    dataloader = get_dataloader(
        processed_dir=r'.\data\processed',
        csv_file=r'.\data\annotations_final.csv',
        batch_size=32,
        shuffle=False
    )
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    model = GNNBertClassifier(num_classes=len(tag_columns))
    checkpoint_path = r'.\models\best_model.pt'
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for graphs, texts, targets in dataloader:
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
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # Calculate Multi-Label Metrics
    macro_roc = roc_auc_score(all_targets, all_preds, average='macro')
    macro_pr = average_precision_score(all_targets, all_preds, average='macro')

    print("\n================ EVALUATION RESULTS ================")
    print(f"Total Samples Evaluated: {len(all_preds)}")
    print(f"Total Target Classes:    {all_preds.shape[1]}")
    print(f"Macro ROC-AUC Score:     {macro_roc:.4f}")
    print(f"Macro PR-AUC Score:      {macro_pr:.4f}")
    print("====================================================\n")

    # Sample Human-Readable Output
    top5_idx = np.argsort(all_preds[0])[-5:][::-1]
    print("Sample Output for Sample #0:")
    for idx in top5_idx:
        print(f"  - Tag: {tag_columns[idx]:<35} | Prob: {all_preds[0][idx]:.4f}")

if __name__ == '__main__':
    evaluate()