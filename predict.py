import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Batch
from transformers import AutoTokenizer
from src.gnn_model import GNNBertClassifier

def predict_tags(graph_file_path, text_description, top_k=5, threshold=0.3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Tag Names
    df = pd.read_csv(r'.\data\annotations_final.csv')
    tag_columns = [col for col in df.columns if col not in ['path', 'track_id', 'text', 'caption']]

    # Load Model
    model = GNNBertClassifier(num_classes=len(tag_columns))
    model.load_state_dict(torch.load(r'.\models\best_model.pt', map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Load Graph Tensor Data
    graph_data = torch.load(graph_file_path, map_location=device)
    batch_graph = Batch.from_data_list([graph_data]).to(device)

    # Tokenize Text Input
    text_inputs = tokenizer(
        [text_description],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    ).to(device)

    # Predict
    with torch.no_grad():
        logits = model(batch_graph, text_inputs)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    # Get Top-K Predicted Tags
    top_indices = np.argsort(probs)[-top_k:][::-1]

    print(f"\n--- Inference Results for: {text_description} ---")
    for idx in top_indices:
        prob = probs[idx]
        status = "[HIGH CONFIDENCE]" if prob >= threshold else "[LOW CONFIDENCE]"
        print(f"{status} Tag: {tag_columns[idx]:<35} | Probability: {prob:.4f}")

if __name__ == '__main__':
    # Test on an existing processed graph file from data/processed
    sample_graph = r'.\data\processed\0.pt'  # Replace with actual file path if needed
    sample_text = "Upbeat electric guitar melody with energetic drums"
    
    predict_tags(sample_graph, sample_text, top_k=5)