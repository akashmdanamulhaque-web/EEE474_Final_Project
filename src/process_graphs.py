import os
import sys
import glob
import torch

# Dynamically add the current directory (src) to Python's search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from audio_features import AudioFeatureExtractor
from graph_builder import build_music_graph

RAW_DIR = r"C:\Users\Laptop Castle\Documents\gnn-bert-music-context\data\raw"
PROCESSED_DIR = r"C:\Users\Laptop Castle\Documents\gnn-bert-music-context\data\processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)
extractor = AudioFeatureExtractor()

# Find audio files in extracted subfolders
audio_files = glob.glob(os.path.join(RAW_DIR, "**", "*.mp3"), recursive=True)
if not audio_files:
    audio_files = glob.glob(os.path.join(RAW_DIR, "**", "*.wav"), recursive=True)

print(f"Found {len(audio_files)} audio files.")

processed_count = 0
for audio_path in audio_files:
    if processed_count >= 20:
        break
    try:
        node_feats = extractor.process_track(audio_path)
        graph_data = build_music_graph(node_feats, similarity_threshold=0.7)
        
        save_path = os.path.join(PROCESSED_DIR, f"example_graph_{processed_count+1:02d}.pt")
        torch.save(graph_data, save_path)
        print(f"Saved [{processed_count+1}/20]: {save_path}")
        processed_count += 1
    except Exception as e:
        print(f"Skipping {audio_path}: {e}")

print(f"\nDone! Saved {processed_count} graph files in {PROCESSED_DIR}")