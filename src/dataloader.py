import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class MusicContextDataset(Dataset):
    def __init__(self, processed_dir, csv_file=None):
        self.processed_dir = processed_dir
        self.file_names = sorted([f for f in os.listdir(processed_dir) if f.endswith(".pt")])
        
        self.annotations = None
        if csv_file and os.path.exists(csv_file):
            # Detect whether the file is tab-delimited or comma-delimited
            try:
                df = pd.read_csv(csv_file, sep='\t')
                if len(df.columns) <= 1:
                    df = pd.read_csv(csv_file, sep=',')
            except Exception:
                df = pd.read_csv(csv_file, sep=',')
            
            # Identify track/clip ID column
            id_col = df.columns[0]
            for col in ['clip_id', 'track_id', 'id']:
                if col in df.columns:
                    id_col = col
                    break
            
            # Clean index keys
            df[id_col] = df[id_col].astype(str).str.strip()
            df = df.set_index(id_col)
            
            # Exclude non-target metadata columns
            meta_cols = ['mp3_path', 'title', 'artist', 'album', 'genre', 'path']
            label_cols = [c for c in df.columns if c.lower() not in meta_cols]
            
            # Cast all label columns to float32
            df_labels = df[label_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
            
            self.annotations = df_labels
            print(f"Loaded annotations: {len(self.annotations)} rows across {self.annotations.shape[1]} target label columns.")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = os.path.join(self.processed_dir, file_name)
        
        data = torch.load(file_path, weights_only=False)
        
        if isinstance(data, dict):
            graph = data['graph']
            text = data.get('text', 'music context audio track')
        else:
            graph = data
            text = getattr(data, 'text', 'music context audio track')
        
        target = None
        if self.annotations is not None:
            raw_id = os.path.splitext(file_name)[0]  # e.g., "000002"
            clean_id = str(int(raw_id)) if raw_id.isdigit() else raw_id  # e.g., "2"
            
            possible_keys = [file_name, raw_id, clean_id, f"{raw_id}.mp3", f"{clean_id}.mp3"]
            
            for key in possible_keys:
                if key in self.annotations.index:
                    target = self.annotations.loc[key].values
                    if target.ndim > 1:
                        target = target[0]
                    break

        if target is None:
            target = getattr(data, 'target', torch.zeros(188))
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target, dtype=torch.float32)
        else:
            target = torch.tensor(target, dtype=torch.float32)

        return graph, text, target

def collate_fn(batch):
    graphs = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    targets = torch.stack([item[2] for item in batch])
    return graphs, texts, targets

def get_dataloader(processed_dir, csv_file=None, batch_size=16, shuffle=True):
    dataset = MusicContextDataset(processed_dir, csv_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)