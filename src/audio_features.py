import os
import librosa
import numpy as np
import torch

class AudioFeatureExtractor:
    def __init__(self, sample_rate=22050, n_mels=128, n_chroma=12, seg_duration=5.0):
        self.sr = sample_rate
        self.n_mels = n_mels
        self.n_chroma = n_chroma
        self.seg_duration = seg_duration

    def process_track(self, audio_path):
        y, _ = librosa.load(audio_path, sr=self.sr)
        duration = librosa.get_duration(y=y, sr=self.sr)
        samples_per_seg = int(self.sr * self.seg_duration)
        num_segments = max(1, int(duration // self.seg_duration))
        
        segment_features = []
        for i in range(num_segments):
            start = i * samples_per_seg
            end = min(start + samples_per_seg, len(y))
            y_seg = y[start:end]
            
            if len(y_seg) < samples_per_seg:
                y_seg = np.pad(y_seg, (0, samples_per_seg - len(y_seg)))
            
            chroma = librosa.feature.chroma_stft(y=y_seg, sr=self.sr, n_chroma=self.n_chroma)
            mel = librosa.feature.melspectrogram(y=y_seg, sr=self.sr, n_mels=self.n_mels)
            log_mel = librosa.power_to_db(mel, ref=np.max)
            
            chroma_mean = np.mean(chroma, axis=1)
            mel_mean = np.mean(log_mel, axis=1)
            feature_vec = np.concatenate([chroma_mean, mel_mean])
            segment_features.append(feature_vec)
            
        return torch.tensor(np.array(segment_features), dtype=torch.float32)