# MVST/datasets/custom_list.py
import numpy as np
import torch
import librosa
import pandas as pd
from torch.utils.data import Dataset

class CustomListDataset(Dataset):
    """
    Lee un CSV con columnas:
      - filepath (ruta al WAV)
      - label    (string de clase)
      - split    (train/valid/test)   [opcional: se filtra por args.split]
      - mode, patient_id              [metadatos opcionales]
    Devuelve (logmel [1, n_mels, T], label_id, meta_dict) para MVST.
    """
    def __init__(
        self,
        csv_path: str,
        split: str = None,
        resample_to: int = 16000,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 320,  # ~20 ms @16k
        fmin: int = 0,
        fmax: int = None,
        pad_seconds: float = 10.0,  # recorta/paddea a duración fija
        zscore: bool = True,
        mono: bool = True,
    ):
        self.df = pd.read_csv(csv_path)
        if split is not None and "split" in self.df.columns:
            self.df = self.df[self.df["split"].astype(str)==str(split)]
        self.df = self.df.reset_index(drop=True)

        # mapeo de etiquetas a enteros (estable y reproducible)
        labels = sorted(self.df["label"].astype(str).unique())
        self.label_to_id = {lab:i for i,lab in enumerate(labels)}
        self.id_to_label = {i:lab for lab,i in self.label_to_id.items()}

        self.resample_to = resample_to
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.pad_samples = int(round((pad_seconds or 0.0) * resample_to))
        self.zscore = zscore
        self.mono = mono

    def __len__(self):
        return len(self.df)

    def _load_audio(self, path):
        y, sr = librosa.load(path, sr=self.resample_to, mono=self.mono)
        # recorte/padding a longitud fija (opcional pero MVST lo agradece)
        if self.pad_samples > 0:
            if len(y) < self.pad_samples:
                y = np.pad(y, (0, self.pad_samples-len(y)), mode="constant")
            else:
                y = y[:self.pad_samples]
        if self.zscore:
            y = (y - np.mean(y)) / (np.std(y) + 1e-8)
        return y, self.resample_to

    def _logmel(self, y, sr):
        S = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax
        )
        S_db = librosa.power_to_db(S, ref=np.max)  # log-mel
        return S_db.astype(np.float32)  # [n_mels, T]

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        path = r["filepath"]
        y, sr = self._load_audio(path)
        mel = self._logmel(y, sr)                  # [M, T]
        x = torch.from_numpy(mel).unsqueeze(0)     # [1, M, T] -> canal único (AST usa [B, 1, M, T])
        lab_id = int(self.label_to_id[str(r["label"])])
        meta = {
            "filepath": path,
            "label": str(r["label"]),
            "patient_id": str(r.get("patient_id","")),
            "mode": str(r.get("mode","")),
        }
        return x, lab_id, meta

