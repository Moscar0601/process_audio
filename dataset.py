import torch
from torch.utils.data import Dataset
from utils import process_audio  
import os
import re

class AudioSegmentDataset(Dataset):
    def __init__(self, data_list):
        self.items = []
        for item in data_list:
            
            # audio path
            audio_path = item["Session"]
            session_name = os.path.basename(audio_path)
            audio_path += ".wav"
            
            # label
            match = re.search(r"Ses0*(\d+)", session_name)
            session_num = int(match.group(1))
            label = item["Soft"]
                
            segments, labels = process_audio(audio_path, label)

            for seg, lbl in zip(segments, labels):
                self.items.append((seg, lbl))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        waveform, label = self.items[idx]
        return waveform.squeeze(0), label
