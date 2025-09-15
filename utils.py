import random
import numpy as np
import torch
import torchaudio
import ast
import torch.nn as nn
import torchaudio.transforms as T

def process_audio(audio_path, label, sample_rate=16000, segment_seconds=7):

    labels = []
    waveform, sr = torchaudio.load(audio_path)
    
    # Get the total number of samples in the audio waveform
    total_samples = waveform.shape[1]
    
   # Calculate the number of samples per segment
    segment_len = segment_seconds * sample_rate
    
    # Calculate the number of segments the audio will be split into
    num_segments = (total_samples + segment_len - 1) // segment_len  
    
    # Padding
    padded_waveform = torch.zeros((1, num_segments * segment_len))
    padded_waveform[0, :total_samples] = waveform[0]

    segments = padded_waveform.view(num_segments, segment_len)

    for i in range(num_segments):
        labels.append(label)

    return segments, labels
