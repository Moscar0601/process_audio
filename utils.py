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
    # 获取音频波形数据的总样本数
    total_samples = waveform.shape[1]
    # 计算音频片段的样本数
    segment_len = segment_seconds * sample_rate
    # 音频分割成的片段数量
    num_segments = (total_samples + segment_len - 1) // segment_len  
    # 填充音频数据
    padded_waveform = torch.zeros((1, num_segments * segment_len))
    padded_waveform[0, :total_samples] = waveform[0]

    segments = padded_waveform.view(num_segments, segment_len)

    for i in range(num_segments):
        labels.append(label)

    return segments, labels