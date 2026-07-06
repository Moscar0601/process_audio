import os 
import re
import json


def get_label_dict(label_path):
    label_dict = {}
    emotions = ["neutral", "angry", "sad", "happy", "excited", "frustrated", 
                "fearful", "surprised", "disgusted", "depressed"]
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ";" not in line:
                continue

            parts = [p.strip() for p in line.split(';') if p.strip()]
        
            if parts[0].endswith('.avi') and parts[0].startswith("UTD"):
                current_session = parts[0].replace(".avi", "")
                if current_session not in label_dict:
                    label_dict[current_session] = []
                continue

            if current_session is None:
                continue

            uid = parts[0]
            if uid.endswith(".avi"):
                continue

            main_label = parts[1].lower()

            multi_labels_raw = parts[2]
            raw_labels = [label.strip().lower() for label in multi_labels_raw.split(",")]
            valid = not any("other" in l for l in raw_labels) and main_label != "other"
            labels = [label for label in raw_labels if any(e in label for e in emotions)]

            if main_label != 'other' and main_label not in labels:
                labels.append(main_label)

            label_dict[current_session].append({
                    "uid": uid,
                    "labels": labels,
                    "valid": valid
            })
    return label_dict


def cal_label(label_dict, audio_path_complete):
    emotions = ["neutral", "angry", "sad", "happy", "excited", "frustrated", 
                "fearful", "surprised", "disgusted", "depressed"]
    emotion_to_index = {emotion: i for i, emotion in enumerate(emotions)}
    
    vector = [0.0] * len(emotions)
    valid_annotator_count = 0
    session_name = os.path.basename(audio_path_complete).split(".wav")[0]
    session_name = "UTD-" + session_name[4:]

    if session_name not in label_dict:
        print("false", session_name)

    for annotator in label_dict[session_name]:
        if annotator["valid"]:
            valid_annotator_count += 1
            labels = annotator["labels"]
            n = len(labels)
            if n == 0:
                continue
            weight = 1.0 / n
            for label in labels:
                if label in emotion_to_index:
                    idx = emotion_to_index[label]
                    vector[idx] += weight
    if valid_annotator_count > 0:
        vector = [v / valid_annotator_count for v in vector]
    return vector

msp_improve_audio_path = '/MSP-IMPROVE/Audio'
label_path = '/MSP-IMPROVE/Audio/Evalution.txt'

session_list = [f'session{i}' for i in range(1, 7)]
session_paths = [os.path.join(msp_improve_audio_path, s) for s in session_list]
label_dict = get_label_dict(label_path)

all_data = []
for session_name, session_path in zip(session_list, session_paths):
    for folder in os.listdir(session_path):
        session_person = os.path.join(session_path, folder)
        for sub in os.listdir(session_person):
            session_person_type = os.path.join(session_person, sub)
            for audio_path in os.listdir(session_person_type):
                audio_path_complete = os.path.join(session_person_type, audio_path)
                emotion = cal_label(label_dict, audio_path_complete)
                all_data.append({
                    'path': audio_path_complete,
                    'emotion': emotion,
                    'session': session_name
                })
print(len(all_data))


folds = {}
for i, test_session in enumerate(session_list):
    fold_name = f'fold{i+1}'
    train_sessions = [s for s in session_list if s != test_session]

    train_data = [ {'X': d['path'], 'Y': d['emotion']} for d in all_data if d['session'] in train_sessions ]
    test_data = [ {'X': d['path'], 'Y': d['emotion']} for d in all_data if d['session'] == test_session ]

    folds[fold_name] = {
        'train': train_data,
        'test': test_data
    }

output_file = '/MSP-IMPROVE/Audio/LOSO6.txt'
raw_json = json.dumps(folds, indent=2, separators=(',', ': '))
def compress_vector(match):
    array = match.group(0)
    array_inline = re.sub(r'\s+', ' ', array)
    array_inline = re.sub(r'\[\s+', '[', array_inline)
    array_inline = re.sub(r'\s+\]', ']', array_inline)
    return array_inline
compressed_json = re.sub(r'"Y"\s*:\s*\[[^\]]*\]', compress_vector, raw_json)

with open(output_file, 'w') as f:
    f.write(compressed_json)

print("finish")
