import os
import re
from collections import defaultdict

def calculate_emotion_vectors_ce_weighted(file_path):

    emotions = ["neutral", "anger", "sadness", "happiness",  "excited", "frustration", "fear", "surprise", "disgust"]
    emotion_keywords = ['neu', 'ang', 'sad', 'hap', 'exc']
    
    with open(file_path ,"r") as file:
        results = {}
        data = file.read()
        blocks = data.strip().split("\n\n")

        for block in blocks:
            lines = block.split('\n')
            emotion_4 = lines[0].split()[4]
            if emotion_4 in emotion_keywords:

                session_name = lines[0].split()[3]
                emotion_counts = defaultdict(float)


                labels = re.findall(r"(C-E\d):\s+(.*)", block, re.MULTILINE)
                valid_labels = []
                for _, emotions_raw in labels:

                    emotions_raw = emotions_raw.split("(")[0]
                    emotions_list = [e.strip().lower() for e in emotions_raw.split(";") if e.strip()]
                    if "other" in emotions_list:
                        valid_labels.append(None)
                    else:
                        valid_labels.append(emotions_list)
                    num_emotions = len(emotions_list)
                valid_count = len([label for label in valid_labels if label is not None])

                if valid_count == 3:
                    weight_per_label = 1  
                elif valid_count == 2:
                    weight_per_label = 1.5  
                elif valid_count == 1:
                    weight_per_label = 3  
                else:
                    weight_per_label = 0 

                for emotions_list in valid_labels:
                    if emotions_list is not None:
                        weight = weight_per_label / len(emotions_list)  
                        for emotion in emotions_list:
                            emotion_counts[emotion] += weight  
                vector = tuple(emotion_counts[emotion] for emotion in emotions)
                results[session_name] = {"emotion": emotion_4, "vector": vector}
    return results

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    total = 0
    for root, _, files in os.walk(input_folder):
        for file_name in files:
            if file_name.endswith(".txt"):
                input_file = os.path.join(root, file_name)
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                os.makedirs(output_dir, exist_ok=True)

                output_file = os.path.join(output_dir, file_name)
                results = calculate_emotion_vectors_ce_weighted(input_file)

                with open(output_file, "w", encoding="utf-8") as out_file:
                    count = 0
                    for session_name, data in results.items():
                        count += 1
                        out_file.write(f"Session: {session_name}\n")
                        out_file.write(f"Emotion:{data['emotion']}\n")
                        out_file.write(f"Vector: {data['vector']}\n\n")
                    total += count


input_folder = "/IEMOCAP/label"

output_folder = "/IEMOCAP/label_5531"

process_folder(input_folder, output_folder)

