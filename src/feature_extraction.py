import numpy as np
import pickle
import os

def extract_features(hand_landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

def process_data(gesture_names):
    data = []
    labels = []
    
    for gesture in gesture_names:
        with open(f'data/raw/{gesture}.pkl', 'rb') as f:
            while True:
                try:
                    landmarks = pickle.load(f)
                    features = extract_features(landmarks)
                    data.append(features)
                    labels.append(gesture)
                except EOFError:
                    break

    data = np.array(data)
    labels = np.array(labels)
    
    # Save processed data
    np.save('data/processed/X.npy', data)
    np.save('data/processed/y.npy', labels)

if __name__ == "__main__":
    gesture_names = ['thumbs_up', 'pointer']
    process_data(gesture_names)