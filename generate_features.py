import numpy as np
import mediapipe as mp
from features import extract_features_from_image

mp_hands = mp.solutions.hands

# txt_file: path to file containing image paths and labels
def process_split (txt_file, X_name, y_name):

    print("Starting processing", txt_file)

    X = [] # to store feature vectors (each entry will be a 10-number array)
    y = [] # to store labels like A, B, C, etc
    dropped = 0 # to count how many images were dropped due to no hand detected

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:  

        with open(txt_file, "r") as f:  
            lines = f.readlines() # each line is like: raw_dataset/asl_alphabet_train/A/image1.jpg,A

        for line in lines:
            path, label = line.strip().split(",")
            
            print("Processing:", path)

            features = extract_features_from_image(path, hands) # extract the 10 features from the image

            if features is None:
                dropped += 1
                continue

            X.append(features)
            y.append(label)
        
    X = np.array(X)
    y = np.array(y)

    np.save(X_name, X)
    np.save(y_name, y)

    if len(X) % 1000 == 0:
        print("Processed:", len(X))

    print(f"{txt_file} done")
    print("Saved:", len(X))
    print("Dropped:", dropped)


if __name__ == "__main__":
    process_split("train.txt", "X_train.npy", "y_train.npy")
    process_split("val.txt", "X_val.npy", "y_val.npy")
    process_split("test.txt", "X_test.npy", "y_test.npy")