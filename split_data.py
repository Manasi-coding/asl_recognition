# split the data into train and test sets
import os
import random

random.seed(42)

source_path = "raw_dataset/asl_alphabet_train/asl_alphabet_train"

# to store lines like: raw_dataset/asl_alphabet_train/A/image1.jpg,A
train_file = open("train.txt", "w")
val_file = open("val.txt", "w")
test_file = open("test.txt", "w")

for folder in os.listdir(source_path):

    # we don't want to include the folders "del", "space", and "nothing" in our dataset
    if folder in ["del", "space", "nothing"]:
        continue

    folder_path = os.path.join(source_path, folder) # construct the path to the folder
    images = os.listdir(folder_path) # list all images in the folder

    random.shuffle(images) # shuffle the images to ensure randomness

    total = len(images)
    train_count = int(0.7 * total) # 70% for training
    val_count = int(0.15 * total) # 15% for validation

    # slice the images into train, val, and test sets
    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    # each line in the txt file becomes: path/to/image.jpg,label
    for img in train_images:
        path = os.path.join(folder_path, img)
        train_file.write(f"{path},{folder}\n")
    for img in val_images:
        path = os.path.join(folder_path, img)
        val_file.write(f"{path},{folder}\n")
    for img in test_images: 
        path = os.path.join(folder_path, img)
        test_file.write(f"{path},{folder}\n")

train_file.close()
val_file.close()    
test_file.close()

print("Split complete")

print(len(open("train.txt").readlines()))
print(len(open("val.txt").readlines()))
print(len(open("test.txt").readlines()))