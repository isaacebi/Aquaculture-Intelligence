import os
import sys
import cv2
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from torch.utils.data import Dataset

FILE_PATH = os.path.abspath(__file__)
MODULE_DIR = os.path.dirname(FILE_PATH)
PROJECT_DIR = os.path.dirname(MODULE_DIR)

sys.path.append(PROJECT_DIR)

from src import GetPath

class MotionHistoryImage(Dataset):
    def __init__(self, mhi_dir="mhi_right_tail", frame_cutoff=15, experiment="all", pca_model=PCA()):
        # Define directory
        self.mhi_dir = os.path.join(GetPath().data(), 'preprocess', mhi_dir)
        if not os.path.exists(self.mhi_dir):
            print(f"Please check data directory. The expected structured is '<current_folder>/data/preprocessed/{mhi_dir}'")

        # MHI frame cutoff
        self.frame_cutoff = frame_cutoff

        # Experiment batch
        self.experiment = experiment

        # Initialization
        self.image_paths = []
        self.image_folders = []
        self.labels = []
        self.start_frames = {
            'level_0': [],
            'level_1': [],
            'level_2': [],
            'level_3': [],
            'level_4': [],
            'level_5': [],
            'level_6': [],
            'level_7': [],
            'level_8': [],
            'level_9': [],
            'level_10': [],
        }

        # attach path to module
        self._load_data()
        if len(self.image_paths) != len(self.labels):
            raise ValueError("Number of images and labels do not match.")

        # For feature decomposition
        self.pca_model = pca_model
        
    def _load_data(self):
        for dir in self.start_frames.keys():
            level_dir = os.path.join(self.mhi_dir, 'samples', dir)
            for frame_folder in os.listdir(level_dir):
                if frame_folder == f"{self.frame_cutoff}_frame":
                    for file in os.listdir(os.path.join(level_dir, frame_folder)):

                        # Train on all data
                        if self.experiment=='all' and file.endswith(".jpg"):
                            image_path = os.path.join(level_dir, frame_folder, file)
                            self.image_paths.append(image_path)
                            self.labels.append(dir)
                            self.start_frames[dir].append(file.split("_")[-1].split(".")[0])

                        elif self.experiment in file:
                            image_path = os.path.join(level_dir, frame_folder, file)
                            self.image_paths.append(image_path)
                            self.labels.append(dir)
                            self.start_frames[dir].append(file.split("_")[-1].split(".")[0])

    def __len__(self):
        return len(self.image_paths)
        
    # Feature extraction
    def get_feature(self, image: np.ndarray):
        features = np.array([])

        # Feature reduction
        features = np.append(features, self.pca_model.transform([image.flatten()]))

        return features

    def __getitem__(self, idx, size=(640, 480), normalize=True, only_feature=False):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.resize(image, size) # updated data automatically size to 224 square
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if normalize:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        label = self.labels[idx]

        # will only return feature instead of the image and label
        if only_feature:
            return self.get_feature(image), label

        return image, label
    
    def get_path(self, idx):
        return self.image_folders[idx]

    def get_frame(self, label):
        return self.start_frames[label]


class Fish_Dataset_Binary(Dataset):
    def __init__(self, frame_cutoff=15, transform=None):
        self.frame_cutoff = frame_cutoff
        self.transform = transform
        self.image_paths = []
        self.image_folders = []
        self.labels = []
        self.start_frames = {
            'normal': [], 'abnormal': []
        }

        self.data_dir = "D:/fish_behavior/data/preprocess/mhi_binary"

        self._load_normal_data()
        self._load_abnormal_data()

        if len(self.image_paths) != len(self.labels):
            raise ValueError("Number of images and labels do not match.")

    def _load_normal_data(self):
        normal_dir = os.path.join(self.data_dir, "normal")
        for folder in os.listdir(normal_dir):
            if folder == f"{self.frame_cutoff}_frame":
                for file in os.listdir(os.path.join(normal_dir, folder)):
                    if file.endswith(".jpg"):
                        image_path = os.path.join(normal_dir, folder, file)
                        self.image_paths.append(image_path)
                        self.labels.append("normal")

            if folder.endswith(".txt"):
                self._load_start_frames("normal", os.path.join(normal_dir, folder))

    def _load_abnormal_data(self):
        abnormal_dir = os.path.join(self.data_dir, "abnormal")
        for folder in os.listdir(abnormal_dir):
            if folder == f"{self.frame_cutoff}_frame":
                for file in os.listdir(os.path.join(abnormal_dir, folder)):
                    if file.endswith(".jpg"):
                        image_path = os.path.join(abnormal_dir, folder, file)
                        self.image_paths.append(image_path)
                        self.labels.append("abnormal")

            if folder.endswith(".txt"):
                self._load_start_frames("abnormal", os.path.join(abnormal_dir, folder))

    def _load_start_frames(self, label, file_path):
        with open(file_path, 'r') as f:
            for line in f.readlines():
                self.start_frames[label].append(line.strip())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224)) # updated data automatically size to 224 square
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.normalize(image, None, -1, 1, cv2.NORM_MINMAX)

        if self.transform:
            image = self.transform(image=image)['image']

        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        label = self.labels[idx]
        return image, label
    

    def get_path(self, idx):
        return self.image_folders[idx]

    def get_frame(self, label):
        return self.start_frames[label]
    
    if __name__ == "__main__":
        mhi_dataset = MotionHistoryImage()
        print(f"All experiment total data: {len(mhi_dataset)}")

        slice_mhi_dataset = MotionHistoryImage(experiment="B1")
        print(f"Experiment B1 total data: {len(slice_mhi_dataset)}")