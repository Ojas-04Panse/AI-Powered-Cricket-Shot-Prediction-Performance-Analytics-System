<<<<<<< HEAD
# data_processor.py
"""
Data processing for cricket shot classification
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pose_extractor import PoseExtractor
from config import config
import pickle

class DataProcessor:
    def __init__(self):
        self.pose_extractor = PoseExtractor()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_dataset(self):
        """Load and process the cricket shot dataset"""
        print("Loading cricket shot dataset...")
        
        data_dir = Path(config.DATA_DIR)
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
        
        features = []
        labels = []
        file_paths = []
        
        # Process each class
        for class_name in config.CRICKET_SHOTS:
            class_dir = data_dir / class_name
            if not class_dir.exists():
                print(f"Warning: Class directory not found: {class_dir}")
                continue
                
            print(f"Processing class: {class_name}")
            
            # Find video files
            video_files = []
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                video_files.extend(class_dir.glob(ext))
            
            for video_file in video_files:
                print(f"  Processing: {video_file.name}")
                
                # Extract pose features
                pose_features = self.pose_extractor.extract_video_features(
                    str(video_file), config.NUM_FRAMES
                )
                
                if pose_features is not None:
                    # Create enhanced features
                    enhanced_features = self.pose_extractor.create_enhanced_features(pose_features)
                    
                    features.append(enhanced_features)
                    labels.append(class_name)
                    file_paths.append(str(video_file))
                else:
                    print(f"    Failed to extract features from {video_file.name}")
        
        print(f"Successfully processed {len(features)} videos")
        
        if len(features) == 0:
            raise ValueError("No features extracted from dataset")
        
        return np.array(features), labels, file_paths
    
    def prepare_data(self, features, labels):
        """Prepare data for training"""
        print("Preparing data for training...")
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, encoded_labels, 
            test_size=config.TEST_SIZE, 
            random_state=config.RANDOM_STATE,
            stratify=encoded_labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=config.VAL_SIZE / (1 - config.TEST_SIZE),
            random_state=config.RANDOM_STATE,
            stratify=y_temp
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training samples: {len(X_train_scaled)}")
        print(f"Validation samples: {len(X_val_scaled)}")
        print(f"Test samples: {len(X_test_scaled)}")
        
        return (X_train_scaled, X_val_scaled, X_test_scaled, 
                y_train, y_val, y_test)
    
    def save_preprocessors(self, output_dir):
        """Save scaler and label encoder"""
        with open(f"{output_dir}/models/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
            
        with open(f"{output_dir}/models/label_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)
    
    def load_preprocessors(self, output_dir):
        """Load scaler and label encoder"""
        with open(f"{output_dir}/models/scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
            
        with open(f"{output_dir}/models/label_encoder.pkl", 'rb') as f:
=======
# data_processor.py
"""
Data processing for cricket shot classification
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pose_extractor import PoseExtractor
from config import config
import pickle

class DataProcessor:
    def __init__(self):
        self.pose_extractor = PoseExtractor()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_dataset(self):
        """Load and process the cricket shot dataset"""
        print("Loading cricket shot dataset...")
        
        data_dir = Path(config.DATA_DIR)
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
        
        features = []
        labels = []
        file_paths = []
        
        # Process each class
        for class_name in config.CRICKET_SHOTS:
            class_dir = data_dir / class_name
            if not class_dir.exists():
                print(f"Warning: Class directory not found: {class_dir}")
                continue
                
            print(f"Processing class: {class_name}")
            
            # Find video files
            video_files = []
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                video_files.extend(class_dir.glob(ext))
            
            for video_file in video_files:
                print(f"  Processing: {video_file.name}")
                
                # Extract pose features
                pose_features = self.pose_extractor.extract_video_features(
                    str(video_file), config.NUM_FRAMES
                )
                
                if pose_features is not None:
                    # Create enhanced features
                    enhanced_features = self.pose_extractor.create_enhanced_features(pose_features)
                    
                    features.append(enhanced_features)
                    labels.append(class_name)
                    file_paths.append(str(video_file))
                else:
                    print(f"    Failed to extract features from {video_file.name}")
        
        print(f"Successfully processed {len(features)} videos")
        
        if len(features) == 0:
            raise ValueError("No features extracted from dataset")
        
        return np.array(features), labels, file_paths
    
    def prepare_data(self, features, labels):
        """Prepare data for training"""
        print("Preparing data for training...")
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, encoded_labels, 
            test_size=config.TEST_SIZE, 
            random_state=config.RANDOM_STATE,
            stratify=encoded_labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=config.VAL_SIZE / (1 - config.TEST_SIZE),
            random_state=config.RANDOM_STATE,
            stratify=y_temp
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training samples: {len(X_train_scaled)}")
        print(f"Validation samples: {len(X_val_scaled)}")
        print(f"Test samples: {len(X_test_scaled)}")
        
        return (X_train_scaled, X_val_scaled, X_test_scaled, 
                y_train, y_val, y_test)
    
    def save_preprocessors(self, output_dir):
        """Save scaler and label encoder"""
        with open(f"{output_dir}/models/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
            
        with open(f"{output_dir}/models/label_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)
    
    def load_preprocessors(self, output_dir):
        """Load scaler and label encoder"""
        with open(f"{output_dir}/models/scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
            
        with open(f"{output_dir}/models/label_encoder.pkl", 'rb') as f:
>>>>>>> 55cf882ae12d7b7a383dc56a61ff28ccc63d8322
            self.label_encoder = pickle.load(f)