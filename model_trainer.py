<<<<<<< HEAD
# model_trainer.py
"""
Random Forest model training for cricket shot classification
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from config import config
import json

class RandomForestTrainer:
    def __init__(self):
        self.model = None
        self.best_params = None
        
    def create_model(self):
        """Create Random Forest model with default parameters"""
        self.model = RandomForestClassifier(
            n_estimators=config.N_ESTIMATORS,
            max_depth=config.MAX_DEPTH,
            min_samples_split=config.MIN_SAMPLES_SPLIT,
            min_samples_leaf=config.MIN_SAMPLES_LEAF,
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        )
        return self.model
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning"""
        print("Performing hyperparameter tuning...")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1)
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, 
            scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return self.model
    
    def train_model(self, X_train, y_train, X_val, y_val, tune_hyperparameters=False):
        """Train the Random Forest model"""
        print("Training Random Forest model...")
        
        if tune_hyperparameters:
            self.hyperparameter_tuning(X_train, y_train)
        else:
            self.create_model()
            self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_predictions = self.model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_predictions)
        
        print(f"Validation accuracy: {val_accuracy:.4f}")
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return val_accuracy
    
    def evaluate_model(self, X_test, y_test, class_names):
        """Evaluate model on test set"""
        print("Evaluating model on test set...")
        
        test_predictions = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_test, test_predictions, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, test_predictions)
        
        return test_accuracy, test_predictions, cm
    
    def save_model(self, output_dir):
        """Save the trained model"""
        model_path = f"{output_dir}/models/random_forest_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save best parameters if available
        if self.best_params:
            with open(f"{output_dir}/models/best_params.json", 'w') as f:
                json.dump(self.best_params, f, indent=2)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load a trained model"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if self.model is None:
            return None
=======
# model_trainer.py
"""
Random Forest model training for cricket shot classification
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from config import config
import json

class RandomForestTrainer:
    def __init__(self):
        self.model = None
        self.best_params = None
        
    def create_model(self):
        """Create Random Forest model with default parameters"""
        self.model = RandomForestClassifier(
            n_estimators=config.N_ESTIMATORS,
            max_depth=config.MAX_DEPTH,
            min_samples_split=config.MIN_SAMPLES_SPLIT,
            min_samples_leaf=config.MIN_SAMPLES_LEAF,
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        )
        return self.model
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning"""
        print("Performing hyperparameter tuning...")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1)
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, 
            scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return self.model
    
    def train_model(self, X_train, y_train, X_val, y_val, tune_hyperparameters=False):
        """Train the Random Forest model"""
        print("Training Random Forest model...")
        
        if tune_hyperparameters:
            self.hyperparameter_tuning(X_train, y_train)
        else:
            self.create_model()
            self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_predictions = self.model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_predictions)
        
        print(f"Validation accuracy: {val_accuracy:.4f}")
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return val_accuracy
    
    def evaluate_model(self, X_test, y_test, class_names):
        """Evaluate model on test set"""
        print("Evaluating model on test set...")
        
        test_predictions = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_test, test_predictions, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, test_predictions)
        
        return test_accuracy, test_predictions, cm
    
    def save_model(self, output_dir):
        """Save the trained model"""
        model_path = f"{output_dir}/models/random_forest_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save best parameters if available
        if self.best_params:
            with open(f"{output_dir}/models/best_params.json", 'w') as f:
                json.dump(self.best_params, f, indent=2)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load a trained model"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if self.model is None:
            return None
>>>>>>> 55cf882ae12d7b7a383dc56a61ff28ccc63d8322
        return self.model.feature_importances_