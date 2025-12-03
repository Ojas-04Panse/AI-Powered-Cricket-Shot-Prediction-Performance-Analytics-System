<<<<<<< HEAD
# main_trainer.py
"""
Main training script for Random Forest cricket shot classifier
"""
import numpy as np
from pathlib import Path
import json
from config import config, SHOT_DESCRIPTIONS
from data_processor import DataProcessor
from model_trainer import RandomForestTrainer
from visualizer import ResultVisualizer

def main():
    print("Cricket Shot Classification using Random Forest")
    print("=" * 50)
    
    # Create output directories
    config.create_directories()
    
    # Initialize components
    data_processor = DataProcessor()
    trainer = RandomForestTrainer()
    visualizer = ResultVisualizer(config.OUTPUT_DIR)
    
    try:
        # Load and process dataset
        features, labels, file_paths = data_processor.load_dataset()
        
        # Prepare data splits
        X_train, X_val, X_test, y_train, y_val, y_test = data_processor.prepare_data(
            features, labels
        )
        
        # Get class names
        class_names = data_processor.label_encoder.classes_
        
        # Create visualizations for dataset
        visualizer.plot_class_distribution(y_train, class_names, "Training Set Distribution")
        
        # Train model
        val_accuracy = trainer.train_model(
            X_train, y_train, X_val, y_val, 
            tune_hyperparameters=True
        )
        
        # Evaluate model
        test_accuracy, test_predictions, cm = trainer.evaluate_model(
            X_test, y_test, class_names
        )
        
        # Calculate per-class accuracies
        per_class_accuracy = {}
        for i, class_name in enumerate(class_names):
            class_mask = y_test == i
            if np.sum(class_mask) > 0:
                class_acc = np.mean(test_predictions[class_mask] == y_test[class_mask])
                per_class_accuracy[class_name] = float(class_acc)
        
        # Create visualizations
        visualizer.plot_confusion_matrix(cm, class_names, "Test Set Confusion Matrix")
        
        # Feature importance
        importance = trainer.get_feature_importance()
        if importance is not None:
            visualizer.plot_feature_importance(importance)
        
        # Results summary
        results = {
            'test_accuracy': float(test_accuracy),
            'val_accuracy': float(val_accuracy),
            'per_class_accuracy': per_class_accuracy,
            'n_estimators': config.N_ESTIMATORS,
            'max_depth': config.MAX_DEPTH,
            'min_samples_split': config.MIN_SAMPLES_SPLIT,
            'total_samples': len(features),
            'num_classes': len(class_names),
            'num_features': X_train.shape[1],
            'class_names': class_names.tolist()
        }
        
        if trainer.best_params:
            results.update(trainer.best_params)
        
        # Save results
        with open(f"{config.OUTPUT_DIR}/results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary visualization
        visualizer.create_results_summary(results)
        
        # Save model and preprocessors
        trainer.save_model(config.OUTPUT_DIR)
        data_processor.save_preprocessors(config.OUTPUT_DIR)
        
        # Print final results
        print("\n" + "=" * 50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Best Parameters: {trainer.best_params}")
        print(f"Results saved to: {config.OUTPUT_DIR}")
        
        print("\nPer-class accuracies:")
        for class_name, acc in per_class_accuracy.items():
            print(f"  {class_name.replace('_', ' ').title()}: {acc:.4f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
=======
# main_trainer.py
"""
Main training script for Random Forest cricket shot classifier
"""
import numpy as np
from pathlib import Path
import json
from config import config, SHOT_DESCRIPTIONS
from data_processor import DataProcessor
from model_trainer import RandomForestTrainer
from visualizer import ResultVisualizer

def main():
    print("Cricket Shot Classification using Random Forest")
    print("=" * 50)
    
    # Create output directories
    config.create_directories()
    
    # Initialize components
    data_processor = DataProcessor()
    trainer = RandomForestTrainer()
    visualizer = ResultVisualizer(config.OUTPUT_DIR)
    
    try:
        # Load and process dataset
        features, labels, file_paths = data_processor.load_dataset()
        
        # Prepare data splits
        X_train, X_val, X_test, y_train, y_val, y_test = data_processor.prepare_data(
            features, labels
        )
        
        # Get class names
        class_names = data_processor.label_encoder.classes_
        
        # Create visualizations for dataset
        visualizer.plot_class_distribution(y_train, class_names, "Training Set Distribution")
        
        # Train model
        val_accuracy = trainer.train_model(
            X_train, y_train, X_val, y_val, 
            tune_hyperparameters=True
        )
        
        # Evaluate model
        test_accuracy, test_predictions, cm = trainer.evaluate_model(
            X_test, y_test, class_names
        )
        
        # Calculate per-class accuracies
        per_class_accuracy = {}
        for i, class_name in enumerate(class_names):
            class_mask = y_test == i
            if np.sum(class_mask) > 0:
                class_acc = np.mean(test_predictions[class_mask] == y_test[class_mask])
                per_class_accuracy[class_name] = float(class_acc)
        
        # Create visualizations
        visualizer.plot_confusion_matrix(cm, class_names, "Test Set Confusion Matrix")
        
        # Feature importance
        importance = trainer.get_feature_importance()
        if importance is not None:
            visualizer.plot_feature_importance(importance)
        
        # Results summary
        results = {
            'test_accuracy': float(test_accuracy),
            'val_accuracy': float(val_accuracy),
            'per_class_accuracy': per_class_accuracy,
            'n_estimators': config.N_ESTIMATORS,
            'max_depth': config.MAX_DEPTH,
            'min_samples_split': config.MIN_SAMPLES_SPLIT,
            'total_samples': len(features),
            'num_classes': len(class_names),
            'num_features': X_train.shape[1],
            'class_names': class_names.tolist()
        }
        
        if trainer.best_params:
            results.update(trainer.best_params)
        
        # Save results
        with open(f"{config.OUTPUT_DIR}/results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary visualization
        visualizer.create_results_summary(results)
        
        # Save model and preprocessors
        trainer.save_model(config.OUTPUT_DIR)
        data_processor.save_preprocessors(config.OUTPUT_DIR)
        
        # Print final results
        print("\n" + "=" * 50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Best Parameters: {trainer.best_params}")
        print(f"Results saved to: {config.OUTPUT_DIR}")
        
        print("\nPer-class accuracies:")
        for class_name, acc in per_class_accuracy.items():
            print(f"  {class_name.replace('_', ' ').title()}: {acc:.4f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
>>>>>>> 55cf882ae12d7b7a383dc56a61ff28ccc63d8322
    main()