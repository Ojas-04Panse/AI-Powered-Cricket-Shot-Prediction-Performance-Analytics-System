<<<<<<< HEAD
# visualizer.py
"""
Visualization utilities for cricket shot classification results
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

class ResultVisualizer:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_confusion_matrix(self, cm, class_names, title="Confusion Matrix"):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        save_path = self.viz_dir / f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {save_path}")
    
    def plot_feature_importance(self, importance_scores, title="Feature Importance"):
        """Plot feature importance"""
        # Get top 20 features
        top_indices = np.argsort(importance_scores)[-20:]
        top_scores = importance_scores[top_indices]
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(top_scores)), top_scores)
        plt.yticks(range(len(top_scores)), [f'Feature_{i}' for i in top_indices])
        plt.xlabel('Importance Score')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.viz_dir / f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance plot saved to {save_path}")
    
    def plot_class_distribution(self, labels, class_names, title="Class Distribution"):
        """Plot class distribution"""
        unique, counts = np.unique(labels, return_counts=True)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar([class_names[i] for i in unique], counts, 
                      color=plt.cm.Set3(np.linspace(0, 1, len(unique))))
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Cricket Shot Class')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        save_path = self.viz_dir / f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Class distribution plot saved to {save_path}")
    
    def create_results_summary(self, results_dict):
        """Create a summary visualization of all results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cricket Shot Classification Results Summary', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Accuracy comparison
        ax1 = axes[0, 0]
        accuracy_types = ['Train', 'Validation', 'Test']
        accuracy_values = [
            results_dict.get('train_accuracy', 0),
            results_dict.get('val_accuracy', 0),
            results_dict.get('test_accuracy', 0)
        ]
        
        bars1 = ax1.bar(accuracy_types, accuracy_values, 
                       color=['blue', 'orange', 'green'])
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        for bar, acc in zip(bars1, accuracy_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Per-class accuracy (if available)
        if 'per_class_accuracy' in results_dict:
            ax2 = axes[0, 1]
            class_names = list(results_dict['per_class_accuracy'].keys())
            class_accuracies = list(results_dict['per_class_accuracy'].values())
            
            bars2 = ax2.bar(range(len(class_names)), class_accuracies)
            ax2.set_title('Per-Class Accuracy')
            ax2.set_ylabel('Accuracy')
            ax2.set_xticks(range(len(class_names)))
            ax2.set_xticklabels([name.replace('_', '\n') for name in class_names], 
                              rotation=45, ha='right')
            ax2.set_ylim(0, 1)
        
        # Plot 3: Model parameters info
        ax3 = axes[1, 0]
        ax3.axis('off')
        
        model_info = f"""
        Model: Random Forest
        
        Parameters:
        • N Estimators: {results_dict.get('n_estimators', 'N/A')}
        • Max Depth: {results_dict.get('max_depth', 'N/A')}
        • Min Samples Split: {results_dict.get('min_samples_split', 'N/A')}
        
        Dataset:
        • Total Samples: {results_dict.get('total_samples', 'N/A')}
        • Classes: {results_dict.get('num_classes', 'N/A')}
        • Features: {results_dict.get('num_features', 'N/A')}
        """
        
        ax3.text(0.1, 0.5, model_info, transform=ax3.transAxes, fontsize=11,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # Plot 4: Training summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
        Training Summary
        
        Best Test Accuracy: {results_dict.get('test_accuracy', 0):.3f}
        
        Top Performing Classes:
        """
        
        if 'per_class_accuracy' in results_dict:
            sorted_classes = sorted(results_dict['per_class_accuracy'].items(), 
                                  key=lambda x: x[1], reverse=True)
            for i, (class_name, acc) in enumerate(sorted_classes[:3]):
                summary_text += f"\n{i+1}. {class_name.replace('_', ' ').title()}: {acc:.3f}"
        
        ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        plt.tight_layout()
        save_path = self.viz_dir / "results_summary.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
=======
# visualizer.py
"""
Visualization utilities for cricket shot classification results
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

class ResultVisualizer:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_confusion_matrix(self, cm, class_names, title="Confusion Matrix"):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        save_path = self.viz_dir / f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {save_path}")
    
    def plot_feature_importance(self, importance_scores, title="Feature Importance"):
        """Plot feature importance"""
        # Get top 20 features
        top_indices = np.argsort(importance_scores)[-20:]
        top_scores = importance_scores[top_indices]
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(top_scores)), top_scores)
        plt.yticks(range(len(top_scores)), [f'Feature_{i}' for i in top_indices])
        plt.xlabel('Importance Score')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.viz_dir / f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance plot saved to {save_path}")
    
    def plot_class_distribution(self, labels, class_names, title="Class Distribution"):
        """Plot class distribution"""
        unique, counts = np.unique(labels, return_counts=True)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar([class_names[i] for i in unique], counts, 
                      color=plt.cm.Set3(np.linspace(0, 1, len(unique))))
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Cricket Shot Class')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        save_path = self.viz_dir / f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Class distribution plot saved to {save_path}")
    
    def create_results_summary(self, results_dict):
        """Create a summary visualization of all results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cricket Shot Classification Results Summary', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Accuracy comparison
        ax1 = axes[0, 0]
        accuracy_types = ['Train', 'Validation', 'Test']
        accuracy_values = [
            results_dict.get('train_accuracy', 0),
            results_dict.get('val_accuracy', 0),
            results_dict.get('test_accuracy', 0)
        ]
        
        bars1 = ax1.bar(accuracy_types, accuracy_values, 
                       color=['blue', 'orange', 'green'])
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        for bar, acc in zip(bars1, accuracy_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Per-class accuracy (if available)
        if 'per_class_accuracy' in results_dict:
            ax2 = axes[0, 1]
            class_names = list(results_dict['per_class_accuracy'].keys())
            class_accuracies = list(results_dict['per_class_accuracy'].values())
            
            bars2 = ax2.bar(range(len(class_names)), class_accuracies)
            ax2.set_title('Per-Class Accuracy')
            ax2.set_ylabel('Accuracy')
            ax2.set_xticks(range(len(class_names)))
            ax2.set_xticklabels([name.replace('_', '\n') for name in class_names], 
                              rotation=45, ha='right')
            ax2.set_ylim(0, 1)
        
        # Plot 3: Model parameters info
        ax3 = axes[1, 0]
        ax3.axis('off')
        
        model_info = f"""
        Model: Random Forest
        
        Parameters:
        • N Estimators: {results_dict.get('n_estimators', 'N/A')}
        • Max Depth: {results_dict.get('max_depth', 'N/A')}
        • Min Samples Split: {results_dict.get('min_samples_split', 'N/A')}
        
        Dataset:
        • Total Samples: {results_dict.get('total_samples', 'N/A')}
        • Classes: {results_dict.get('num_classes', 'N/A')}
        • Features: {results_dict.get('num_features', 'N/A')}
        """
        
        ax3.text(0.1, 0.5, model_info, transform=ax3.transAxes, fontsize=11,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # Plot 4: Training summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
        Training Summary
        
        Best Test Accuracy: {results_dict.get('test_accuracy', 0):.3f}
        
        Top Performing Classes:
        """
        
        if 'per_class_accuracy' in results_dict:
            sorted_classes = sorted(results_dict['per_class_accuracy'].items(), 
                                  key=lambda x: x[1], reverse=True)
            for i, (class_name, acc) in enumerate(sorted_classes[:3]):
                summary_text += f"\n{i+1}. {class_name.replace('_', ' ').title()}: {acc:.3f}"
        
        ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        plt.tight_layout()
        save_path = self.viz_dir / "results_summary.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
>>>>>>> 55cf882ae12d7b7a383dc56a61ff28ccc63d8322
        print(f"Results summary saved to {save_path}")