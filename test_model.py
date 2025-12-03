<<<<<<< HEAD
# test_model.py
"""
Test script for the trained model
"""
import sys
from pathlib import Path
from predictor import CricketShotPredictor
from config import config

def test_single_video():
    """Test prediction on a single video"""
    # Get video path from command line or use default
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        print("Usage: python test_model.py <video_path>")
        print("Example: python test_model.py sample_video.mp4")
        return
    
    if not Path(video_path).exists():
        print(f"Video file not found: {video_path}")
        return
    
    print(f"Testing model on: {video_path}")
    print("-" * 40)
    
    # Load predictor
    try:
        predictor = CricketShotPredictor()
        predictor.load_model(config.OUTPUT_DIR)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Make prediction
    try:
        result = predictor.predict_video(video_path)
        
        if result['error']:
            print(f"Prediction failed: {result['message']}")
        else:
            print(f"Predicted Shot: {result['predicted_class'].replace('_', ' ').title()}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Description: {result['description']}")
            
            print("\nTop 3 Predictions:")
            for i, pred in enumerate(result['top_3_predictions'], 1):
                shot_name = pred['class'].replace('_', ' ').title()
                print(f"{i}. {shot_name}: {pred['probability']:.4f}")
            
            print("\nAll Probabilities:")
            sorted_probs = sorted(result['all_probabilities'].items(), 
                                key=lambda x: x[1], reverse=True)
            for class_name, prob in sorted_probs:
                shot_name = class_name.replace('_', ' ').title()
                print(f"  {shot_name}: {prob:.4f}")
                
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
=======
# test_model.py
"""
Test script for the trained model
"""
import sys
from pathlib import Path
from predictor import CricketShotPredictor
from config import config

def test_single_video():
    """Test prediction on a single video"""
    # Get video path from command line or use default
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        print("Usage: python test_model.py <video_path>")
        print("Example: python test_model.py sample_video.mp4")
        return
    
    if not Path(video_path).exists():
        print(f"Video file not found: {video_path}")
        return
    
    print(f"Testing model on: {video_path}")
    print("-" * 40)
    
    # Load predictor
    try:
        predictor = CricketShotPredictor()
        predictor.load_model(config.OUTPUT_DIR)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Make prediction
    try:
        result = predictor.predict_video(video_path)
        
        if result['error']:
            print(f"Prediction failed: {result['message']}")
        else:
            print(f"Predicted Shot: {result['predicted_class'].replace('_', ' ').title()}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Description: {result['description']}")
            
            print("\nTop 3 Predictions:")
            for i, pred in enumerate(result['top_3_predictions'], 1):
                shot_name = pred['class'].replace('_', ' ').title()
                print(f"{i}. {shot_name}: {pred['probability']:.4f}")
            
            print("\nAll Probabilities:")
            sorted_probs = sorted(result['all_probabilities'].items(), 
                                key=lambda x: x[1], reverse=True)
            for class_name, prob in sorted_probs:
                shot_name = class_name.replace('_', ' ').title()
                print(f"  {shot_name}: {prob:.4f}")
                
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
>>>>>>> 55cf882ae12d7b7a383dc56a61ff28ccc63d8322
    test_single_video()