# config.py
"""
Configuration file for Cricket Shot Classification using Random Forest
"""
import os
from pathlib import Path

class Config:
    # Dataset paths
    DATA_DIR = r"D:\Programming\hackx_project\cricketclips"
    OUTPUT_DIR = "./rf_cricket_outputs"
    
    # Your 7 cricket shot classes
    CRICKET_SHOTS = [
        "backfoot_punch",
        "cover_drive", 
        "cut",
        "defensive",
        "Loft",
        "Other",
        "pull"
    ]
    
    # Video processing parameters
    NUM_FRAMES = 15
    FRAME_SIZE = 224
    
    # Random Forest parameters
    N_ESTIMATORS = 200
    MAX_DEPTH = 15
    MIN_SAMPLES_SPLIT = 5
    MIN_SAMPLES_LEAF = 2
    RANDOM_STATE = 42
    
    # Data parameters
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    
    @classmethod
    def create_directories(cls):
        Path(cls.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.OUTPUT_DIR + "/models").mkdir(parents=True, exist_ok=True)
        Path(cls.OUTPUT_DIR + "/visualizations").mkdir(parents=True, exist_ok=True)

config = Config()

# Shot descriptions
SHOT_DESCRIPTIONS = {
    "backfoot_punch": "A stroke played off the back foot with straight bat",
    "cover_drive": "An elegant front foot drive through the covers",
    "cut": "A horizontal bat stroke played square on off side",
    "defensive": "A defensive stroke blocking the ball safely",
    "Loft": "An aggressive aerial stroke over fielders",
    "Other": "Other cricket strokes not in main categories",
    "pull": "A cross-batted stroke to short-pitched balls"
}