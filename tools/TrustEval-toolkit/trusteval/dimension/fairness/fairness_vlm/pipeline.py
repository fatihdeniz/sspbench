import os
from src.utils import colored_print as print
from .preference import PreferenceGenerator
from .stereotype import StereotypeGenerator

def run(base_dir=None):
    print("Running StereotypeGenerator ...")
    os.makedirs(base_dir, exist_ok=True)
    print(f"Using base folder path: {base_dir}")
    stereotype_generator = StereotypeGenerator(base_dir,samples=10)
    stereotype_generator.process()
    
    print("Running PreferenceGenerator ...")
    preference_generator = PreferenceGenerator(base_dir, sample_size=5)
    preference_generator.process()
    
    print("All dataset generation finished.", color="GREEN")

    
    
