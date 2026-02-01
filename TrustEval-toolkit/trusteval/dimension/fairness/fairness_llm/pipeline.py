import os
from src.utils import colored_print as print
from .preference import PreferenceGenerator
from .stereotype import StereotypeGenerator
from .disparagement import DisparagementGenerator




def run(base_dir=None, subset=['stereotype', 'preference', 'disparagement']):
    if 'stereotype' in subset:
        print("Running StereotypeGenerator ...")
        stereotype_generator = StereotypeGenerator(base_dir)
        stereotype_generator.run()
    
    if 'preference' in subset:
        print("Running PreferenceGenerator ...")
        preference_generator = PreferenceGenerator(base_dir)
        preference_generator.run()
    
    if 'disparagement' in subset:
        print("Running DisparagementGenerator ...")
        disparagement_generator = DisparagementGenerator(base_dir)
        disparagement_generator.run()
    
    print("All dataset generation finished.", color="GREEN")

    
        
