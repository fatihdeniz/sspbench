"""
Data Utilities

Functions for loading and saving datasets.
"""

import json
from typing import List, Dict, Any
from pathlib import Path


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Load dataset from JSONL file.
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_dataset(data: List[Dict[str, Any]], file_path: str):
    """
    Save dataset to JSONL file.
    """
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')