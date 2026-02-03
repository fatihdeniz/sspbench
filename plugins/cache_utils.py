import os
import pandas as pd

import utils
from models.llm_base import EmptyModelResponseError
    
def load_cache(input_path, output_path, output_cols, input_lines=True):
    """
    Load existing cached results and merge with input data.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file (cache)
        output_col: Column names for output data (e.g., 'response', 'label', 'pred')
    
    Returns:
        df: Merged dataframe
        indices_to_process: List of indices that need processing
    """
    df = pd.read_json(input_path, lines=input_lines)
    for output_col in output_cols:
        if output_col not in df.columns:
            df[output_col] = None
    
    if os.path.exists(output_path):
        print(f"Loading existing results from {output_path}")
        existing_df = pd.read_json(output_path, lines=True)
        
        for output_col in output_cols:
            if output_col in existing_df.columns:
                df[output_col] = existing_df[output_col]
        
    mask_needs_processing = df[output_cols[0]].isna() | df[output_cols[0]].apply(utils.is_no_response)
    indices_to_process = df[mask_needs_processing].index.tolist()
    return df, indices_to_process


def validate_and_save(df, output_path, output_col, threshold=0.01):
    """
    Validate response quality and save if acceptable.
    
    Args:
        df: Dataframe to save
        responses: List of newly generated responses
        output_path: Path to save results
        output_col: Column name for output data
        threshold: Maximum acceptable empty response ratio (default: 1%)
    
    Raises:
        EmptyModelResponseError: If empty response ratio exceeds threshold
    """
    df.to_json(output_path, orient="records", lines=True)
    
    empty_count, total_count, empty_ratio, _ = utils.compute_no_response_stats(df[output_col].tolist())
    
    if total_count and empty_ratio > threshold:
        raise EmptyModelResponseError(
            f"Model returned {empty_count} empty responses out of {total_count} prompts "
            f"({empty_ratio:.2%})."
        )
    
    print(f"Successfully stored results to {output_path}")