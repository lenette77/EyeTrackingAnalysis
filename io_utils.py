import os
import pandas as pd


def load_data(surface_path):
    """Load a CSV file containing fixation data for one surface."""
    try:
        data = pd.read_csv(surface_path)
        print("Data loaded successfully")
        print(f"Surface shape: {data.shape}")
        print(f"Surface columns: {list(data.columns)}")
        return data
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        print("File not found.")
        return None


def normalize_fixation_cols(df):
    """Normalize column names to match script expectations."""
    data = df.copy()
    data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')
    rename_map = {
        'fixation_id': 'fixationid',
    }
    data = data.rename(columns=rename_map)
    return data


def load_surface_fixations(surfaces_dir, surfaces, allow_missing=False):
    """Load and normalize per-surface fixation files."""
    surface_data_map = {}
    for surface in surfaces:
        path = os.path.join(surfaces_dir, surface["file"])
        if not os.path.exists(path):
            if allow_missing:
                print(f"Missing surface file; skipping: {path}")
                continue
            return None

        df = load_data(path)
        if df is None:
            if allow_missing:
                continue
            return None

        surface_data_map[surface["label"]] = normalize_fixation_cols(df)

    if not surface_data_map:
        return None
    return surface_data_map


def load_all_fixations(data_root):
    """Load the combined fixations file if it exists."""
    all_fixations_path = os.path.join(data_root, "fixations.csv")
    if os.path.exists(all_fixations_path):
        return normalize_fixation_cols(pd.read_csv(all_fixations_path))
    print(f"Note: all-fixations file not found at: {all_fixations_path}")
    return None


def load_gaze_positions(path):
    """Load continuous gaze positions (e.g., gaze_positions.csv)."""
    try:
        df = pd.read_csv(path)
        print(f"Loaded gaze positions: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Gaze file not found: {path}")
        return None
