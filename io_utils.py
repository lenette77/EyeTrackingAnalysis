import json
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


def _build_source_mtime_map(surfaces_dir, surfaces, data_root, allow_missing=False):
    files = []
    for surface in surfaces:
        path = os.path.join(surfaces_dir, surface["file"])
        if not os.path.exists(path):
            if allow_missing:
                continue
            return None
        files.append(path)

    all_fixations_path = os.path.join(data_root, "fixations.csv")
    if os.path.exists(all_fixations_path):
        files.append(all_fixations_path)

    return {path: os.path.getmtime(path) for path in files}


def load_cached_fixations(cache_dir, surfaces_dir, surfaces, data_root, allow_missing=False):
    data_path = os.path.join(cache_dir, "fixation_cache.pkl")
    meta_path = os.path.join(cache_dir, "fixation_cache_meta.json")

    current = _build_source_mtime_map(surfaces_dir, surfaces, data_root, allow_missing=allow_missing)
    if current is None:
        return None, None, False

    if not (os.path.exists(data_path) and os.path.exists(meta_path)):
        return None, None, False

    try:
        with open(meta_path, "r") as handle:
            meta = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None, None, False

    if meta.get("files") != current:
        return None, None, False

    try:
        cached = pd.read_pickle(data_path)
    except (OSError, ValueError):
        return None, None, False

    surface_data_map = cached.get("surface_data_map") if isinstance(cached, dict) else None
    all_fixations = cached.get("all_fixations") if isinstance(cached, dict) else None
    if surface_data_map is None:
        return None, None, False

    return surface_data_map, all_fixations, True


def save_cached_fixations(cache_dir, surfaces_dir, surfaces, data_root,
                          surface_data_map, all_fixations, allow_missing=False):
    current = _build_source_mtime_map(surfaces_dir, surfaces, data_root, allow_missing=allow_missing)
    if current is None:
        return

    os.makedirs(cache_dir, exist_ok=True)
    data_path = os.path.join(cache_dir, "fixation_cache.pkl")
    meta_path = os.path.join(cache_dir, "fixation_cache_meta.json")
    payload = {
        "surface_data_map": surface_data_map,
        "all_fixations": all_fixations
    }
    pd.to_pickle(payload, data_path)
    with open(meta_path, "w") as handle:
        json.dump({"files": current}, handle, indent=2)


def load_cached_analysis(cache_dir, surfaces_dir, surfaces, data_root, allow_missing=False):
    data_path = os.path.join(cache_dir, "analysis_cache.pkl")
    meta_path = os.path.join(cache_dir, "analysis_cache_meta.json")

    current = _build_source_mtime_map(surfaces_dir, surfaces, data_root, allow_missing=allow_missing)
    if current is None:
        return None, None, False

    if not (os.path.exists(data_path) and os.path.exists(meta_path)):
        return None, None, False

    try:
        with open(meta_path, "r") as handle:
            meta = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None, None, False

    if meta.get("files") != current:
        return None, None, False

    try:
        cached = pd.read_pickle(data_path)
    except (OSError, ValueError):
        return None, None, False

    aoi_data_map = cached.get("aoi_data_map") if isinstance(cached, dict) else None
    combined = cached.get("combined") if isinstance(cached, dict) else None
    if aoi_data_map is None or combined is None:
        return None, None, False

    return aoi_data_map, combined, True


def save_cached_analysis(cache_dir, surfaces_dir, surfaces, data_root,
                         aoi_data_map, combined, allow_missing=False):
    current = _build_source_mtime_map(surfaces_dir, surfaces, data_root, allow_missing=allow_missing)
    if current is None:
        return

    os.makedirs(cache_dir, exist_ok=True)
    data_path = os.path.join(cache_dir, "analysis_cache.pkl")
    meta_path = os.path.join(cache_dir, "analysis_cache_meta.json")
    payload = {
        "aoi_data_map": aoi_data_map,
        "combined": combined
    }
    pd.to_pickle(payload, data_path)
    with open(meta_path, "w") as handle:
        json.dump({"files": current}, handle, indent=2)
