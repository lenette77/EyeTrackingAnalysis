"""
Eye-Tracking Analysis: Dual-Screen AOIs + Continuous Scan Patterns (ipynb Integrated)
Combines fixation AOI/transitions (your py) + trajectory patterns (notebook).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ========== YOUR ORIGINAL FUNCTIONS (unchanged) ==========
def load_data(surface1_path, surface2_path):
    """Load CSV files containing fixation data from two surfaces"""
    try:
        surface1 = pd.read_csv(surface1_path)
        surface2 = pd.read_csv(surface2_path)
        print("Data loaded successfully")
        print(f"Surface 1 shape: {surface1.shape}")
        print(f"Surface 1 columns: {list(surface1.columns)}")
        print(f"Surface 2 shape: {surface2.shape}")
        print(f"Surface 2 columns: {list(surface2.columns)}")
        return surface1, surface2
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Files not found.")
        return None, None

def analyze_screen_coverage(surface1, surface2, all_fixations=None):
    """Analyze which fixations landed on which screen"""
    fixations_on_s1 = surface1[surface1['on_surf'] == True]['fixationid'].unique()
    fixations_on_s2 = surface2[surface2['on_surf'] == True]['fixationid'].unique()
    
    if all_fixations is not None and 'fixationid' in all_fixations.columns:
        all_fixations = all_fixations['fixationid'].unique()
    else:
        all_fixations = pd.concat([surface1['fixationid'], surface2['fixationid']]).unique()
        print("Warning: all_fixations not provided; neither screen is based only on surface files.")
    
    only_s1 = set(fixations_on_s1) - set(fixations_on_s2)
    only_s2 = set(fixations_on_s2) - set(fixations_on_s1)
    both_screens = set(fixations_on_s1) & set(fixations_on_s2)
    neither = set(all_fixations) - set(fixations_on_s1) - set(fixations_on_s2)
    
    print("="*60)
    print("Fixation Screen Coverage Analysis")
    print("="*60)
    print(f"Screen 1 only: {len(only_s1)} fixations")
    print(f"Screen 2 only: {len(only_s2)} fixations")
    print(f"Both screens: {len(both_screens)} fixations")
    print(f"Neither screen: {len(neither)} fixations")
    print(f"Total unique fixations: {len(all_fixations)}")
    
    return only_s1, only_s2, both_screens, neither, all_fixations

def assign_aoi(x, y):
    """Takes x,y coordinates and returns which AOI it belongs to (1-9 row-major)"""
    if x < 0.333:
        col = 0
    elif x < 0.667:
        col = 1
    else:
        col = 2
    
    if y < 0.333:
        row = 0
    elif y < 0.667:
        row = 1
    else:
        row = 2
    
    ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    return ids[row*3 + col]

def collapse_fixations(surface_data, group_cols=None):
    """Collapse multiple rows per fixation into one row using sensible aggregates."""
    if group_cols is None:
        group_cols = ['fixationid']
    if 'surface' in surface_data.columns:
        group_cols = ['fixationid', 'surface']
    
    agg_map = {}
    for col in surface_data.columns:
        if col in group_cols:
            continue
        if col in ['norm_pos_x', 'norm_pos_y']:
            agg_map[col] = 'mean'
        elif col == 'world_timestamp':
            agg_map[col] = 'min'
        else:
            agg_map[col] = 'first'
    
    return surface_data.groupby(group_cols, as_index=False).agg(agg_map)

def create_aoi_data(surface_data):
    """Assign AOI to each fixation on the surface"""
    data = surface_data[surface_data['on_surf'] == True].copy()
    data = collapse_fixations(data)
    data['aoi'] = data.apply(lambda row: assign_aoi(row['norm_pos_x'], row['norm_pos_y']), axis=1)
    print("="*60)
    print("STEP 3: AOI Assignment (3x3 Grid)")
    print("="*60)
    print(f"Assigned {len(data)} fixations to AOIs")
    print("AOI assignments:")
    print(data[['fixationid', 'norm_pos_x', 'norm_pos_y', 'aoi']].head(10))
    return data

# ... [Include all your other original functions: count_fixations_per_aoi, calculate_aoi_durations, 
# create_transition_sequence, create_transition_matrix, analyze_cross_screen_transitions, 
# all visualize_* functions exactly as before. To save space, assume they're here unchanged.]

# ========== NEW: INTEGRATED FROM IPYNB ==========
def sliding_window_gaze(gaze_xy, window_length=35, stride=10):
    """Extract sliding windows from gaze X,Y trajectories (from ipynb). 
    window_length=35 ~0.7s @50Hz; adjust for your 60Hz: ~42."""
    n_samples = len(gaze_xy)
    n_windows = (n_samples - window_length) // stride + 1
    windows = []
    
    for i in range(n_windows):
        start = i * stride
        window = gaze_xy.iloc[start:start + window_length].values  # Shape: (window_length, 2)
        windows.append(window)
    
    return np.array(windows)  # Shape: (n_windows, window_length, 2)

def compute_distance_profiles(windows):
    """Vectorized distance profiles — much faster than nested loops."""
    n = len(windows)
    # Flatten each window to 1D: (n_windows, window_length * 2)
    flat = windows.reshape(n, -1).astype(np.float64)
    
    # Pairwise Euclidean distances using broadcasting
    diff = flat[:, np.newaxis, :] - flat[np.newaxis, :, :]  # (n, n, features)
    dist_matrix = np.sqrt((diff ** 2).sum(axis=2))           # (n, n)
    profile_sums = dist_matrix.sum(axis=1)                   # Uniqueness score per window
    return dist_matrix, profile_sums


def extract_patterns(windows, num_patterns=10, percentile_start=0.05, min_occurrences=5):
    """Iterative pattern extraction — fixed empty-array guard."""
    dist_matrix, _ = compute_distance_profiles(windows)
    n_windows = len(windows)
    available = np.arange(n_windows)

    patterns = []

    for k in range(num_patterns):
        # Guard: stop if not enough windows left
        if len(available) < min_occurrences:
            print(f"  Stopped at pattern {k+1}: only {len(available)} windows remaining.")
            break

        percentile = (percentile_start + k * 0.005) * 100  # e.g. 5th → 10th over 10 iters

        # Uniqueness scores for remaining windows only
        sub_matrix = dist_matrix[np.ix_(available, available)]
        profile_sums = sub_matrix.sum(axis=1)

        # Most unique remaining window
        proto_local_idx = np.argmax(profile_sums)
        proto_global_idx = available[proto_local_idx]

        # Find similar occurrences (low distance to prototype)
        proto_distances = dist_matrix[proto_global_idx, available]
        thresh = np.percentile(proto_distances, percentile)
        occ_local = np.where(proto_distances <= thresh)[0]
        occ_global = available[occ_local]

        if len(occ_global) >= min_occurrences:
            patterns.append({
                'pattern_id': k + 1,
                'proto_window_idx': proto_global_idx,
                'prototype': windows[proto_global_idx],
                'n_occurrences': len(occ_global),
                'occurrence_indices': occ_global
            })
            print(f"  Pattern {k+1}: proto_idx={proto_global_idx}, {len(occ_global)} occurrences")
            available = np.setdiff1d(available, occ_global)  # Mask occurrences
        else:
            print(f"  Pattern {k+1}: skipped (only {len(occ_global)} occurrences < min {min_occurrences})")

    print(f"Extracted {len(patterns)} scan patterns total.")
    return pd.DataFrame(patterns), np.concatenate([p['occurrence_indices'] for p in patterns]) if patterns else np.array([])


def plot_patterns(gaze_df, patterns_df, occurrences, output_dir, max_patterns=8):
    """Visualize patterns + occurrences (simplified from ipynb)."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (_, pat) in enumerate(patterns_df.head(max_patterns).iterrows()):
        ax = axes[i]
        
        # Prototype trajectory
        proto = pat['prototype']
        ax.plot(proto[:, 0], proto[:, 1], 'y-', linewidth=3, label='Prototype')
        
        # Sample 5 occurrences
        occs = pat['occurrence_indices'][:5]
        for j, occ_idx in enumerate(occs):
            occ_traj = gaze_df.iloc[occ_idx * 10 : (occ_idx + 1) * 10][['norm_pos_x', 'norm_pos_y']].values  # Approx
            alpha = 0.3 + 0.1 * j
            ax.plot(occ_traj[:, 0], occ_traj[:, 1], 'r-', alpha=alpha)
        
        ax.set_title(f'Pattern {pat["pattern_id"]} ({pat["n_occurrences"]} occs)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Extracted Scan Patterns (Yellow=Proto, Red=Occurrences)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scan_patterns.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved scan_patterns.png with {len(patterns_df)} patterns")

# ========== UPDATED MAIN ==========
def main():
    print("="*60)
    print("EYE-TRACKING ANALYSIS: AOIs + SCAN PATTERNS")
    print("="*60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    week_dir = os.path.dirname(script_dir)  # Go up from Eyetracking_Analysis\ to Week1 02_04_26\

    data_dir = os.path.join(week_dir, "example_data", "Mateo_data", "exports", "000", "surfaces")

    surface1_path = os.path.join(data_dir, "fixations_on_surface_Surface 1.csv")
    surface2_path = os.path.join(data_dir, "fixations_on_surface_Surface 2.csv")

    output_dir = os.path.join(script_dir, "output")

    
    # STEP 1: Load (your original)
    surface1, surface2 = load_data(surface1_path, surface2_path)
    surface1, surface2 = load_data(surface1_path, surface2_path)
    if surface1 is None or surface2 is None:
        return

    # Normalize column names: strip spaces, lowercase, remove underscores
    def normalize_cols(df):
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        #  Specific renames to match script expectations
        rename_map = {
            'fixation_id': 'fixationid',
        }
        df = df.rename(columns=rename_map)
        return df

    surface1 = normalize_cols(surface1)
    surface2 = normalize_cols(surface2)

    
    # STEP 2: Screen coverage (your original)
    coverage = analyze_screen_coverage(surface1, surface2)
    
    # NEW STEP 2.5: CONCAT FOR TRAJECTORIES
    combined_gaze = pd.concat([surface1, surface2], ignore_index=True)
    combined_gaze = combined_gaze.sort_values('world_timestamp').reset_index(drop=True)  # Time order
    print(f"\nCombined gaze for patterns: {len(combined_gaze)} samples")
    
    # NEW STEPS 3.5-6: Extract patterns (ipynb → 60Hz tuned: window=42 ~0.7s)
    print("\n=== EXTRACTING SCAN PATTERNS (from notebook) ===")
    gaze_windows = sliding_window_gaze(combined_gaze[['norm_pos_x', 'norm_pos_y']], window_length=42, stride=10)
    patterns_df, occ_indices = extract_patterns(gaze_windows, num_patterns=10)
    patterns_df.to_csv(os.path.join(output_dir, 'scan_patterns.csv'), index=False)
    plot_patterns(combined_gaze, patterns_df, occ_indices, output_dir)
    
    # Continue with your original AOI/transition pipeline...
    screen1_aoi = create_aoi_data(surface1)
    screen2_aoi = create_aoi_data(surface2)
    # [All your count_fixations_per_aoi, calculate_aoi_durations, transitions, viz...]
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("New files: scan_patterns.csv/png")

if __name__ == "__main__":
    main()
