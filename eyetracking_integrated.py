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

try:
    import stumpy
except ImportError:
    stumpy = None

try:
    from tslearn.metrics import dtw_path
except ImportError:
    dtw_path = None

try:
    import matplotlib.animation as animation
    import matplotlib.cm as cm
except ImportError:
    animation = None
    cm = None
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
def load_gaze_positions(path):
    """Load continuous gaze positions (e.g., gaze_positions.csv)."""
    try:
        df = pd.read_csv(path)
        print(f"Loaded gaze positions: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Gaze file not found: {path}")
        return None


def preprocess_gaze_positions(df, time_col='gaze_timestamp', x_col='norm_pos_x', y_col='norm_pos_y',
                              confidence_col='confidence', confidence_min=0.6):
    """Basic cleanup + time alignment for continuous gaze data."""
    data = df.copy()
    for col in [time_col, x_col, y_col]:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    if confidence_col in data.columns:
        data.loc[data[confidence_col] < confidence_min, [x_col, y_col]] = np.nan

    data[[x_col, y_col]] = data[[x_col, y_col]].apply(pd.to_numeric, errors='coerce')
    data[[x_col, y_col]] = data[[x_col, y_col]].interpolate(limit_direction='both')

    data[time_col] = pd.to_numeric(data[time_col], errors='coerce')
    data = data.dropna(subset=[time_col]).reset_index(drop=True)
    data['rec_time_s'] = data[time_col] - data[time_col].min()

    return data


def compute_diffwhere(tx, ty, diff=2, quantile=0.85):
    """Select candidate indices based on large changes in X/Y."""
    xdiff = tx[diff:] - tx[:-diff]
    ydiff = ty[diff:] - ty[:-diff]
    xwhere = np.where(np.abs(xdiff) > np.quantile(np.abs(xdiff), quantile))[0]
    ywhere = np.where(np.abs(ydiff) > np.quantile(np.abs(ydiff), quantile))[0]
    return np.union1d(xwhere, ywhere)


def extract_matrix_profile_patterns(tx, ty, rec_time_s, m=95, k=10, diff=2, q_min=0.001, q_max=0.01,
                                    min_masks=5):
    """Notebook-style matrix profile pattern extraction using stumpy.mass."""
    if stumpy is None:
        print("stumpy is not installed; skipping matrix profile pattern extraction.")
        return None

    pad_width = (0, int(m * np.ceil(tx.shape[0] / m) - tx.shape[0]))
    tx_padded = np.pad(tx, pad_width, mode='constant', constant_values=np.nan)
    ty_padded = np.pad(ty, pad_width, mode='constant', constant_values=np.nan)
    n_padded = tx_padded.shape[0]

    diffwhere = compute_diffwhere(tx, ty, diff=diff)
    diffwhere = diffwhere[diffwhere < n_padded - m + 1]
    if len(diffwhere) == 0:
        print("No candidate indices found for matrix profile.")
        return None

    dx = np.empty((len(diffwhere), tx.shape[0] - m + 1), dtype=np.float64)
    dy = np.empty((len(diffwhere), ty.shape[0] - m + 1), dtype=np.float64)

    for i, start in enumerate(diffwhere):
        stop = start + m
        sx = tx_padded[start:stop]
        sy = ty_padded[start:stop]
        dx[i, :] = stumpy.mass(sx, tx, normalize=False, p=2.0)
        dy[i, :] = stumpy.mass(sy, ty, normalize=False, p=2.0)

    d = np.sqrt(dx ** 2 + dy ** 2)
    d_plot = d.copy()

    snippets_x = np.empty((k, m), dtype=np.float64)
    snippets_y = np.empty((k, m), dtype=np.float64)
    snippets_indices = np.empty(k, dtype=np.int64)
    snippets_profiles = np.empty((k, dx.shape[-1]), dtype=np.float64)
    snippets_areas = np.empty(k, dtype=np.float64)

    indices = diffwhere
    mask = np.full(tx.shape, -np.inf)
    tx_process = tx.copy()
    ty_process = ty.copy()
    mask_list = []
    patterns = []

    count = 0
    while count < k:
        q_threshold = (q_max - q_min) / (k - 1) * count + q_min
        profile_areas = np.nansum(d_plot, axis=1)
        valid = profile_areas[profile_areas < np.inf]
        if len(valid) == 0:
            break

        idx = np.where(profile_areas == max(valid))[0][0]

        mask_num = 1
        prev_maskidx = None
        for maskidx in np.where(d_plot[idx] <= np.nanquantile(d_plot[idx], q_threshold))[0]:
            if prev_maskidx is None:
                prev_maskidx = maskidx
            elif maskidx - prev_maskidx > 1:
                mask_num += 1
                prev_maskidx = maskidx
            else:
                prev_maskidx = maskidx

        if mask_num < min_masks:
            d_plot[np.array([np.max(mask[index:index + m]) >= 0 for index in indices]), :] = np.nan
            d_plot[:, np.isnan(tx_process[:-m + 1])] = np.nan
            d_plot[np.where(np.abs(indices - indices[idx]) < int(m / 2))[0], :] = np.nan
            continue

        snippets_x[count] = tx[indices[idx]: indices[idx] + m]
        snippets_y[count] = ty[indices[idx]: indices[idx] + m]
        snippets_indices[count] = indices[idx]
        snippets_profiles[count] = d[idx]
        snippets_areas[count] = np.sum(d[idx])
        mask[indices[idx]: indices[idx] + m] = count

        for maskidx in np.where(d_plot[idx] <= np.nanquantile(d_plot[idx], 0.010))[0]:
            mask[(maskidx):(maskidx + m)] = count
            tx_process[(maskidx - int(m / 2)):(maskidx + m)] = np.nan
            ty_process[(maskidx - int(m / 2)):(maskidx + m)] = np.nan

        mask_list.append(np.append(np.where(d_plot[idx] <= np.nanquantile(d_plot[idx], 0.010))[0], indices[idx]))

        patterns.append({
            'pattern_id': count + 1,
            'proto_index': int(indices[idx]),
            'n_occurrences': int(np.sum(mask == count)),
            'area': float(snippets_areas[count])
        })

        d_plot[np.array([np.max(mask[index:index + m]) >= 0 for index in indices]), :] = np.nan
        d_plot[:, np.isnan(tx_process[:-m + 1])] = np.nan
        d_plot[np.where(np.abs(indices - indices[idx]) < int(m / 2))[0], :] = np.nan

        count += 1

    patterns_df = pd.DataFrame(patterns)
    return {
        'patterns_df': patterns_df,
        'snippets_x': snippets_x[:count],
        'snippets_y': snippets_y[:count],
        'snippets_indices': snippets_indices[:count],
        'snippets_profiles': snippets_profiles[:count],
        'snippets_areas': snippets_areas[:count],
        'mask': mask,
        'mask_list': mask_list,
        'rec_time_s': rec_time_s
    }


def plot_pattern_grid(snippets_x, snippets_y, output_dir, filename, alpha_by_time=False):
    """Scatter plot of extracted patterns."""
    os.makedirs(output_dir, exist_ok=True)
    k = len(snippets_x)

    fig, axs = plt.subplots(3, 4, figsize=(12, 9))
    for i in range(k):
        axs.flat[i].set_title(f'Pattern #{i + 1}')
        for t in range(snippets_x.shape[1]):
            alpha = 0.2 + 0.8 * t / max(1, snippets_x.shape[1] - 1) if alpha_by_time else 1.0
            axs.flat[i].plot(snippets_x[i, t], snippets_y[i, t], 'o', color='black', markersize=6, alpha=alpha)
        axs.flat[i].set_xlim([0 - 0.05, 1 + 0.05])
        axs.flat[i].set_ylim([0 - 0.05, 1 + 0.05])

    for j in range(k, 12):
        axs.flat[j].set_facecolor('white')
        axs.flat[j].set_xticks([])
        axs.flat[j].set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


def plot_pattern_time_series(tx, ty, rec_time_s, mask, output_dir, filename):
    """Time-series view with pattern overlays."""
    os.makedirs(output_dir, exist_ok=True)
    k = int(np.nanmax(mask) + 1) if np.any(mask >= 0) else 0
    if k == 0:
        print("No patterns to plot in time series.")
        return

    sns.set(rc={'figure.figsize': (16, 6)})
    fig, axs = plt.subplots(2)
    for i in range(k):
        for t in rec_time_s[np.where(mask == i)[0]]:
            axs[0].axvline(t, color=sns.color_palette("tab20")[i], ls='-', lw=1, alpha=0.1)
            axs[1].axvline(t, color=sns.color_palette("tab20")[i], ls='-', lw=1, alpha=0.1)

    axs[0].plot(rec_time_s, tx, color='black')
    axs[1].plot(rec_time_s, ty, color='black')
    axs[0].set_title('X Coordinate')
    axs[1].set_title('Y Coordinate')
    for ax in axs:
        ax.set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


def compute_dtw_averages(tx, ty, rec_time_s, mask, gap_threshold=0.05):
    """Average each pattern's segments with DTW alignment."""
    if dtw_path is None:
        print("tslearn is not installed; skipping DTW averaging.")
        return [], []

    def align_series(ref_series, series):
        try:
            path, _ = dtw_path(ref_series, series, itakura_max_slope=1)
        except RuntimeWarning:
            # Fallback when constraint is infeasible for short/unequal series.
            path, _ = dtw_path(ref_series, series)
        return series[np.array(path)[:, 1]]

    k = int(np.nanmax(mask) + 1) if np.any(mask >= 0) else 0
    snippet_xavg = []
    snippet_yavg = []

    for i in range(k):
        t = np.array(rec_time_s[np.where(mask == i)[0]])
        if t.size == 0:
            snippet_xavg.append(np.array([]))
            snippet_yavg.append(np.array([]))
            continue

        gap_locs = np.where(t[1:] - t[:-1] > gap_threshold)[0]
        t_segments = np.split(t, gap_locs + 1)
        tx_segments = np.split(tx[np.where(mask == i)[0]], gap_locs + 1)
        ty_segments = np.split(ty[np.where(mask == i)[0]], gap_locs + 1)

        ref_series = max(tx_segments, key=len)
        aligned = []
        for s in tx_segments:
            aligned.append(align_series(ref_series, s))
        if aligned:
            min_len = min(len(a) for a in aligned)
            aligned_trim = [a[:min_len] for a in aligned]
            snippet_xavg.append(np.mean(aligned_trim, axis=0))
        else:
            snippet_xavg.append(np.array([]))

        ref_series = max(ty_segments, key=len)
        aligned = []
        for s in ty_segments:
            aligned.append(align_series(ref_series, s))
        if aligned:
            min_len = min(len(a) for a in aligned)
            aligned_trim = [a[:min_len] for a in aligned]
            snippet_yavg.append(np.mean(aligned_trim, axis=0))
        else:
            snippet_yavg.append(np.array([]))

    return snippet_xavg, snippet_yavg

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
    
    # NEW STEP 2.5: Continuous gaze patterns (from notebook)
    gaze_path = os.path.join(week_dir, "example_data", "Mateo_data", "exports", "000", "gaze_positions.csv")
    gaze_df = load_gaze_positions(gaze_path)
    if gaze_df is not None:
        print("\n=== EXTRACTING SCAN PATTERNS (matrix profile) ===")
        gaze_df = preprocess_gaze_positions(gaze_df)

        tx = np.array(gaze_df['norm_pos_x'])
        ty = np.array(gaze_df['norm_pos_y'])
        rec_time_s = np.array(gaze_df['rec_time_s'])

        pattern_results = extract_matrix_profile_patterns(
            tx,
            ty,
            rec_time_s,
            m=95,
            k=10,
            diff=2,
            q_min=0.001,
            q_max=0.01,
            min_masks=5
        )

        if pattern_results is not None:
            patterns_df = pattern_results['patterns_df']
            patterns_df.to_csv(os.path.join(output_dir, 'scan_patterns.csv'), index=False)

            plot_pattern_grid(pattern_results['snippets_x'], pattern_results['snippets_y'], output_dir,
                              filename='scan_patterns.png', alpha_by_time=False)
            plot_pattern_grid(pattern_results['snippets_x'], pattern_results['snippets_y'], output_dir,
                              filename='scan_patterns_fade.png', alpha_by_time=True)
            plot_pattern_time_series(tx, ty, rec_time_s, pattern_results['mask'], output_dir,
                                     filename='scan_patterns_time_series.png')

            snippet_xavg, snippet_yavg = compute_dtw_averages(tx, ty, rec_time_s, pattern_results['mask'])
            if snippet_xavg and animation is not None:
                np.save(os.path.join(output_dir, 'scan_patterns_dtw_xavg.npy'), np.array(snippet_xavg, dtype=object))
                np.save(os.path.join(output_dir, 'scan_patterns_dtw_yavg.npy'), np.array(snippet_yavg, dtype=object))
        else:
            print("Pattern extraction skipped.")
    else:
        print("Skipping scan pattern extraction (gaze_positions.csv not found).")
    
    # Continue with your original AOI/transition pipeline...
    screen1_aoi = create_aoi_data(surface1)
    screen2_aoi = create_aoi_data(surface2)
    # [All your count_fixations_per_aoi, calculate_aoi_durations, transitions, viz...]
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("New files: scan_patterns.csv/png (if gaze_positions.csv is available)")

if __name__ == "__main__":
    main()
