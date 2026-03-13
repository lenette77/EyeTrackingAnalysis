"""
Eye-Tracking Analysis: Multi-Surface AOIs + Continuous Scan Patterns (ipynb Integrated)
Combines fixation AOI/transitions + trajectory patterns.
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
def load_data(surface_path):
    """Load a CSV file containing fixation data for one surface."""
    try:
        data = pd.read_csv(surface_path)
        print("Data loaded successfully")
        print(f"Surface shape: {data.shape}")
        print(f"Surface columns: {list(data.columns)}")
        return data
    except FileNotFoundError as e:
        print(f"Error: {e}")
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


def analyze_surface_coverage(surface_map, all_fixations=None):
    """Analyze which fixations landed on which surface (multi-surface)."""
    surface_sets = {}
    for label, df in surface_map.items():
        if df is None or 'on_surf' not in df.columns or 'fixationid' not in df.columns:
            surface_sets[label] = set()
            continue
        surface_sets[label] = set(df[df['on_surf'] == True]['fixationid'].unique())

    if all_fixations is not None and 'fixationid' in all_fixations.columns:
        all_fixation_ids = set(all_fixations['fixationid'].unique())
    else:
        combined = set()
        for s in surface_sets.values():
            combined |= s
        all_fixation_ids = combined
        print("Warning: all_fixations not provided; 'neither surface' is based only on surface files.")

    any_surface = set()
    for s in surface_sets.values():
        any_surface |= s

    surface_counts = {label: len(s) for label, s in surface_sets.items()}
    neither = all_fixation_ids - any_surface

    overlap_counts = {}
    for fix_id in any_surface:
        count = sum(1 for s in surface_sets.values() if fix_id in s)
        overlap_counts[fix_id] = count
    multi_surface = {fid for fid, count in overlap_counts.items() if count > 1}

    print("="*60)
    print("Fixation Surface Coverage Analysis")
    print("="*60)
    for label in surface_map.keys():
        print(f"{label}: {surface_counts.get(label, 0)} fixations")
    print(f"On any surface: {len(any_surface)} fixations")
    print(f"Multi-surface: {len(multi_surface)} fixations")
    print(f"Neither surface: {len(neither)} fixations")
    print(f"Total unique fixations: {len(all_fixation_ids)}")

    return {
        'surface_sets': surface_sets,
        'any_surface': any_surface,
        'multi_surface': multi_surface,
        'neither': neither,
        'all_fixations': all_fixation_ids
    }

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


def create_aoi_data_for_surface(surface_data, surface_label):
    """Create AOI assignments and tag rows with surface label (surface-aware AOIs)."""
    labeled = surface_data.copy()
    labeled['surface'] = surface_label
    df = create_aoi_data(labeled)
    df['surface'] = surface_label
    df['aoi_id'] = df['surface'] + '|' + df['aoi'].astype(str)
    return df


def count_fixations_per_aoi(surface_data, output_dir="output", output_filename="aoi_fixation_counts.csv"):
    """Count how many unique fixations landed in each AOI."""
    aoi_counts = surface_data.groupby('aoi')['fixationid'].nunique()
    aoi_counts_df = aoi_counts.reset_index()
    aoi_counts_df.columns = ['AOI', 'Fixation_Count']
    aoi_counts_df = aoi_counts_df.sort_values('Fixation_Count', ascending=False)

    print("="*60)
    print("STEP 4: Fixation Count Per AOI")
    print("="*60)
    print(aoi_counts_df)

    os.makedirs(output_dir, exist_ok=True)
    aoi_counts_df.to_csv(os.path.join(output_dir, output_filename), index=False)
    print(f"Saved to: {output_dir}/{output_filename}")

    return aoi_counts_df


def calculate_aoi_durations(surface_data, output_dir="output", output_filename="aoi_durations.csv"):
    """Sum up how long people looked at each AOI."""
    aoi_durations = surface_data.groupby('aoi')['duration'].sum()
    aoi_durations_df = aoi_durations.reset_index()
    aoi_durations_df.columns = ['AOI', 'Total_Duration_ms']
    aoi_durations_df = aoi_durations_df.sort_values('Total_Duration_ms', ascending=False)

    print("="*60)
    print("STEP 5: Total Duration Per AOI")
    print("="*60)
    print(aoi_durations_df)

    os.makedirs(output_dir, exist_ok=True)
    aoi_durations_df.to_csv(os.path.join(output_dir, output_filename), index=False)
    print(f"Saved to: {output_dir}/{output_filename}")

    return aoi_durations_df


def create_transition_sequence(surface_data, output_dir="output"):
    """Track how gaze moves from one AOI to another over time."""
    data = surface_data.copy()
    group_cols = ['fixationid']
    if 'surface' in data.columns:
        group_cols = ['fixationid', 'surface']

    if data.duplicated(subset=group_cols).any():
        data = collapse_fixations(data, group_cols=group_cols)
        data['aoi'] = data.apply(
            lambda row: assign_aoi(row['norm_pos_x'], row['norm_pos_y']),
            axis=1
        )
        if 'surface' in data.columns:
            data['aoi_id'] = data['surface'] + '|' + data['aoi'].astype(str)

    fixation_sequence = data.sort_values('world_timestamp').reset_index(drop=True)
    aoi_col = 'aoi_id' if 'aoi_id' in fixation_sequence.columns else 'aoi'

    os.makedirs(output_dir, exist_ok=True)
    seq_cols = ['fixationid', aoi_col, 'world_timestamp']
    seq_cols = [c for c in seq_cols if c in fixation_sequence.columns]
    fixation_sequence[seq_cols].to_csv(
        os.path.join(output_dir, 'full_fixation_sequence.csv'),
        index=False
    )

    fixation_sequence['next_fixation_id'] = fixation_sequence['fixationid'].shift(-1)
    fixation_sequence['next_world_timestamp'] = fixation_sequence['world_timestamp'].shift(-1)
    fixation_sequence['next_aoi'] = fixation_sequence[aoi_col].shift(-1)

    transitions = fixation_sequence[fixation_sequence['next_aoi'].notna()].copy()
    transitions['transition'] = transitions[aoi_col].astype(str) + ' -> ' + transitions['next_aoi'].astype(str)
    transitions['transition_duration'] = transitions['next_world_timestamp'] - transitions['world_timestamp']

    if aoi_col == 'aoi_id':
        transitions['from_surface'] = transitions[aoi_col].apply(
            lambda s: s.split('|')[0] if isinstance(s, str) and '|' in s else ''
        )
        transitions['to_surface'] = transitions['next_aoi'].apply(
            lambda s: s.split('|')[0] if isinstance(s, str) and '|' in s else ''
        )
    else:
        transitions['from_surface'] = ''
        transitions['to_surface'] = ''

    print("="*60)
    print("STEP 6: Transition Sequence")
    print("="*60)
    display_cols = [aoi_col, 'next_aoi', 'transition', 'world_timestamp', 'next_world_timestamp', 'transition_duration']
    available = [c for c in display_cols if c in transitions.columns]
    print(transitions[available].head(20))

    out_cols = ['fixationid', aoi_col, 'world_timestamp', 'next_fixation_id', 'next_aoi',
                'next_world_timestamp', 'transition_duration', 'transition']
    out_cols = [c for c in out_cols if c in transitions.columns]
    transitions[out_cols].to_csv(os.path.join(output_dir, 'transition_sequence.csv'), index=False)
    print(f"Saved to: {output_dir}/transition_sequence.csv")
    print(f"Saved to: {output_dir}/full_fixation_sequence.csv")

    return transitions


def _get_surface_order_from_transitions(transitions, surface_order=None):
    if surface_order:
        return surface_order
    if 'from_surface' in transitions.columns:
        surfaces = list(pd.unique(transitions['from_surface'].dropna()))
    else:
        surfaces = []
    return [s for s in surfaces if s]


def create_transition_matrix(transitions, output_dir="output", surface_order=None):
    """Count how many times each AOI-to-AOI transition happens."""
    from_col = 'aoi_id' if 'aoi_id' in transitions.columns else 'aoi'
    transition_counts = transitions.groupby([from_col, 'next_aoi']).size().reset_index(name='count')
    matrix = transition_counts.pivot(index=from_col, columns='next_aoi', values='count')
    matrix = matrix.fillna(0)

    if from_col == 'aoi_id':
        base_aois = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        surface_order = _get_surface_order_from_transitions(transitions, surface_order)
        all_aois = []
        for s in surface_order:
            for b in base_aois:
                all_aois.append(f"{s}|{b}")
    else:
        all_aois = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    for aoi in all_aois:
        if aoi not in matrix.index:
            matrix.loc[aoi] = 0
        if aoi not in matrix.columns:
            matrix[aoi] = 0

    matrix = matrix.loc[all_aois, all_aois]

    print("="*60)
    print("STEP 7: Transition Matrix")
    print("="*60)
    print(matrix)

    os.makedirs(output_dir, exist_ok=True)
    matrix.to_csv(os.path.join(output_dir, 'transition_matrix.csv'))
    print(f"Saved to: {output_dir}/transition_matrix.csv")

    return matrix


def analyze_cross_screen_transitions(transitions, output_dir="output"):
    """Extract transitions that cross surfaces, compute counts and durations, and save details."""
    if 'from_surface' not in transitions.columns or 'to_surface' not in transitions.columns:
        print("No surface information in transitions; cannot compute cross-surface transitions.")
        return None

    cross = transitions[transitions['from_surface'] != transitions['to_surface']].copy()
    total_cross = len(cross)
    if total_cross > 0 and 'transition_duration' in cross.columns:
        avg_duration = cross['transition_duration'].mean()
        median_duration = cross['transition_duration'].median()
    else:
        avg_duration = None
        median_duration = None

    os.makedirs(output_dir, exist_ok=True)
    cols = ['fixationid', 'aoi_id' if 'aoi_id' in cross.columns else 'aoi', 'world_timestamp',
            'next_fixation_id', 'next_aoi', 'next_world_timestamp', 'transition_duration', 'from_surface', 'to_surface']
    cols = [c for c in cols if c in cross.columns]
    cross.to_csv(os.path.join(output_dir, 'cross_screen_transitions.csv'), index=False)

    summary = {
        'total_cross_transitions': int(total_cross),
        'avg_transition_duration': float(avg_duration) if avg_duration is not None else None,
        'median_transition_duration': float(median_duration) if median_duration is not None else None
    }
    print("Cross-surface transitions summary:")
    print(summary)

    return cross, summary


def save_cross_screen_summary_and_visuals(cross, summary, output_dir="output"):
    """Save cross-surface summary (JSON/CSV) and create visualizations."""
    import json

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'cross_screen_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(output_dir, 'cross_screen_summary.csv'), index=False)

    if cross is None or cross.empty:
        print("No cross-surface transitions to visualize.")
        return

    if 'transition_duration' not in cross.columns:
        print('No transition_duration available; skipping duration plots')

    pair_counts = cross.groupby(['from_surface', 'to_surface']).size().reset_index(name='count')
    pair_counts['pair'] = pair_counts['from_surface'] + ' -> ' + pair_counts['to_surface']
    plt.figure(figsize=(8, 4))
    sns.barplot(data=pair_counts, x='pair', y='count', palette='muted')
    plt.title('Cross-Surface Transition Counts')
    plt.xlabel('Transition Pair')
    plt.ylabel('Count')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_transition_counts.png'), dpi=300)
    plt.close()

    if 'transition_duration' in cross.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(cross['transition_duration'].dropna(), bins=20, kde=False, color='steelblue')
        plt.title('Cross-Surface Transition Duration')
        plt.xlabel('Duration (same units as world_timestamp)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cross_transition_duration_hist.png'), dpi=300)
        plt.close()

    if 'world_timestamp' in cross.columns:
        plt.figure(figsize=(10, 3))
        cross = cross.copy()
        cross['pair'] = cross['from_surface'] + ' -> ' + cross['to_surface']
        sns.scatterplot(data=cross, x='world_timestamp', y='transition_duration', hue='pair', s=50)
        plt.title('Cross-Surface Transitions Timeline')
        plt.xlabel('World Timestamp')
        plt.ylabel('Transition Duration')
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cross_transition_timeline.png'), dpi=300)
        plt.close()

    print(f"Saved cross-surface summary and visualizations to: {output_dir}")


def visualize_fixation_heatmap(aoi_counts_df, output_dir="output", output_filename="fixation_heatmap.png"):
    """Create a visual grid showing where people looked most."""
    grid = np.zeros((3, 3))
    aoi_to_position = {
        1: (0, 0), 2: (0, 1), 3: (0, 2),
        4: (1, 0), 5: (1, 1), 6: (1, 2),
        7: (2, 0), 8: (2, 1), 9: (2, 2)
    }

    for _, row in aoi_counts_df.iterrows():
        aoi_value = row['AOI']
        count = row['Fixation_Count']
        try:
            aoi_value = int(aoi_value)
        except (TypeError, ValueError):
            continue
        if aoi_value in aoi_to_position:
            pos = aoi_to_position[aoi_value]
            grid[pos[0], pos[1]] = count

    plt.figure(figsize=(8, 6))
    sns.heatmap(grid, annot=True, fmt='.0f', cmap='YlOrRd',
                xticklabels=['1/4/7', '2/5/8', '3/6/9'],
                yticklabels=['1-3', '4-6', '7-9'])
    plt.title('Fixation Count Heatmap')
    plt.xlabel('AOI Column (by id)')
    plt.ylabel('AOI Row (by id)')

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, output_filename), dpi=300, bbox_inches='tight')
    print(f"Saved to: {output_dir}/{output_filename}")
    plt.close()


def visualize_transition_heatmap(matrix, output_dir="output", annotate=False):
    """Create a heatmap of the transition matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=annotate, fmt='.0f', cmap='Blues')
    plt.title('AOI Transition Matrix')
    plt.xlabel('To AOI')
    plt.ylabel('From AOI')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'transition_matrix_heatmap.png'), dpi=300)
    print(f"Saved to: {output_dir}/transition_matrix_heatmap.png")
    plt.close()


def visualize_fixation_density(surface_data, output_dir="output", output_filename="fixation_density.png"):
    """Create a 2D density chart of fixation locations on a surface."""
    data = surface_data[surface_data['on_surf'] == True].copy()
    if data.empty:
        print("No on-surface fixations found; skipping density chart.")
        return

    data = collapse_fixations(data)

    plt.figure(figsize=(6, 5), facecolor='black')
    sns.kdeplot(
        data=data,
        x='norm_pos_x',
        y='norm_pos_y',
        fill=True,
        levels=50,
        cmap='mako',
        bw_adjust=0.8
    )
    ax = plt.gca()
    ax.set_facecolor('black')
    ax.set_title('Fixation Density', color='white')
    ax.set_xlabel('Normalized X', color='white')
    ax.set_ylabel('Normalized Y', color='white')
    ax.tick_params(colors='white')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, output_filename), dpi=300)
    print(f"Saved to: {output_dir}/{output_filename}")
    plt.close()


def visualize_transition_path(sequence_data, surface_order, output_dir="output", output_filename="transition_path.png"):
    """Visualize the full fixation sequence across surfaces as a traced path."""
    data = sequence_data.copy()
    if 'surface' not in data.columns or 'norm_pos_x' not in data.columns or 'norm_pos_y' not in data.columns:
        print("Missing surface or normalized positions for transition path; skipping.")
        return

    group_cols = ['fixationid', 'surface'] if 'surface' in data.columns else ['fixationid']
    if data.duplicated(subset=group_cols).any():
        data = collapse_fixations(data, group_cols=group_cols)

    data = data.sort_values('world_timestamp').reset_index(drop=True)

    grid_size = 3
    gap = 1
    surface_set = set(surface_order)
    if {'Left', 'Mid', 'Right', 'Dashboard'}.issubset(surface_set):
        offsets = {
            'Left': (0, 0),
            'Mid': (grid_size + gap, 0),
            'Right': (2 * (grid_size + gap), 0),
            'Dashboard': (grid_size + gap, grid_size + gap)
        }
    else:
        offsets = {label: (i * (grid_size + gap), 0) for i, label in enumerate(surface_order)}

    xs = []
    ys = []
    labels = []
    for _, row in data.iterrows():
        surface = row['surface']
        if surface not in offsets:
            continue
        try:
            x_norm = float(row['norm_pos_x'])
            y_norm = float(row['norm_pos_y'])
        except (TypeError, ValueError):
            continue
        if not (0 <= x_norm <= 1 and 0 <= y_norm <= 1):
            continue
        x_off, y_off = offsets[surface]
        x = x_off + (x_norm * grid_size)
        y = y_off + (y_norm * grid_size)
        xs.append(x)
        ys.append(y)
        labels.append(row['fixationid'])

    if len(xs) < 2:
        print("Not enough fixations for transition path; skipping.")
        return

    plt.figure(figsize=(12, 4))
    ax = plt.gca()
    ax.set_aspect('equal')

    for label, (x_off, y_off) in offsets.items():
        for i in range(grid_size + 1):
            ax.plot([x_off, x_off + grid_size], [y_off + i, y_off + i], color='gray', linewidth=0.6)
            ax.plot([x_off + i, x_off + i], [y_off, y_off + grid_size], color='gray', linewidth=0.6)
        ax.text(x_off + 1.5, y_off - 0.4, label, ha='center', va='top')

    ax.plot(xs, ys, color='tab:orange', linewidth=0.8, alpha=0.35, linestyle='--')
    ax.scatter(xs, ys, color='tab:orange', s=18, alpha=0.9, zorder=3)
    for x, y, label in zip(xs, ys, labels):
        ax.text(x, y, str(label), fontsize=6, ha='center', va='center', color='black', zorder=4)
    ax.scatter(xs[0], ys[0], color='green', s=30, label='Start')
    ax.scatter(xs[-1], ys[-1], color='red', s=30, label='End')

    max_x = max(pos[0] for pos in offsets.values()) + grid_size + 0.2
    max_y = max(pos[1] for pos in offsets.values()) + grid_size + 0.2
    ax.set_xlim(-0.2, max_x)
    ax.set_ylim(-0.8, max_y)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Full Fixation Transition Path')
    ax.legend(loc='upper right', frameon=False)

    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_filename), dpi=300)
    print(f"Saved to: {output_dir}/{output_filename}")
    plt.close()

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
    if x_col not in data.columns and 'x_norm' in data.columns:
        data[x_col] = data['x_norm']
    if y_col not in data.columns and 'y_norm' in data.columns:
        data[y_col] = data['y_norm']
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


def compute_diffwhere(tx, ty, diff=2, quantile=0.85, max_candidates=None):
    """Select candidate indices based on large changes in X/Y."""
    xdiff = tx[diff:] - tx[:-diff]
    ydiff = ty[diff:] - ty[:-diff]
    xwhere = np.where(np.abs(xdiff) > np.quantile(np.abs(xdiff), quantile))[0]
    ywhere = np.where(np.abs(ydiff) > np.quantile(np.abs(ydiff), quantile))[0]
    candidates = np.union1d(xwhere, ywhere)
    if max_candidates is not None and len(candidates) > max_candidates:
        step = max(1, int(np.ceil(len(candidates) / max_candidates)))
        candidates = candidates[::step]
    return candidates


def downsample_gaze(data, factor):
    """Downsample gaze data by a fixed factor to reduce memory use."""
    if factor is None or factor <= 1:
        return data
    return data.iloc[::factor].reset_index(drop=True)


def extract_matrix_profile_patterns(tx, ty, rec_time_s, m=95, k=10, diff=2, q_min=0.001, q_max=0.01,
                                    min_masks=5, max_candidates=None):
    """Notebook-style matrix profile pattern extraction using stumpy.mass."""
    if stumpy is None:
        print("stumpy is not installed; skipping matrix profile pattern extraction.")
        return None

    pad_width = (0, int(m * np.ceil(tx.shape[0] / m) - tx.shape[0]))
    tx_padded = np.pad(tx, pad_width, mode='constant', constant_values=np.nan)
    ty_padded = np.pad(ty, pad_width, mode='constant', constant_values=np.nan)
    n_padded = tx_padded.shape[0]

    diffwhere = compute_diffwhere(tx, ty, diff=diff, max_candidates=max_candidates)
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

    def time_color(t_index, t_max):
        if t_max <= 1:
            return (1.0, 0.0, 0.0)
        ratio = t_index / (t_max - 1)
        return (1.0 - ratio, ratio, 0.0)

    fig, axs = plt.subplots(3, 4, figsize=(12, 9))
    for i in range(k):
        axs.flat[i].set_title(f'Pattern #{i + 1}')
        for t in range(snippets_x.shape[1]):
            alpha = 0.2 + 0.8 * t / max(1, snippets_x.shape[1] - 1) if alpha_by_time else 1.0
            color = time_color(t, snippets_x.shape[1])
            axs.flat[i].plot(snippets_x[i, t], snippets_y[i, t], 'o', color=color, markersize=6, alpha=alpha)
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


def compute_aoi_summary(aoi_df, output_dir, filename):
    """Collate AOI stats: counts and durations."""
    summary = (
        aoi_df.groupby(['surface', 'aoi_id'])
        .agg(
            fixation_count=('fixationid', 'nunique'),
            total_duration_ms=('duration', 'sum'),
            mean_duration_ms=('duration', 'mean')
        )
        .reset_index()
        .sort_values(['surface', 'aoi_id'])
    )
    os.makedirs(output_dir, exist_ok=True)
    summary.to_csv(os.path.join(output_dir, filename), index=False)
    return summary


def compute_transition_summary(transitions, output_dir, filename):
    """Collate transition counts and durations."""
    summary = (
        transitions.groupby(['aoi_id', 'next_aoi'])
        .agg(
            transition_count=('transition', 'count'),
            mean_duration=('transition_duration', 'mean'),
            median_duration=('transition_duration', 'median')
        )
        .reset_index()
        .sort_values('transition_count', ascending=False)
    )
    os.makedirs(output_dir, exist_ok=True)
    summary.to_csv(os.path.join(output_dir, filename), index=False)
    return summary


def _surface_offsets(surface_order, grid_size=3, gap=1):
    surface_set = set(surface_order)
    if {'Left', 'Mid', 'Right', 'Dashboard'}.issubset(surface_set):
        return {
            'Left': (0, 0),
            'Mid': (grid_size + gap, 0),
            'Right': (2 * (grid_size + gap), 0),
            'Dashboard': (grid_size + gap, grid_size + gap)
        }
    return {label: (i * (grid_size + gap), 0) for i, label in enumerate(surface_order)}


def _aoi_center(aoi_id, offsets, grid_size=3):
    if isinstance(aoi_id, str) and '|' in aoi_id:
        surface, aoi = aoi_id.split('|', 1)
    else:
        return None
    try:
        aoi = int(aoi)
    except ValueError:
        return None
    if surface not in offsets:
        return None
    col = (aoi - 1) % 3
    row = (aoi - 1) // 3
    x_off, y_off = offsets[surface]
    return x_off + (col + 0.5), y_off + (row + 0.5)


def visualize_transition_flow_map(transitions, surface_order, output_dir, filename):
    """Flow map with shaded paths (no labels)."""
    if transitions.empty:
        print("No transitions to plot; skipping flow map.")
        return

    offsets = _surface_offsets(surface_order)
    counts = transitions.groupby(['aoi_id', 'next_aoi']).size().reset_index(name='count')
    max_count = max(counts['count']) if len(counts) else 1

    plt.figure(figsize=(12, 4))
    ax = plt.gca()
    ax.set_aspect('equal')

    grid_size = 3
    for label, (x_off, y_off) in offsets.items():
        for i in range(grid_size + 1):
            ax.plot([x_off, x_off + grid_size], [y_off + i, y_off + i], color='lightgray', linewidth=0.6)
            ax.plot([x_off + i, x_off + i], [y_off, y_off + grid_size], color='lightgray', linewidth=0.6)
        ax.text(x_off + 1.5, y_off - 0.4, label, ha='center', va='top')

    for _, row in counts.iterrows():
        start = _aoi_center(row['aoi_id'], offsets, grid_size=grid_size)
        end = _aoi_center(row['next_aoi'], offsets, grid_size=grid_size)
        if start is None or end is None:
            continue
        alpha = 0.1 + 0.8 * (row['count'] / max_count)
        linewidth = 0.5 + 2.5 * (row['count'] / max_count)
        ax.plot([start[0], end[0]], [start[1], end[1]], color='steelblue', alpha=alpha, linewidth=linewidth)

    max_x = max(pos[0] for pos in offsets.values()) + grid_size + 0.2
    max_y = max(pos[1] for pos in offsets.values()) + grid_size + 0.2
    ax.set_xlim(-0.2, max_x)
    ax.set_ylim(-0.8, max_y)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Transition Flow Map')

    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


def visualize_cross_transition_density(transitions, surface_order, output_dir, filename):
    """Density map of cross-surface transition midpoints."""
    if transitions.empty or 'from_surface' not in transitions.columns:
        print("No cross-surface transitions to plot; skipping density map.")
        return

    cross = transitions[transitions['from_surface'] != transitions['to_surface']].copy()
    if cross.empty:
        print("No cross-surface transitions found; skipping density map.")
        return

    offsets = _surface_offsets(surface_order)
    grid_size = 3
    mids_x = []
    mids_y = []
    for _, row in cross.iterrows():
        start = _aoi_center(row['aoi_id'], offsets, grid_size=grid_size)
        end = _aoi_center(row['next_aoi'], offsets, grid_size=grid_size)
        if start is None or end is None:
            continue
        mids_x.append((start[0] + end[0]) / 2.0)
        mids_y.append((start[1] + end[1]) / 2.0)

    if len(mids_x) < 2:
        print("Not enough cross-surface transitions for density map; skipping.")
        return

    plt.figure(figsize=(12, 4))
    ax = plt.gca()
    ax.set_aspect('equal')

    for label, (x_off, y_off) in offsets.items():
        for i in range(grid_size + 1):
            ax.plot([x_off, x_off + grid_size], [y_off + i, y_off + i], color='lightgray', linewidth=0.6)
            ax.plot([x_off + i, x_off + i], [y_off, y_off + grid_size], color='lightgray', linewidth=0.6)
        ax.text(x_off + 1.5, y_off - 0.4, label, ha='center', va='top')

    sns.kdeplot(x=mids_x, y=mids_y, fill=True, cmap='mako', thresh=0.05, levels=20, alpha=0.8)
    max_x = max(pos[0] for pos in offsets.values()) + grid_size + 0.2
    max_y = max(pos[1] for pos in offsets.values()) + grid_size + 0.2
    ax.set_xlim(-0.2, max_x)
    ax.set_ylim(-0.8, max_y)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Cross-Surface Transition Density')

    os.makedirs(output_dir, exist_ok=True)
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

    data_base = os.path.join("C:\\Users\\lenny\\Desktop\\Wisco\\HFML Lab\\2026_03_03")
    run_ids = ["000", "001"]

    surfaces = [
        {"label": "Left", "file": "fixations_on_surface_Left.csv", "id": 1},
        {"label": "Mid", "file": "fixations_on_surface_Mid.csv", "id": 2},
        {"label": "Right", "file": "fixations_on_surface_Right.csv", "id": 3},
        {"label": "Dashboard", "file": "fixations_on_surface_Dashboard.csv", "id": 4},
    ]

    for run_id in run_ids:
        print("\n" + "=" * 60)
        print(f"PROCESSING RUN: {run_id}")
        print("=" * 60)

        data_root = os.path.join(data_base, run_id, "exports", run_id)
        surfaces_dir = os.path.join(data_root, "surfaces")
        output_dir = os.path.join(script_dir, "output", run_id)
        os.makedirs(output_dir, exist_ok=True)

        surface_data_map = {}
        for s in surfaces:
            path = os.path.join(surfaces_dir, s["file"])
            df = load_data(path)
            if df is None:
                return
            surface_data_map[s["label"]] = normalize_fixation_cols(df)

        all_fixations_path = os.path.join(data_root, "fixations.csv")
        all_fixations = None
        if os.path.exists(all_fixations_path):
            all_fixations = normalize_fixation_cols(pd.read_csv(all_fixations_path))
        else:
            print(f"Note: all-fixations file not found at: {all_fixations_path}")

        surface_order = [s["label"] for s in surfaces]

        # STEP 2: Surface coverage
        coverage = analyze_surface_coverage(surface_data_map, all_fixations)

        # STEP 2.5: Continuous gaze patterns (single gaze_positions.csv)
        gaze_path = os.path.join(data_root, "gaze_positions.csv")
        gaze_df = load_gaze_positions(gaze_path)
        if gaze_df is not None:
            print("\n=== EXTRACTING SCAN PATTERNS (matrix profile) ===")
            gaze_df = preprocess_gaze_positions(gaze_df)

            # Downsample to reduce memory while preserving pattern structure
            gaze_df = downsample_gaze(gaze_df, factor=3)

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
                min_masks=5,
                max_candidates=5000
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

        # Continue with AOI/transition pipeline for all surfaces
        aoi_data = []
        for s in surfaces:
            label = s["label"]
            df = surface_data_map[label]
            aoi_df = create_aoi_data_for_surface(df, label)
            aoi_data.append(aoi_df)

            label_slug = label.lower().replace(' ', '_')
            aoi_counts = count_fixations_per_aoi(
                aoi_df,
                output_dir,
                output_filename=f"aoi_fixation_counts_{label_slug}.csv"
            )
            calculate_aoi_durations(
                aoi_df,
                output_dir,
                output_filename=f"aoi_durations_{label_slug}.csv"
            )
            visualize_fixation_heatmap(
                aoi_counts,
                output_dir,
                output_filename=f"fixation_heatmap_{label_slug}.png"
            )
            visualize_fixation_density(
                df,
                output_dir,
                output_filename=f"fixation_density_{label_slug}.png"
            )

        combined = pd.concat(aoi_data).sort_values('world_timestamp')
        transitions = create_transition_sequence(combined, output_dir)
        matrix = create_transition_matrix(transitions, output_dir, surface_order=surface_order)
        visualize_transition_heatmap(matrix, output_dir)
        visualize_transition_path(combined, surface_order, output_dir, output_filename="transition_path.png")
        visualize_transition_flow_map(transitions, surface_order, output_dir, filename="transition_flow_map.png")

        cross_details = analyze_cross_screen_transitions(transitions, output_dir)
        if cross_details is not None:
            cross_df, cross_summary = cross_details
            save_cross_screen_summary_and_visuals(cross_df, cross_summary, output_dir)
            visualize_cross_transition_density(transitions, surface_order, output_dir,
                                               filename="cross_transition_density.png")

        compute_aoi_summary(combined, output_dir, filename="aoi_summary.csv")
        compute_transition_summary(transitions, output_dir, filename="transition_summary.csv")
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("New files: scan_patterns.csv/png (if gaze_positions.csv is available)")

if __name__ == "__main__":
    main()
