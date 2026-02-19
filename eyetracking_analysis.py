"""
Eye-Tracking Analysis Project
Analyzing fixation data from two screens to understand gaze patterns and transitions

Objectives:
1. Calculate number of fixations on each screen (screen 1 only, screen 2 only, both screens, neither screen)
2. Calculate total duration of fixations
3. Create a 3x3 grid (AOI) and assign each fixation to an AOI based on its normalized x/y coordinates
4. Count how many fixations landed in each AOI
5. Sum up how long each AOI was looked at
6. Determine the path of gaze transitions between the AOIs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_data(surface1_path, surface2_path):
    """
    Load CSV files containing fixation data from two surfaces
    
    Args:
        surface1_path: Path to Surface 1 CSV file
        surface2_path: Path to Surface 2 CSV file
    
    Returns:
        surface1, surface2: Loaded dataframes
    """
    try:
        surface1 = pd.read_csv(surface1_path)
        surface2 = pd.read_csv(surface2_path)
        
        print("Data loaded successfully")
        print(f"\nSurface 1 shape: {surface1.shape}")
        print(f"Surface 1 columns: {list(surface1.columns)}")
        print(f"\nSurface 2 shape: {surface2.shape}")
        print(f"Surface 2 columns: {list(surface2.columns)}")
        
        return surface1, surface2
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Files not found.")
        return None, None


# Counting fixations on each screen and overlaps

def analyze_screen_coverage(surface1, surface2, all_fixations=None):
    """
    Analyze which fixations landed on which screen
    
    Args: Dataframe for Surface 1, Dataframe for Surface 2
    
    Returns dictionary with counts of fixations in each category
    """
    # Unique fixations on each surface
    fixations_on_s1 = surface1[surface1['on_surf'] == True]['fixation_id'].unique()
    fixations_on_s2 = surface2[surface2['on_surf'] == True]['fixation_id'].unique()
    
    # All fixations including ones that don't land on any surfaces
    if all_fixations is not None and 'fixation_id' in all_fixations.columns:
        all_fixations = all_fixations['fixation_id'].unique()
    else:
        all_fixations = pd.concat([surface1['fixation_id'], 
                                   surface2['fixation_id']]).unique()
        print("Warning: 'all_fixations' not provided; 'neither screen' is based only on surface files.")
    
    # Find unique and overlapping fixations
    only_s1 = set(fixations_on_s1) - set(fixations_on_s2)
    only_s2 = set(fixations_on_s2) - set(fixations_on_s1)
    both_screens = set(fixations_on_s1) & set(fixations_on_s2)
    neither = set(all_fixations) - set(fixations_on_s1) - set(fixations_on_s2)
    
    print("\n" + "="*60)
    print("Fixation Screen Coverage Analysis")
    print("="*60)
    print(f"Screen 1 only: {len(only_s1)} fixations")   
    print(f"Screen 2 only: {len(only_s2)} fixations")
    print(f"Both screens: {len(both_screens)} fixations")
    print(f"Neither screen: {len(neither)} fixations")
    print(f"Total unique fixations: {len(all_fixations)}")
    
    return {
        'only_s1': only_s1,
        'only_s2': only_s2,
        'both_screens': both_screens,
        'neither': neither,
        'all_fixations': all_fixations
    }


# Creating AOI assignments based on normalized x/y coordinates

def assign_aoi(x, y):
    """
    Takes x, y coordinates and returns which AOI it belongs to
    
    Args:
        x: horizontal position (0-1)
        y: vertical position (0-1)
    
    Returns:
        aoi_id: integer 1-9 (row-major: top-left=1 ... bottom-right=9)
    """
    # columns (0=left, 1=center, 2=right)
    if x < 0.333:
        col = 0
    elif x < 0.667:
        col = 1
    else:
        col = 2
    
    # rows (0=top, 1=middle, 2=bottom)
    if y < 0.333:
        row = 0
    elif y < 0.667:
        row = 1
    else:
        row = 2
    
    # Numeric grid positions (row-major)
    ids = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    
    return ids[row][col]


def collapse_fixations(surface_data, group_cols=None):
    """
    Collapse multiple rows per fixation into one row using sensible aggregates.
    """
    if group_cols is None:
        group_cols = ['fixation_id']
        if 'surface' in surface_data.columns:
            group_cols = ['fixation_id', 'surface']

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
    """
    Assign AOI to each fixation on the surface
    
    Args:
        surface_data: Dataframe with fixation data
    
    Returns:
        Dataframe with AOI assignments
    """
    # Only consider the fixations that landed on the surface
    data = surface_data[surface_data['on_surf'] == True].copy()
    # Collapse multiple rows per fixation to a single fixation-level row
    data = collapse_fixations(data)
    
    # Assign aoi to each fixation using the normalized x/y coordinates
    data['aoi'] = data.apply(
        lambda row: assign_aoi(row['norm_pos_x'], row['norm_pos_y']), 
        axis=1
    )
    
    print("\n" + "="*60)
    print("STEP 3: AOI Assignment (3x3 Grid)")
    print("="*60)
    print(f"Assigned {len(data)} fixations to AOIs")
    print("\nSample AOI assignments:")
    print(data[['fixation_id', 'norm_pos_x', 'norm_pos_y', 'aoi']].head(10))
    
    return data


def create_aoi_data_for_surface(surface_data, surface_label):
    """
    Wrapper to create AOI assignments and tag rows with surface label
    Returns dataframe with columns: all original + 'surface' and 'aoi_id' (surface-aware)
    """
    labeled = surface_data.copy()
    labeled['surface'] = surface_label
    df = create_aoi_data(labeled)
    df['surface'] = surface_label
    # create surface-aware AOI id used for cross-screen transitions
    df['aoi_id'] = df['surface'] + '|' + df['aoi'].astype(str)
    return df


# Fixations per aoi

def count_fixations_per_aoi(surface_data, output_dir="output", output_filename="aoi_fixation_counts.csv"):
    """
    Count how many unique fixations landed in each AOI
    
    Args:
        surface_data: Dataframe with AOI assignments
        output_dir: Directory to save results
    
    Returns:
        Dataframe with AOI counts
    """
    aoi_counts = surface_data.groupby('aoi')['fixation_id'].nunique()
    
    aoi_counts_df = aoi_counts.reset_index()
    aoi_counts_df.columns = ['AOI', 'Fixation_Count']
    
    aoi_counts_df = aoi_counts_df.sort_values('Fixation_Count', ascending=False)
    
    print("\n" + "="*60)
    print("STEP 4: Fixation Count Per AOI")
    print("="*60)
    print(aoi_counts_df)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    aoi_counts_df.to_csv(os.path.join(output_dir, output_filename), index=False)
    print(f"\nSaved to: {output_dir}/{output_filename}")
    
    return aoi_counts_df


#Total durations per aoi
def calculate_aoi_durations(surface_data, output_dir="output", output_filename="aoi_durations.csv"):
    """
    Sum up how long people looked at each AOI
    
    Args:
        surface_data: Dataframe with AOI assignments
        output_dir: Directory to save results
    
    Returns:
        Dataframe with total durations per AOI
    """
    # Group by AOI and sum durations
    aoi_durations = surface_data.groupby('aoi')['duration'].sum()
    
    # Convert to dataframe
    aoi_durations_df = aoi_durations.reset_index()
    aoi_durations_df.columns = ['AOI', 'Total_Duration_ms']
    
    # Sort by duration
    aoi_durations_df = aoi_durations_df.sort_values('Total_Duration_ms', ascending=False)
    
    print("\n" + "="*60)
    print("STEP 5: Total Duration Per AOI")
    print("="*60)
    print(aoi_durations_df)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    aoi_durations_df.to_csv(os.path.join(output_dir, output_filename), index=False)
    print(f"\nSaved to: {output_dir}/{output_filename}")
    
    return aoi_durations_df


# Transition sequence (how each gaze moves)
def create_transition_sequence(surface_data, output_dir="output"):
    """
    Track how gaze moves from one AOI to another over time
    
    Args:
        surface_data: Dataframe with AOI assignments
        output_dir: Directory to save results
    
    Returns:
        Dataframe with transitions
    """
    data = surface_data.copy()
    group_cols = ['fixation_id']
    if 'surface' in data.columns:
        group_cols = ['fixation_id', 'surface']

    if data.duplicated(subset=group_cols).any():
        data = collapse_fixations(data, group_cols=group_cols)
        data['aoi'] = data.apply(
            lambda row: assign_aoi(row['norm_pos_x'], row['norm_pos_y']),
            axis=1
        )
        if 'surface' in data.columns:
            data['aoi_id'] = data['surface'] + '|' + data['aoi'].astype(str)

    # Sort by time to get temporal order
    fixation_sequence = data.sort_values('world_timestamp').reset_index(drop=True)

    # Determine AOI column to use (prefer surface-aware 'aoi_id')
    aoi_col = 'aoi_id' if 'aoi_id' in fixation_sequence.columns else 'aoi'

    # Save full fixation sequence (entire ordered path)
    os.makedirs(output_dir, exist_ok=True)
    seq_cols = ['fixation_id', aoi_col, 'world_timestamp']
    seq_cols = [c for c in seq_cols if c in fixation_sequence.columns]
    fixation_sequence[seq_cols].to_csv(
        os.path.join(output_dir, 'full_fixation_sequence.csv'),
        index=False
    )

    # Build next-fixation columns by shifting
    fixation_sequence['next_fixation_id'] = fixation_sequence['fixation_id'].shift(-1)
    fixation_sequence['next_world_timestamp'] = fixation_sequence['world_timestamp'].shift(-1)
    fixation_sequence['next_aoi'] = fixation_sequence[aoi_col].shift(-1)

    # Remove last row (no next AOI)
    transitions = fixation_sequence[fixation_sequence['next_aoi'].notna()].copy()

    # Create transition pairs (label from selected AOI column to next)
    transitions['transition'] = transitions[aoi_col].astype(str) + ' -> ' + transitions['next_aoi'].astype(str)

    # Compute transition duration (time between fixation timestamps)
    # world_timestamp may be in seconds or ms depending on source; preserve units
    transitions['transition_duration'] = transitions['next_world_timestamp'] - transitions['world_timestamp']

    # Extract surface info if using aoi_id format 'Surface X|AOI'
    if aoi_col == 'aoi_id':
        transitions['from_surface'] = transitions[aoi_col].apply(lambda s: s.split('|')[0] if isinstance(s, str) and '|' in s else '')
        transitions['to_surface'] = transitions['next_aoi'].apply(lambda s: s.split('|')[0] if isinstance(s, str) and '|' in s else '')
    else:
        transitions['from_surface'] = ''
        transitions['to_surface'] = ''

    print("\n" + "="*60)
    print("STEP 6: Transition Sequence")
    print("="*60)
    print("First 20 transitions:")
    # Print the columns used for transitions (surface-aware if present)
    display_cols = [aoi_col, 'next_aoi', 'transition', 'world_timestamp', 'next_world_timestamp', 'transition_duration']
    available = [c for c in display_cols if c in transitions.columns]
    print(transitions[available].head(20))

    # Save to CSV (include detailed fields)
    os.makedirs(output_dir, exist_ok=True)
    out_cols = ['fixation_id', aoi_col, 'world_timestamp', 'next_fixation_id', 'next_aoi', 'next_world_timestamp', 'transition_duration', 'transition']
    out_cols = [c for c in out_cols if c in transitions.columns]
    transitions[out_cols].to_csv(os.path.join(output_dir, 'transition_sequence.csv'), index=False)
    print(f"\nSaved to: {output_dir}/transition_sequence.csv")
    print(f"Saved to: {output_dir}/full_fixation_sequence.csv")

    return transitions



def create_transition_matrix(transitions, output_dir="output"):
    """
    Count how many times each AOI-to-AOI transition happens
    
    Args:
        transitions: Dataframe with transitions from Step 6
        output_dir: Directory to save results
    
    Returns:
        Transition matrix
    """
    # Count occurrences of each transition
    # Determine which AOI column is present
    from_col = 'aoi'
    if 'aoi_id' in transitions.columns:
        from_col = 'aoi_id'
    transition_counts = transitions.groupby([from_col, 'next_aoi']).size().reset_index(name='count')
    
    matrix = transition_counts.pivot(index=from_col, columns='next_aoi', values='count')
    
    matrix = matrix.fillna(0)
    
    # Make sure all 9 AOIs are in the matrix (even if count is 0)
    # If using surface-aware AOIs, ensure both surfaces' AOIs are present
    if from_col == 'aoi_id':
        base_aois = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        all_aois = []
        for s in ['Surface 1', 'Surface 2']:
            for b in base_aois:
                all_aois.append(f"{s}|{b}")
    else:
        all_aois = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    for aoi in all_aois:
        if aoi not in matrix.index:
            matrix.loc[aoi] = 0
        if aoi not in matrix.columns:
            matrix[aoi] = 0

    # Reorder to standard layout
    matrix = matrix.loc[all_aois, all_aois]
    
    print("\n" + "="*60)
    print("STEP 7: Transition Matrix")
    print("="*60)
    print(matrix)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    matrix.to_csv(os.path.join(output_dir, 'transition_matrix.csv'))
    print(f"\nSaved to: {output_dir}/transition_matrix.csv")
    
    return matrix


def analyze_cross_screen_transitions(transitions, output_dir="output"):
    """
    Extract transitions that cross surfaces, compute counts and durations, and save details.
    """
    if 'from_surface' not in transitions.columns or 'to_surface' not in transitions.columns:
        print("No surface information in transitions; cannot compute cross-screen transitions.")
        return None

    # Filter cross-screen
    cross = transitions[transitions['from_surface'] != transitions['to_surface']].copy()

    # Compute summary statistics
    total_cross = len(cross)
    if total_cross > 0 and 'transition_duration' in cross.columns:
        avg_duration = cross['transition_duration'].mean()
        median_duration = cross['transition_duration'].median()
    else:
        avg_duration = None
        median_duration = None

    # Save details
    os.makedirs(output_dir, exist_ok=True)
    cols = ['fixation_id', 'aoi_id' if 'aoi_id' in cross.columns else 'aoi', 'world_timestamp',
            'next_fixation_id', 'next_aoi', 'next_world_timestamp', 'transition_duration', 'from_surface', 'to_surface']
    cols = [c for c in cols if c in cross.columns]
    cross.to_csv(os.path.join(output_dir, 'cross_screen_transitions.csv'), index=False)

    # Save summary
    summary = {
        'total_cross_transitions': int(total_cross),
        'avg_transition_duration': float(avg_duration) if avg_duration is not None else None,
        'median_transition_duration': float(median_duration) if median_duration is not None else None
    }
    print("\nCross-screen transitions summary:")
    print(summary)

    return cross, summary


def save_cross_screen_summary_and_visuals(cross, summary, output_dir="output"):
    """
    Save cross-screen summary (JSON/CSV) and create visualizations:
      - bar chart of counts per from->to pair
      - histogram of transition durations
      - timeline scatter of transition timestamps
    """
    import json

    os.makedirs(output_dir, exist_ok=True)

    # Save JSON summary
    with open(os.path.join(output_dir, 'cross_screen_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Save CSV already done by analyze_cross_screen_transitions, also save summary CSV
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(output_dir, 'cross_screen_summary.csv'), index=False)

    if cross is None or cross.empty:
        print("No cross-screen transitions to visualize.")
        return

    # Ensure transition_duration exists
    if 'transition_duration' not in cross.columns:
        print('No transition_duration available; skipping duration plots')

    # Bar chart: counts per from->to pair
    pair_counts = cross.groupby(['from_surface', 'to_surface']).size().reset_index(name='count')
    pair_counts['pair'] = pair_counts['from_surface'] + ' -> ' + pair_counts['to_surface']
    plt.figure(figsize=(8, 4))
    sns.barplot(data=pair_counts, x='pair', y='count', palette='muted')
    plt.title('Cross-Screen Transition Counts')
    plt.xlabel('Transition Pair')
    plt.ylabel('Count')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_transition_counts.png'), dpi=300)
    plt.close()

    # Histogram of transition durations
    if 'transition_duration' in cross.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(cross['transition_duration'].dropna(), bins=20, kde=False, color='steelblue')
        plt.title('Cross-Screen Transition Duration')
        plt.xlabel('Duration (same units as world_timestamp)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cross_transition_duration_hist.png'), dpi=300)
        plt.close()

    # Timeline scatter: timestamp vs duration, colored by direction
    if 'world_timestamp' in cross.columns:
        plt.figure(figsize=(10, 3))
        cross['pair'] = cross['from_surface'] + ' -> ' + cross['to_surface']
        sns.scatterplot(data=cross, x='world_timestamp', y='transition_duration', hue='pair', s=50)
        plt.title('Cross-Screen Transitions Timeline')
        plt.xlabel('World Timestamp')
        plt.ylabel('Transition Duration')
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cross_transition_timeline.png'), dpi=300)
        plt.close()

    print(f"Saved cross-screen summary and visualizations to: {output_dir}")


def visualize_fixation_heatmap(aoi_counts_df, output_dir="output", output_filename="fixation_heatmap.png"):
    """
    Create a visual grid showing where people looked most
    
    Args:
        aoi_counts_df: Dataframe with fixation counts per AOI
        output_dir: Directory to save results
    """
    # Create 3x3 grid of counts
    grid = np.zeros((3, 3))
    
    # Map numeric AOI ids (1-9) to grid positions
    aoi_to_position = {
        1: (0, 0), 2: (0, 1), 3: (0, 2),
        4: (1, 0), 5: (1, 1), 6: (1, 2),
        7: (2, 0), 8: (2, 1), 9: (2, 2)
    }
    
    # Fill grid with counts
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
    
    # Create heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(grid, annot=True, fmt='.0f', cmap='YlOrRd',
                xticklabels=['1/4/7', '2/5/8', '3/6/9'],
                yticklabels=['1-3', '4-6', '7-9'])
    plt.title('Fixation Count Heatmap')
    plt.xlabel('AOI Column (by id)')
    plt.ylabel('AOI Row (by id)')
    
    # Save image
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, output_filename), dpi=300, bbox_inches='tight')
    print("\n" + "="*60)
    print("STEP 8: Fixation Heatmap")
    print("="*60)
    print(f"Saved to: {output_dir}/{output_filename}")
    plt.close()



def visualize_transition_heatmap(matrix, output_dir="output"):
    """
    Create a heatmap of the transition matrix
    
    Args:
        matrix: Transition matrix from Step 7
        output_dir: Directory to save results
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='.0f', cmap='Blues')
    plt.title('AOI Transition Matrix')
    plt.xlabel('To AOI')
    plt.ylabel('From AOI')
    plt.tight_layout()
    
    # Save image
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'transition_matrix_heatmap.png'), dpi=300)
    print("\n" + "="*60)
    print("STEP 9: Transition Matrix Heatmap")
    print("="*60)
    print(f"Saved to: {output_dir}/transition_matrix_heatmap.png")
    plt.close()


def visualize_fixation_density(surface_data, output_dir="output", output_filename="fixation_density.png"):
    """
    Create a 2D density chart of fixation locations on a surface.

    Args:
        surface_data: Dataframe with fixation data (expects norm_pos_x, norm_pos_y)
        output_dir: Directory to save results
    """
    data = surface_data[surface_data['on_surf'] == True].copy()
    if data.empty:
        print("No on-surface fixations found; skipping density chart.")
        return

    # Collapse to one row per fixation before plotting
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


def visualize_transition_path(sequence_data, output_dir="output", output_filename="transition_path.png"):
    """
    Visualize the full fixation sequence across two surfaces as a traced path
    using absolute normalized coordinates.

    Args:
        sequence_data: Dataframe with fixation-level rows (expects surface, norm_pos_x, norm_pos_y, world_timestamp)
        output_dir: Directory to save results
    """
    data = sequence_data.copy()
    if 'surface' not in data.columns or 'norm_pos_x' not in data.columns or 'norm_pos_y' not in data.columns:
        print("Missing surface or normalized positions for transition path; skipping.")
        return

    group_cols = ['fixation_id', 'surface'] if 'surface' in data.columns else ['fixation_id']
    if data.duplicated(subset=group_cols).any():
        data = collapse_fixations(data, group_cols=group_cols)

    data = data.sort_values('world_timestamp').reset_index(drop=True)

    grid_size = 3
    gap = 1
    offsets = {
        'Surface 1': 0,
        'Surface 2': grid_size + gap
    }

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
        x = offsets[surface] + (x_norm * grid_size)
        y = y_norm * grid_size
        xs.append(x)
        ys.append(y)
        labels.append(row['fixation_id'])

    if len(xs) < 2:
        print("Not enough fixations for transition path; skipping.")
        return

    plt.figure(figsize=(9, 4))
    ax = plt.gca()
    ax.set_aspect('equal')

    # Draw grids for both surfaces (3x3)
    for label, x_off in offsets.items():
        for i in range(grid_size + 1):
            ax.plot([x_off, x_off + grid_size], [i, i], color='gray', linewidth=0.6)
            ax.plot([x_off + i, x_off + i], [0, grid_size], color='gray', linewidth=0.6)
        ax.text(x_off + 1.5, -0.4, label, ha='center', va='top')

    # Draw light connecting segments, then label dots
    ax.plot(xs, ys, color='tab:orange', linewidth=0.8, alpha=0.35, linestyle='--')
    ax.scatter(xs, ys, color='tab:orange', s=18, alpha=0.9, zorder=3)
    for x, y, label in zip(xs, ys, labels):
        ax.text(x, y, str(label), fontsize=6, ha='center', va='center', color='black', zorder=4)
    ax.scatter(xs[0], ys[0], color='green', s=30, label='Start')
    ax.scatter(xs[-1], ys[-1], color='red', s=30, label='End')

    ax.set_xlim(-0.2, offsets['Surface 2'] + grid_size + 0.2)
    ax.set_ylim(-0.8, grid_size + 0.2)
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



def main():
    """
    Main function to run the complete analysis pipeline
    """
    print("\n" + "="*60)
    print("EYE-TRACKING ANALYSIS PIPELINE")
    print("="*60)
    
    # Define file paths relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    surface1_path = os.path.join(parent_dir, "example_data/Mateo_data/exports/000/surfaces/fixations_on_surface_Surface 1.csv")
    surface2_path = os.path.join(parent_dir, "example_data/Mateo_data/exports/000/surfaces/fixations_on_surface_Surface 2.csv")
    all_fixations_path = os.path.join(parent_dir, "example_data/Mateo_data/exports/000/fixations.csv")
    output_dir = os.path.join(script_dir, "output")
    
    # STEP 1: Load data
    print("\nSTEP 1: Loading data...")
    surface1, surface2 = load_data(surface1_path, surface2_path)
    
    if surface1 is None or surface2 is None:
        print("Failed to load data. Please check file paths.")
        print(f"Looking for: {surface1_path}")
        print(f"Looking for: {surface2_path}")
        return
    
    # STEP 2: Analyze screen coverage
    all_fixations = None
    if os.path.exists(all_fixations_path):
        all_fixations = pd.read_csv(all_fixations_path)
    else:
        print(f"Note: all-fixations file not found at: {all_fixations_path}")
    coverage = analyze_screen_coverage(surface1, surface2, all_fixations)

    # STEP 3: Create AOI assignments for both screens (surface-aware)
    screen1_aoi = create_aoi_data_for_surface(surface1, "Surface 1")
    screen2_aoi = create_aoi_data_for_surface(surface2, "Surface 2")

    # STEP 4: Count fixations per AOI for each surface and save separate files
    aoi_counts_s1 = count_fixations_per_aoi(
        screen1_aoi,
        output_dir,
        output_filename="aoi_fixation_counts_surface1.csv"
    )
    aoi_counts_s2 = count_fixations_per_aoi(
        screen2_aoi,
        output_dir,
        output_filename="aoi_fixation_counts_surface2.csv"
    )

    # STEP 5: Calculate durations per AOI (per surface)
    aoi_durations_s1 = calculate_aoi_durations(
        screen1_aoi,
        output_dir,
        output_filename="aoi_durations_surface1.csv"
    )
    aoi_durations_s2 = calculate_aoi_durations(
        screen2_aoi,
        output_dir,
        output_filename="aoi_durations_surface2.csv"
    )

    # STEP 6: Create transition sequence across both surfaces
    combined = pd.concat([screen1_aoi, screen2_aoi]).sort_values('world_timestamp')
    transitions = create_transition_sequence(combined, output_dir)

    # STEP 7: Create transition matrix (surface-aware)
    matrix = create_transition_matrix(transitions, output_dir)

    # STEP 8: Visualize fixation heatmaps per surface
    visualize_fixation_heatmap(aoi_counts_s1, output_dir, output_filename="fixation_heatmap_surface1.png")
    visualize_fixation_heatmap(aoi_counts_s2, output_dir, output_filename="fixation_heatmap_surface2.png")

    # STEP 9: Visualize transition heatmap
    visualize_transition_heatmap(matrix, output_dir)

    # STEP 10: Visualize fixation density per surface
    visualize_fixation_density(surface1, output_dir, output_filename="fixation_density_surface1.png")
    visualize_fixation_density(surface2, output_dir, output_filename="fixation_density_surface2.png")

    # STEP 11: Visualize full transition path across both surfaces
    visualize_transition_path(combined, output_dir, output_filename="transition_path.png")

    # EXTRA: Analyze cross-screen transitions (counts, durations, path info)
    cross_details = analyze_cross_screen_transitions(transitions, output_dir)
    if cross_details is not None:
        cross_df, cross_summary = cross_details
        save_cross_screen_summary_and_visuals(cross_df, cross_summary, output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"All results saved to: {output_dir}/")
    print("Generated files:")
    print("  - aoi_fixation_counts_surface1.csv")
    print("  - aoi_fixation_counts_surface2.csv")
    print("  - aoi_durations_surface1.csv")
    print("  - aoi_durations_surface2.csv")
    print("  - transition_sequence.csv")
    print("  - transition_matrix.csv")
    print("  - fixation_heatmap_surface1.png")
    print("  - fixation_heatmap_surface2.png")
    print("  - transition_matrix_heatmap.png")
    print("  - fixation_density_surface1.png")
    print("  - fixation_density_surface2.png")
    print("  - transition_path.png")


if __name__ == "__main__":
    main()
