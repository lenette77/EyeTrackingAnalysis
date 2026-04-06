import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import seaborn as sns


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

    print("=" * 60)
    print("Fixation Surface Coverage Analysis")
    print("=" * 60)
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
    """Takes x,y coordinates and returns which AOI it belongs to (1-9 row-major)."""
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
    return ids[row * 3 + col]


def collapse_fixations(surface_data, group_cols=None):
    """Collapse multiple rows per fixation into one row using sensible aggregates."""
    if group_cols is None:
        group_cols = ['fixationid']
    # If surface labels exist, include them unless caller provided a different set.
    if 'surface' in surface_data.columns and 'surface' not in group_cols:
        group_cols = list(group_cols) + ['surface']

    # Drop any grouping columns that are not present to avoid KeyError.
    group_cols = [col for col in group_cols if col in surface_data.columns]
    if not group_cols:
        return surface_data

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


def create_aoi_data(surface_data, include_off_surf=False):
    """Assign AOI to each fixation on the surface."""
    if include_off_surf:
        data = surface_data.copy()
    else:
        data = surface_data[surface_data['on_surf'] == True].copy()

    data = collapse_fixations(data)

    if include_off_surf:
        data['aoi'] = data.apply(
            lambda row: assign_aoi(row['norm_pos_x'], row['norm_pos_y'])
            if row.get('on_surf') is True else 'OFF_SURF',
            axis=1
        )
    else:
        data['aoi'] = data.apply(lambda row: assign_aoi(row['norm_pos_x'], row['norm_pos_y']), axis=1)
    print("-" * 60)
    print("STEP 3: AOI Assignment (3x3 Grid)")
    print("-" * 60)
    print(f"Assigned {len(data)} fixations to AOIs")
    print("AOI assignments:")
    print(data[['fixationid', 'norm_pos_x', 'norm_pos_y', 'aoi']].head(10))
    return data


def create_aoi_data_for_surface(surface_data, surface_label, include_off_surf=False):
    """Create AOI assignments and tag rows with surface label (surface-aware AOIs)."""
    labeled = surface_data.copy()
    labeled['surface'] = surface_label
    df = create_aoi_data(labeled, include_off_surf=include_off_surf)
    df['surface'] = surface_label
    df['aoi_id'] = df['surface'] + '|' + df['aoi'].astype(str)
    return df


def count_fixations_per_aoi(surface_data, output_dir="output", output_filename="aoi_fixation_counts.csv"):
    """Count how many unique fixations landed in each AOI."""
    aoi_counts = surface_data.groupby('aoi')['fixationid'].nunique()
    aoi_counts_df = aoi_counts.reset_index()
    aoi_counts_df.columns = ['AOI', 'Fixation_Count']
    aoi_counts_df = aoi_counts_df.sort_values('Fixation_Count', ascending=False)

    print("=" * 60)
    print("STEP 4: Fixation Count Per AOI")
    print("=" * 60)
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

    print("=" * 60)
    print("STEP 5: Total Duration Per AOI")
    print("=" * 60)
    print(aoi_durations_df)

    os.makedirs(output_dir, exist_ok=True)
    aoi_durations_df.to_csv(os.path.join(output_dir, output_filename), index=False)
    print(f"Saved to: {output_dir}/{output_filename}")

    return aoi_durations_df


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


def visualize_fixation_density(surface_data, output_dir="output", output_filename="fixation_density.png"):
    """Create a 2D density chart of fixation locations on a surface."""
    data = surface_data[surface_data['on_surf'] == True].copy()
    if data.empty:
        print("No on-surface fixations found; skipping density chart.")
        return

    data = collapse_fixations(data)

    density_cmap = mcolors.LinearSegmentedColormap.from_list(
        'density_gray_red', ['#FFFFFF', '#898989', '#9B0000']
    )

    plt.figure(figsize=(6, 5), facecolor='white')
    sns.kdeplot(
        data=data,
        x='norm_pos_x',
        y='norm_pos_y',
        fill=True,
        levels=50,
        cmap=density_cmap,
        bw_adjust=0.8
    )
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.set_title('Fixation Density', color='#898989')
    ax.set_xlabel('Normalized X', color='#898989')
    ax.set_ylabel('Normalized Y', color='#898989')
    ax.tick_params(colors='#898989')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, output_filename), dpi=300)
    print(f"Saved to: {output_dir}/{output_filename}")
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


def run_aoi_metrics(surface_data_map, surfaces, output_dir, cached_aoi_data_map=None, cached_combined=None):
    """Compute AOI metrics and return combined AOI data."""
    aoi_data = []
    aoi_data_map = {}
    for surface in surfaces:
        label = surface["label"]
        df = surface_data_map[label]
        if cached_aoi_data_map and label in cached_aoi_data_map:
            aoi_df = cached_aoi_data_map[label].copy()
        else:
            aoi_df = create_aoi_data_for_surface(df, label)
        aoi_data.append(aoi_df)
        aoi_data_map[label] = aoi_df

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

    if cached_combined is not None:
        combined = cached_combined.copy()
    else:
        combined = pd.concat(aoi_data).sort_values('world_timestamp')
    compute_aoi_summary(combined, output_dir, filename="aoi_summary.csv")
    return combined, aoi_data_map
