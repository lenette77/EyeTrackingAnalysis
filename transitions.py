import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from aoi_analysis import collapse_fixations, assign_aoi


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

    print("=" * 60)
    print("STEP 6: Transition Sequence")
    print("=" * 60)
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
        for surface in surface_order:
            for base in base_aois:
                all_aois.append(f"{surface}|{base}")
    else:
        all_aois = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    for aoi in all_aois:
        if aoi not in matrix.index:
            matrix.loc[aoi] = 0
        if aoi not in matrix.columns:
            matrix[aoi] = 0

    matrix = matrix.loc[all_aois, all_aois]

    print("=" * 60)
    print("STEP 7: Transition Matrix")
    print("=" * 60)
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
    offsets = _surface_offsets(surface_order, grid_size=grid_size, gap=gap)

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


def visualize_transition_flow_map(transitions, surface_order, output_dir, filename, title="Transition Flow Map"):
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
    ax.set_title(title)

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


def run_transition_analysis(combined, surface_order, output_dir):
    """Compute transition tables and visualizations."""
    transitions = create_transition_sequence(combined, output_dir)
    matrix = create_transition_matrix(transitions, output_dir, surface_order=surface_order)
    visualize_transition_heatmap(matrix, output_dir)
    visualize_transition_path(combined, surface_order, output_dir, output_filename="transition_path.png")
    visualize_transition_flow_map(transitions, surface_order, output_dir,
                                 filename="transition_flow_map.png")

    if 'from_surface' in transitions.columns and 'to_surface' in transitions.columns:
        intra = transitions[transitions['from_surface'] == transitions['to_surface']].copy()
        inter = transitions[transitions['from_surface'] != transitions['to_surface']].copy()
        visualize_transition_flow_map(intra, surface_order, output_dir,
                                     filename="transition_flow_map_intra.png",
                                     title="Intra-Surface Transition Flow")
        visualize_transition_flow_map(inter, surface_order, output_dir,
                                     filename="transition_flow_map_inter.png",
                                     title="Inter-Surface Transition Flow")
    else:
        print("No surface labels; skipping intra/inter flow map split.")

    cross_details = analyze_cross_screen_transitions(transitions, output_dir)
    if cross_details is not None:
        cross_df, cross_summary = cross_details
        save_cross_screen_summary_and_visuals(cross_df, cross_summary, output_dir)
        visualize_cross_transition_density(transitions, surface_order, output_dir,
                                           filename="cross_transition_density.png")

    compute_transition_summary(transitions, output_dir, filename="transition_summary.csv")
