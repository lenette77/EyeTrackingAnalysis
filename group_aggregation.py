import os
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import DATA_BASE, GROUP_DATA_BASE, RUN_IDS, SURFACES
from io_utils import load_surface_fixations
from aoi_analysis import create_aoi_data, create_aoi_data_for_surface, visualize_fixation_density
from transitions import (
    analyze_cross_screen_transitions,
    compute_transition_summary,
    create_transition_matrix,
    save_cross_screen_summary_and_visuals,
    visualize_combined_density_and_flow_map,
    visualize_cross_transition_density,
    visualize_transition_flow_map,
    visualize_transition_flow_heatmap,
    _surface_offsets,
)


DELAY_LABELS = [
    'baseline',
    'delay1',
    'delay2',
    'delay3',
]

DELAY_OUTPUT_LABELS = {
    'baseline': 'baseline',
    'delay1': 'delay1',
    'delay2': 'delay2',
    'delay3': 'delay3',
}


def _find_export_roots(session_dir):
    exports_dir = os.path.join(session_dir, 'exports')
    if os.path.isdir(exports_dir):
        subdirs = [os.path.join(exports_dir, name) for name in os.listdir(exports_dir)
                   if os.path.isdir(os.path.join(exports_dir, name))]
        if subdirs:
            return [path for path in subdirs if os.path.isdir(os.path.join(path, 'surfaces'))]
        if os.path.isdir(os.path.join(exports_dir, 'surfaces')):
            return [exports_dir]
    if os.path.isdir(os.path.join(session_dir, 'surfaces')):
        return [session_dir]
    return []


def _iter_group_sessions(group_dir):
    for name in os.listdir(group_dir):
        path = os.path.join(group_dir, name)
        if os.path.isdir(path):
            yield name, path


def _match_delay_label(session_name):
    if '_' not in session_name:
        return None
    remainder = session_name.split('_', 1)[1].lower()
    for label in DELAY_LABELS:
        if remainder == label or remainder.startswith(f"{label}_"):
            return label
    return None


def _is_ae_session(session_name):
    if '_' not in session_name:
        return False
    remainder = session_name.split('_', 1)[1].lower()
    return remainder.startswith('ae')


def _build_session_transitions(surface_data_map, session_id):
    aoi_data = []
    for surface in SURFACES:
        label = surface['label']
        df = surface_data_map.get(label)
        if df is None:
            continue
        aoi_data.append(create_aoi_data_for_surface(df, label))

    if not aoi_data:
        return None, None

    combined = pd.concat(aoi_data).sort_values('world_timestamp')
    combined['session_id'] = session_id

    seq = combined.sort_values('world_timestamp').reset_index(drop=True)
    seq['next_fixation_id'] = seq['fixationid'].shift(-1)
    seq['next_world_timestamp'] = seq['world_timestamp'].shift(-1)
    seq['next_aoi'] = seq['aoi_id'].shift(-1)

    transitions = seq[seq['next_aoi'].notna()].copy()
    transitions['transition'] = transitions['aoi_id'].astype(str) + ' -> ' + transitions['next_aoi'].astype(str)
    transitions['transition_duration'] = transitions['next_world_timestamp'] - transitions['world_timestamp']
    transitions['from_surface'] = transitions['aoi_id'].apply(
        lambda s: s.split('|')[0] if isinstance(s, str) and '|' in s else ''
    )
    transitions['to_surface'] = transitions['next_aoi'].apply(
        lambda s: s.split('|')[0] if isinstance(s, str) and '|' in s else ''
    )
    transitions['session_id'] = session_id

    return combined, transitions


def _accumulate_session(surface_data_map, session_id, surface_data_combined, transitions_all, sequences_all):
    combined_session, transitions = _build_session_transitions(surface_data_map, session_id)
    if combined_session is not None:
        sequences_all.append(combined_session)

    if transitions is not None and not transitions.empty:
        transitions_all.append(transitions)

    for label, df in surface_data_map.items():
        df = df.copy()
        df['session_id'] = session_id
        surface_data_combined[label].append(df)


def _finalize_group(group_name, combined_surface_map, transitions_all, sequences_all, output_base):
    group_output_dir = os.path.join(output_base, group_name)
    os.makedirs(group_output_dir, exist_ok=True)

    for label, df in combined_surface_map.items():
        if df.empty:
            continue
        label_slug = label.lower().replace(' ', '_')
        visualize_fixation_density(df, group_output_dir, f'group_fixation_density_{label_slug}.png')

    surface_counts = {}
    for label, df in combined_surface_map.items():
        if df.empty:
            continue
        aoi_df = create_aoi_data(df)
        counts = aoi_df.groupby('aoi')['fixationid'].nunique()
        surface_counts[label] = counts.to_dict()

    _plot_group_heatmaps(surface_counts, group_output_dir)

    if transitions_all:
        transitions_df = pd.concat(transitions_all, ignore_index=True)
        surface_order = [s['label'] for s in SURFACES]
        matrix = create_transition_matrix(transitions_df, output_dir=group_output_dir, surface_order=surface_order)
        visualize_transition_flow_heatmap(
            surface_counts,
            transitions_df,
            surface_order,
            group_output_dir,
            'group_transition_flow_heatmap.png'
        )

        intra = transitions_df[transitions_df['from_surface'] == transitions_df['to_surface']].copy()
        inter = transitions_df[transitions_df['from_surface'] != transitions_df['to_surface']].copy()
        visualize_transition_flow_map(intra, surface_order, group_output_dir,
                                      filename='group_transition_flow_map_intra.png',
                                      title='Group Intra-Surface Transition Flow')
        visualize_transition_flow_map(inter, surface_order, group_output_dir,
                                      filename='group_transition_flow_map_inter.png',
                                      title='Group Inter-Surface Transition Flow')

        cross_details = analyze_cross_screen_transitions(transitions_df, output_dir=group_output_dir)
        if cross_details is not None:
            cross_df, cross_summary = cross_details
            save_cross_screen_summary_and_visuals(cross_df, cross_summary, group_output_dir)
            visualize_cross_transition_density(transitions_df, surface_order, group_output_dir,
                                               filename='group_cross_transition_density.png')

        compute_transition_summary(transitions_df, output_dir=group_output_dir,
                                   filename='group_transition_summary.csv')

        visualize_combined_density_and_flow_map(
            combined_surface_map,
            transitions_df,
            group_output_dir,
            filename='group_fixation_density_and_transition_flow_map.png'
        )

    _plot_group_transition_path(sequences_all, [s['label'] for s in SURFACES],
                                group_output_dir, 'group_transition_path.png')


def _plot_group_transition_path(all_sequences, surface_order, output_dir, filename):
    if not all_sequences:
        print('No sequences available; skipping group transition path.')
        return

    grid_size = 3
    gap = 1
    offsets = _surface_offsets(surface_order, grid_size=grid_size, gap=gap)

    plt.figure(figsize=(12, 4))
    ax = plt.gca()
    ax.set_aspect('equal')

    for label, (x_off, y_off) in offsets.items():
        for i in range(grid_size + 1):
            ax.plot([x_off, x_off + grid_size], [y_off + i, y_off + i], color='gray', linewidth=0.6)
            ax.plot([x_off + i, x_off + i], [y_off, y_off + grid_size], color='gray', linewidth=0.6)
        ax.text(x_off + 1.5, y_off - 0.4, label, ha='center', va='top')

    for seq in all_sequences:
        xs = []
        ys = []
        for _, row in seq.iterrows():
            surface = row.get('surface')
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
            xs.append(x_off + (x_norm * grid_size))
            ys.append(y_off + (y_norm * grid_size))

        if len(xs) < 2:
            continue
        ax.plot(xs, ys, color='tab:orange', linewidth=0.6, alpha=0.2)

    max_x = max(pos[0] for pos in offsets.values()) + grid_size + 0.2
    max_y = max(pos[1] for pos in offsets.values()) + grid_size + 0.2
    ax.set_xlim(-0.2, max_x)
    ax.set_ylim(-0.8, max_y)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Group Transition Path (Overlay)')

    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


def _build_heatmap_grid(aoi_counts, max_count, ax, title):
    grid = np.zeros((3, 3))
    aoi_to_position = {
        1: (0, 0), 2: (0, 1), 3: (0, 2),
        4: (1, 0), 5: (1, 1), 6: (1, 2),
        7: (2, 0), 8: (2, 1), 9: (2, 2)
    }

    for aoi_value, count in aoi_counts.items():
        try:
            aoi_value = int(aoi_value)
        except (TypeError, ValueError):
            continue
        if aoi_value in aoi_to_position:
            pos = aoi_to_position[aoi_value]
            grid[pos[0], pos[1]] = count

    sns.heatmap(
        grid,
        annot=True,
        fmt='.0f',
        cmap='YlOrRd',
        vmin=0,
        vmax=max_count if max_count > 0 else None,
        xticklabels=['1/4/7', '2/5/8', '3/6/9'],
        yticklabels=['1-3', '4-6', '7-9'],
        ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel('AOI Column (by id)')
    ax.set_ylabel('AOI Row (by id)')


def _plot_group_heatmaps(surface_counts, output_dir):
    if not surface_counts:
        print('No AOI counts available; skipping group heatmaps.')
        return

    max_count = max(int(max(counts.values())) for counts in surface_counts.values() if len(counts))
    labels = list(surface_counts.keys())
    cols = 2
    rows = int(math.ceil(len(labels) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    for idx, label in enumerate(labels):
        r = idx // cols
        c = idx % cols
        _build_heatmap_grid(surface_counts[label], max_count, axes[r, c], f'{label} Fixation Heatmap')

    for idx in range(len(labels), rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis('off')

    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'group_fixation_heatmaps.png'), dpi=300)
    plt.close(fig)


def aggregate_group(group_name, group_dir, output_base):
    print('=' * 60)
    print(f'AGGREGATING GROUP: {group_name}')
    print('=' * 60)

    surface_data_combined = {surface['label']: [] for surface in SURFACES}
    transitions_all = []
    sequences_all = []

    for session_name, session_dir in _iter_group_sessions(group_dir):
        if _is_ae_session(session_name):
            continue
        export_roots = _find_export_roots(session_dir)
        if not export_roots:
            print(f'No exports for {session_name}; skipping.')
            continue

        for export_root in export_roots:
            surfaces_dir = os.path.join(export_root, 'surfaces')
            surface_data_map = load_surface_fixations(surfaces_dir, SURFACES, allow_missing=True)
            if surface_data_map is None:
                continue

            session_id = f'{session_name}:{os.path.basename(export_root)}'
            _accumulate_session(surface_data_map, session_id, surface_data_combined, transitions_all, sequences_all)

    if not any(surface_data_combined[label] for label in surface_data_combined):
        print(f'No surface data found for group {group_name}; skipping.')
        return

    combined_surface_map = {
        label: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        for label, parts in surface_data_combined.items()
    }

    _finalize_group(group_name, combined_surface_map, transitions_all, sequences_all, output_base)


def aggregate_group_dirs(group_name, group_dirs, output_base):
    print('=' * 60)
    print(f'AGGREGATING GROUP (PREFIX): {group_name}')
    print('=' * 60)

    surface_data_combined = {surface['label']: [] for surface in SURFACES}
    transitions_all = []
    sequences_all = []

    for group_dir in group_dirs:
        if not os.path.isdir(group_dir):
            continue
        for session_name, session_dir in _iter_group_sessions(group_dir):
            if _is_ae_session(session_name):
                continue
            export_roots = _find_export_roots(session_dir)
            if not export_roots:
                print(f'No exports for {session_name}; skipping.')
                continue

            for export_root in export_roots:
                surfaces_dir = os.path.join(export_root, 'surfaces')
                surface_data_map = load_surface_fixations(surfaces_dir, SURFACES, allow_missing=True)
                if surface_data_map is None:
                    continue

                participant_id = os.path.basename(group_dir)
                session_id = f'{participant_id}:{session_name}:{os.path.basename(export_root)}'
                _accumulate_session(surface_data_map, session_id, surface_data_combined, transitions_all, sequences_all)

    if not any(surface_data_combined[label] for label in surface_data_combined):
        print(f'No surface data found for group {group_name}; skipping.')
        return

    combined_surface_map = {
        label: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        for label, parts in surface_data_combined.items()
    }

    _finalize_group(group_name, combined_surface_map, transitions_all, sequences_all, output_base)


def aggregate_group_dirs_by_delay(group_name, group_dirs, output_base, delay_label):
    print('=' * 60)
    print(f'AGGREGATING GROUP (PREFIX + DELAY): {group_name}')
    print('=' * 60)

    surface_data_combined = {surface['label']: [] for surface in SURFACES}
    transitions_all = []
    sequences_all = []

    for group_dir in group_dirs:
        if not os.path.isdir(group_dir):
            continue
        for session_name, session_dir in _iter_group_sessions(group_dir):
            if _is_ae_session(session_name):
                continue
            if _match_delay_label(session_name) != delay_label:
                continue

            export_roots = _find_export_roots(session_dir)
            if not export_roots:
                print(f'No exports for {session_name}; skipping.')
                continue

            for export_root in export_roots:
                surfaces_dir = os.path.join(export_root, 'surfaces')
                surface_data_map = load_surface_fixations(surfaces_dir, SURFACES, allow_missing=True)
                if surface_data_map is None:
                    continue

                participant_id = os.path.basename(group_dir)
                session_id = f'{participant_id}:{session_name}:{os.path.basename(export_root)}'
                _accumulate_session(surface_data_map, session_id, surface_data_combined, transitions_all, sequences_all)

    if not any(surface_data_combined[label] for label in surface_data_combined):
        print(f'No surface data found for group {group_name}; skipping.')
        return

    combined_surface_map = {
        label: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        for label, parts in surface_data_combined.items()
    }

    _finalize_group(group_name, combined_surface_map, transitions_all, sequences_all, output_base)


def aggregate_runs_as_group(group_name, run_ids, output_base):
    print('=' * 60)
    print(f'AGGREGATING GROUP (RUN IDS): {group_name}')
    print('=' * 60)

    surface_data_combined = {surface['label']: [] for surface in SURFACES}
    transitions_all = []
    sequences_all = []

    for run_id in run_ids:
        data_root = os.path.join(DATA_BASE, run_id, 'exports', run_id)
        surfaces_dir = os.path.join(data_root, 'surfaces')
        if not os.path.isdir(surfaces_dir):
            print(f'No exports for {run_id}; skipping.')
            continue

        surface_data_map = load_surface_fixations(surfaces_dir, SURFACES, allow_missing=True)
        if surface_data_map is None:
            continue

        _accumulate_session(surface_data_map, run_id, surface_data_combined, transitions_all, sequences_all)

    if not any(surface_data_combined[label] for label in surface_data_combined):
        print(f'No surface data found for group {group_name}; skipping.')
        return

    combined_surface_map = {
        label: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        for label, parts in surface_data_combined.items()
    }

    _finalize_group(group_name, combined_surface_map, transitions_all, sequences_all, output_base)


def main():
    output_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'groups')
    os.makedirs(output_base, exist_ok=True)

    if GROUP_DATA_BASE and os.path.isdir(GROUP_DATA_BASE):
        participant_names = []
        prefix_groups = {'A': [], 'NA': []}
        for name in os.listdir(GROUP_DATA_BASE):
            group_dir = os.path.join(GROUP_DATA_BASE, name)
            if not os.path.isdir(group_dir):
                continue
            if name.startswith('NA'):
                prefix_groups['NA'].append(name)
                participant_names.append(name)
            elif name.startswith('A'):
                prefix_groups['A'].append(name)
                participant_names.append(name)

        for participant in participant_names:
            participant_dir = os.path.join(GROUP_DATA_BASE, participant)
            aggregate_group(participant, participant_dir, output_base)

        for prefix, group_names in prefix_groups.items():
            group_dirs = [os.path.join(GROUP_DATA_BASE, name) for name in group_names]
            aggregate_group_dirs(prefix, group_dirs, output_base)

            for delay_label in DELAY_LABELS:
                output_label = DELAY_OUTPUT_LABELS[delay_label]
                delay_group_name = f'{prefix}_{output_label}'
                aggregate_group_dirs_by_delay(delay_group_name, group_dirs, output_base, delay_label)
    else:
        print(f'GROUP_DATA_BASE not found: {GROUP_DATA_BASE}')

    if RUN_IDS:
        aggregate_runs_as_group('demo_runs', RUN_IDS, output_base)


if __name__ == '__main__':
    main()
