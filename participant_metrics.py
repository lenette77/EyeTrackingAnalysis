import math
import os

import pandas as pd

from config import DATA_BASE, RUN_IDS, SURFACES
from io_utils import load_surface_fixations
from aoi_analysis import create_aoi_data, create_aoi_data_for_surface


def _safe_entropy(counts):
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log(p, 2) for p in probs)


def _build_transition_sequence(df, aoi_col):
    data = df.sort_values('world_timestamp').reset_index(drop=True)
    data['next_fixation_id'] = data['fixationid'].shift(-1)
    data['next_world_timestamp'] = data['world_timestamp'].shift(-1)
    data['next_aoi'] = data[aoi_col].shift(-1)

    transitions = data[data['next_aoi'].notna()].copy()
    transitions['transition'] = transitions[aoi_col].astype(str) + ' -> ' + transitions['next_aoi'].astype(str)
    transitions['transition_duration'] = transitions['next_world_timestamp'] - transitions['world_timestamp']

    return data, transitions


def _ensure_square_matrix(matrix, labels):
    for label in labels:
        if label not in matrix.index:
            matrix.loc[label] = 0
        if label not in matrix.columns:
            matrix[label] = 0
    return matrix.loc[labels, labels]


def compute_mid_aoi_proportions_from_combined(combined, metrics_dir):
    if combined is None or combined.empty or 'surface' not in combined.columns:
        print('Missing combined AOI data; skipping mid AOI proportions.')
        return None

    mid = combined[combined['surface'] == 'Mid'].copy()
    if mid.empty:
        print('No Mid AOI data in combined; skipping mid AOI proportions.')
        return None

    counts_df = mid.groupby('aoi')['fixationid'].nunique().reset_index()
    counts_df.columns = ['AOI', 'Fixation_Count']

    durations_df = mid.groupby('aoi')['duration'].sum().reset_index()
    durations_df.columns = ['AOI', 'Total_Duration_ms']

    merged = pd.merge(counts_df, durations_df, on='AOI', how='outer').fillna(0)

    total_counts = merged['Fixation_Count'].sum()
    total_duration = merged['Total_Duration_ms'].sum()

    merged['Fixation_Count_Pct'] = merged['Fixation_Count'].apply(
        lambda v: (v / total_counts * 100.0) if total_counts > 0 else 0.0
    )
    merged['Total_Duration_Pct'] = merged['Total_Duration_ms'].apply(
        lambda v: (v / total_duration * 100.0) if total_duration > 0 else 0.0
    )

    merged = merged.sort_values('AOI')
    os.makedirs(metrics_dir, exist_ok=True)
    out_path = os.path.join(metrics_dir, 'aoi_mid_proportions.csv')
    merged.to_csv(out_path, index=False)
    print(f'Saved mid AOI proportions to: {out_path}')
    return merged


def compute_mid_transitions(surface_data_map, metrics_dir):
    mid_df = surface_data_map.get('Mid')
    if mid_df is None:
        print('Mid surface data missing; skipping mid transitions.')
        return None, None, None

    mid_aoi = create_aoi_data(mid_df)
    full_seq, transitions = _build_transition_sequence(mid_aoi, aoi_col='aoi')

    os.makedirs(metrics_dir, exist_ok=True)
    full_seq.to_csv(os.path.join(metrics_dir, 'mid_full_fixation_sequence.csv'), index=False)
    transitions.to_csv(os.path.join(metrics_dir, 'mid_transition_sequence.csv'), index=False)

    base_aois = list(range(1, 10))
    matrix = transitions.groupby(['aoi', 'next_aoi']).size().unstack(fill_value=0)
    matrix = _ensure_square_matrix(matrix, base_aois)
    matrix.to_csv(os.path.join(metrics_dir, 'mid_transition_matrix.csv'))

    return mid_aoi, full_seq, transitions


def compute_screen_transitions(transitions, metrics_dir):
    if 'from_surface' not in transitions.columns or 'to_surface' not in transitions.columns:
        print('No surface info in transitions; skipping screen transition matrix/sequence.')
        return None

    surfaces = [s['label'] for s in SURFACES]
    matrix = transitions.groupby(['from_surface', 'to_surface']).size().unstack(fill_value=0)
    matrix = _ensure_square_matrix(matrix, surfaces)

    os.makedirs(metrics_dir, exist_ok=True)
    matrix.to_csv(os.path.join(metrics_dir, 'screen_transition_matrix.csv'))

    seq_cols = ['from_surface', 'to_surface', 'world_timestamp', 'next_world_timestamp', 'transition_duration']
    surface_seq = transitions[[c for c in seq_cols if c in transitions.columns]].copy()
    surface_seq.to_csv(os.path.join(metrics_dir, 'screen_transition_sequence.csv'), index=False)

    return matrix


def compute_combined_transitions(surface_data_map=None, combined=None):
    if combined is None:
        if surface_data_map is None:
            return None, None, None

        aoi_data = []
        for surface in SURFACES:
            label = surface['label']
            df = surface_data_map.get(label)
            if df is None:
                continue
            aoi_data.append(create_aoi_data_for_surface(df, label))

        if not aoi_data:
            return None, None, None

        combined = pd.concat(aoi_data).sort_values('world_timestamp')

    full_seq, transitions = _build_transition_sequence(combined, aoi_col='aoi_id')

    transitions['from_surface'] = transitions['aoi_id'].apply(
        lambda s: s.split('|')[0] if isinstance(s, str) and '|' in s else ''
    )
    transitions['to_surface'] = transitions['next_aoi'].apply(
        lambda s: s.split('|')[0] if isinstance(s, str) and '|' in s else ''
    )

    return combined, full_seq, transitions


def compute_combined_transitions_with_off_surf(surface_data_map):
    aoi_data = []
    for surface in SURFACES:
        label = surface['label']
        df = surface_data_map.get(label)
        if df is None:
            continue
        aoi_data.append(create_aoi_data_for_surface(df, label, include_off_surf=True))

    if not aoi_data:
        return None, None, None

    combined = pd.concat(aoi_data).sort_values('world_timestamp')
    full_seq, transitions = _build_transition_sequence(combined, aoi_col='aoi_id')

    transitions['from_surface'] = transitions['aoi_id'].apply(
        lambda s: s.split('|')[0] if isinstance(s, str) and '|' in s else ''
    )
    transitions['to_surface'] = transitions['next_aoi'].apply(
        lambda s: s.split('|')[0] if isinstance(s, str) and '|' in s else ''
    )

    return combined, full_seq, transitions


def compute_participant_metrics(run_id, surface_data_map, combined, output_dir):
    metrics_dir = os.path.join(output_dir, 'participant_metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    compute_mid_aoi_proportions_from_combined(combined, metrics_dir)
    mid_aoi, _, mid_transitions = compute_mid_transitions(surface_data_map, metrics_dir)

    combined, _, transitions = compute_combined_transitions(combined=combined)
    if transitions is None or transitions.empty:
        print(f'No combined transitions for {run_id}; skipping transition duration stats and screen matrix.')
        return None

    compute_screen_transitions(transitions, metrics_dir)

    mean_duration = transitions['transition_duration'].mean()
    sd_duration = transitions['transition_duration'].std()

    combined_off, _, transitions_off = compute_combined_transitions_with_off_surf(surface_data_map)
    if combined_off is not None and transitions_off is not None and not transitions_off.empty:
        stationary_all = _safe_entropy(
            combined_off.groupby('aoi_id')['fixationid'].nunique().values
        )
        transition_entropy_all = _safe_entropy(
            transitions_off.groupby(['aoi_id', 'next_aoi']).size().values
        )
    else:
        stationary_all = 0.0
        transition_entropy_all = 0.0

    stationary_mid = 0.0
    transition_entropy_mid = 0.0
    if mid_aoi is not None and mid_transitions is not None:
        stationary_mid = _safe_entropy(
            mid_aoi.groupby('aoi')['fixationid'].nunique().values
        )
        transition_entropy_mid = _safe_entropy(
            mid_transitions.groupby(['aoi', 'next_aoi']).size().values
        )

    return {
        'participant': run_id,
        'transition_duration_mean': mean_duration,
        'transition_duration_sd': sd_duration,
        'stationary_entropy_all_screens': stationary_all,
        'transition_entropy_all_screens': transition_entropy_all,
        'stationary_entropy_mid': stationary_mid,
        'transition_entropy_mid': transition_entropy_mid
    }


def save_participant_summary(summary_rows, output_dir):
    if not summary_rows:
        return
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, 'participant_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f'Saved participant summary to: {summary_path}')


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    summary_rows = []

    for run_id in RUN_IDS:
        print('=' * 60)
        print(f'PARTICIPANT: {run_id}')
        print('=' * 60)

        data_root = os.path.join(DATA_BASE, run_id, 'exports', run_id)
        surfaces_dir = os.path.join(data_root, 'surfaces')
        output_dir = os.path.join(script_dir, 'output', run_id)

        surface_data_map = load_surface_fixations(surfaces_dir, SURFACES)
        if surface_data_map is None:
            continue

        combined, _, _ = compute_combined_transitions(surface_data_map)
        summary = compute_participant_metrics(run_id, surface_data_map, combined, output_dir)
        if summary is not None:
            summary_rows.append(summary)

    save_participant_summary(summary_rows, os.path.join(script_dir, 'output'))


if __name__ == '__main__':
    main()
