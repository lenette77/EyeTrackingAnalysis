import os
import warnings

from config import DATA_BASE, RUN_IDS, SURFACES
from io_utils import (
    load_surface_fixations,
    load_all_fixations,
    load_cached_fixations,
    save_cached_fixations,
    load_cached_analysis,
    save_cached_analysis
)
from aoi_analysis import analyze_surface_coverage, run_aoi_metrics
from transitions import run_transition_analysis
from participant_metrics import compute_participant_metrics, save_participant_summary
from group_aggregation import main as run_group_aggregation

warnings.filterwarnings('ignore')


def _find_export_roots(session_dir):
    exports_dir = os.path.join(session_dir, 'exports')
    if os.path.isdir(exports_dir):
        subdirs = [os.path.join(exports_dir, name) for name in os.listdir(exports_dir)
                   if os.path.isdir(os.path.join(exports_dir, name))]
        if subdirs:
            candidates = [path for path in subdirs if os.path.isdir(os.path.join(path, 'surfaces'))]
            if not candidates:
                return []
            most_recent = max(candidates, key=os.path.getmtime)
            return [most_recent]
        if os.path.isdir(os.path.join(exports_dir, 'surfaces')):
            return [exports_dir]
    if os.path.isdir(os.path.join(session_dir, 'surfaces')):
        return [session_dir]
    return []


def _iter_participant_sessions(data_base):
    for participant_name in os.listdir(data_base):
        participant_dir = os.path.join(data_base, participant_name)
        if not os.path.isdir(participant_dir):
            continue
        for session_name in os.listdir(participant_dir):
            session_dir = os.path.join(participant_dir, session_name)
            if os.path.isdir(session_dir):
                yield participant_name, session_name, session_dir


def _is_ae_session(session_name):
    if '_' not in session_name:
        return False
    remainder = session_name.split('_', 1)[1].lower()
    return remainder.startswith('ae')


def _build_output_dir(script_dir, participant_name, session_name, export_root):
    output_dir = os.path.join(script_dir, 'output', participant_name, session_name)
    export_name = os.path.basename(export_root)
    if export_name not in (session_name, 'exports'):
        output_dir = os.path.join(output_dir, export_name)
    return output_dir


def _infer_group(participant_id):
    if not participant_id:
        return ''
    if participant_id.startswith('NA'):
        return 'NA'
    if participant_id.startswith('A'):
        return 'A'
    return ''

def main():
    """Run per-participant analyses, participant metrics, and group aggregation."""
    print("-" * 60)
    print("EYE-TRACKING ANALYSIS: AOIs + SCAN PATTERNS")
    print("-" * 60)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    summary_rows = []

    if RUN_IDS:
        for run_id in RUN_IDS:
            print("\n" + "-" * 60)
            print(f"Currently processing RUN: {run_id}")
            print("-" * 60)

            data_root = os.path.join(DATA_BASE, run_id, "exports", run_id)
            surfaces_dir = os.path.join(data_root, "surfaces")
            output_dir = os.path.join(script_dir, "output", run_id)
            os.makedirs(output_dir, exist_ok=True)
            cache_dir = os.path.join(output_dir, ".cache")
            surface_data_map, all_fixations, cache_hit = load_cached_fixations(
                cache_dir,
                surfaces_dir,
                SURFACES,
                data_root,
                allow_missing=False
            )
            if not cache_hit:
                surface_data_map = load_surface_fixations(surfaces_dir, SURFACES)
                if surface_data_map is None:
                    return

                all_fixations = load_all_fixations(data_root)
                save_cached_fixations(
                    cache_dir,
                    surfaces_dir,
                    SURFACES,
                    data_root,
                    surface_data_map,
                    all_fixations,
                    allow_missing=False
                )
            surface_order = [s["label"] for s in SURFACES]

            aoi_data_map_cached, combined_cached, analysis_cache_hit = load_cached_analysis(
                cache_dir,
                surfaces_dir,
                SURFACES,
                data_root,
                allow_missing=False
            )

            analyze_surface_coverage(surface_data_map, all_fixations)

            combined, aoi_data_map = run_aoi_metrics(
                surface_data_map,
                SURFACES,
                output_dir,
                cached_aoi_data_map=aoi_data_map_cached if analysis_cache_hit else None,
                cached_combined=combined_cached if analysis_cache_hit else None
            )
            if not analysis_cache_hit:
                save_cached_analysis(
                    cache_dir,
                    surfaces_dir,
                    SURFACES,
                    data_root,
                    aoi_data_map,
                    combined,
                    allow_missing=False
                )
            run_transition_analysis(combined, surface_order, output_dir, surface_data_map=surface_data_map)

            summary = compute_participant_metrics(run_id, surface_data_map, combined, output_dir)
            if summary is not None:
                summary['group'] = _infer_group(run_id)
                summary['participant_id'] = run_id
                summary['session_id'] = ''
                summary_rows.append(summary)
    else:
        for participant_name, session_name, session_dir in _iter_participant_sessions(DATA_BASE):
            if _is_ae_session(session_name):
                continue

            export_roots = _find_export_roots(session_dir)
            if not export_roots:
                continue

            for export_root in export_roots:
                print("\n" + "-" * 60)
                print(f"Currently processing SESSION: {participant_name}/{session_name}")
                print("-" * 60)

                data_root = export_root
                surfaces_dir = os.path.join(data_root, "surfaces")
                output_dir = _build_output_dir(script_dir, participant_name, session_name, export_root)
                os.makedirs(output_dir, exist_ok=True)
                cache_dir = os.path.join(output_dir, ".cache")
                surface_data_map, all_fixations, cache_hit = load_cached_fixations(
                    cache_dir,
                    surfaces_dir,
                    SURFACES,
                    data_root,
                    allow_missing=True
                )
                if not cache_hit:
                    surface_data_map = load_surface_fixations(surfaces_dir, SURFACES, allow_missing=True)
                    if surface_data_map is None:
                        continue

                    all_fixations = load_all_fixations(data_root)
                    save_cached_fixations(
                        cache_dir,
                        surfaces_dir,
                        SURFACES,
                        data_root,
                        surface_data_map,
                        all_fixations,
                        allow_missing=True
                    )
                surface_order = [s["label"] for s in SURFACES]

                aoi_data_map_cached, combined_cached, analysis_cache_hit = load_cached_analysis(
                    cache_dir,
                    surfaces_dir,
                    SURFACES,
                    data_root,
                    allow_missing=True
                )

                analyze_surface_coverage(surface_data_map, all_fixations)

                combined, aoi_data_map = run_aoi_metrics(
                    surface_data_map,
                    SURFACES,
                    output_dir,
                    cached_aoi_data_map=aoi_data_map_cached if analysis_cache_hit else None,
                    cached_combined=combined_cached if analysis_cache_hit else None
                )
                if not analysis_cache_hit:
                    save_cached_analysis(
                        cache_dir,
                        surfaces_dir,
                        SURFACES,
                        data_root,
                        aoi_data_map,
                        combined,
                        allow_missing=True
                    )
                run_transition_analysis(combined, surface_order, output_dir, surface_data_map=surface_data_map)

                run_id_label = f"{participant_name}:{session_name}"
                summary = compute_participant_metrics(run_id_label, surface_data_map, combined, output_dir)
                if summary is not None:
                    summary['group'] = _infer_group(participant_name)
                    summary['participant_id'] = participant_name
                    summary['session_id'] = session_name
                    summary_rows.append(summary)

    print("\n=== PARTICIPANT METRICS ===")
    save_participant_summary(summary_rows, os.path.join(script_dir, 'output'))

    print("\n=== GROUP AGGREGATION ===")
    try:
        run_group_aggregation()
    except FileNotFoundError as exc:
        print(str(exc))

    print("\n=== ANALYSIS COMPLETE ===")
    print("Analysis outputs saved to the output folder")


if __name__ == "__main__":
    main()
