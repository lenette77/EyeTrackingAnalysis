import os
import warnings

from config import DATA_BASE, RUN_IDS, SURFACES
from io_utils import load_surface_fixations, load_all_fixations
from aoi_analysis import analyze_surface_coverage, run_aoi_metrics
from scan_patterns import run_scan_patterns
from transitions import run_transition_analysis
from participant_metrics import compute_participant_metrics, save_participant_summary
from group_aggregation import main as run_group_aggregation

warnings.filterwarnings('ignore')

def main():
    """Run per-participant analyses, participant metrics, and group aggregation."""
    print("-" * 60)
    print("EYE-TRACKING ANALYSIS: AOIs + SCAN PATTERNS")
    print("-" * 60)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    summary_rows = []

    for run_id in RUN_IDS:
        print("\n" + "-" * 60)
        print(f"Currently processing RUN: {run_id}")
        print("-" * 60)

        data_root = os.path.join(DATA_BASE, run_id, "exports", run_id)
        surfaces_dir = os.path.join(data_root, "surfaces")
        output_dir = os.path.join(script_dir, "output", run_id)
        os.makedirs(output_dir, exist_ok=True)

        surface_data_map = load_surface_fixations(surfaces_dir, SURFACES)
        if surface_data_map is None:
            return

        all_fixations = load_all_fixations(data_root)
        surface_order = [s["label"] for s in SURFACES]

        analyze_surface_coverage(surface_data_map, all_fixations)
        run_scan_patterns(data_root, output_dir)

        combined = run_aoi_metrics(surface_data_map, SURFACES, output_dir)
        run_transition_analysis(combined, surface_order, output_dir, surface_data_map=surface_data_map)

        summary = compute_participant_metrics(run_id, surface_data_map, combined, output_dir)
        if summary is not None:
            summary_rows.append(summary)

    print("\n=== PARTICIPANT METRICS ===")
    save_participant_summary(summary_rows, os.path.join(script_dir, 'output'))

    print("\n=== GROUP AGGREGATION ===")
    try:
        run_group_aggregation()
    except FileNotFoundError as exc:
        print(str(exc))

    print("\n=== ANALYSIS COMPLETE ===")
    print("New files: scan_patterns.csv/png (if gaze_positions.csv is available)")


if __name__ == "__main__":
    main()
