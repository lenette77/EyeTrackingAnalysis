import os
import numpy as np
import pandas as pd

try:
    import stumpy
except ImportError:
    stumpy = None

try:
    from tslearn.metrics import dtw_path
except ImportError:
    dtw_path = None

from io_utils import load_gaze_positions


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
    import matplotlib.pyplot as plt

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
    import matplotlib.pyplot as plt
    import seaborn as sns

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


def run_scan_patterns(data_root, output_dir):
    """Extract scan patterns from gaze_positions.csv."""
    gaze_path = os.path.join(data_root, "gaze_positions.csv")
    gaze_df = load_gaze_positions(gaze_path)
    if gaze_df is None:
        print("Skipping scan pattern extraction (gaze_positions.csv not found).")
        return

    print("\n=== EXTRACTING SCAN PATTERNS (matrix profile) ===")
    gaze_df = preprocess_gaze_positions(gaze_df)
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

    if pattern_results is None:
        print("Pattern extraction skipped.")
        return

    pattern_results['patterns_df'].to_csv(os.path.join(output_dir, 'scan_patterns.csv'), index=False)
    plot_pattern_grid(pattern_results['snippets_x'], pattern_results['snippets_y'], output_dir,
                      filename='scan_patterns.png', alpha_by_time=False)
    plot_pattern_grid(pattern_results['snippets_x'], pattern_results['snippets_y'], output_dir,
                      filename='scan_patterns_fade.png', alpha_by_time=True)
    plot_pattern_time_series(tx, ty, rec_time_s, pattern_results['mask'], output_dir,
                             filename='scan_patterns_time_series.png')

    snippet_xavg, snippet_yavg = compute_dtw_averages(tx, ty, rec_time_s, pattern_results['mask'])
    if snippet_xavg:
        np.save(os.path.join(output_dir, 'scan_patterns_dtw_xavg.npy'), np.array(snippet_xavg, dtype=object))
        np.save(os.path.join(output_dir, 'scan_patterns_dtw_yavg.npy'), np.array(snippet_yavg, dtype=object))
