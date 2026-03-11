import argparse
import os
from typing import List, Optional

from utils.flight_recorder import FlightDataRecorder


def _parse_int_list(spec: str) -> Optional[List[int]]:
    spec = (spec or "").strip()
    if not spec:
        return None
    values = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return values if values else None


def _parse_str_list(spec: str) -> Optional[List[str]]:
    spec = (spec or "").strip()
    if not spec:
        return None
    values = []
    for item in spec.split(","):
        item = item.strip()
        if item:
            values.append(item)
    return values if values else None


def _iter_selected_csv_paths(
    flight_recorder_dir: str,
    episodes: Optional[List[int]],
    env_indices: Optional[List[int]],
):
    if episodes is None:
        episode_dirs = []
        for name in os.listdir(flight_recorder_dir):
            abs_path = os.path.join(flight_recorder_dir, name)
            if os.path.isdir(abs_path) and name.isdigit():
                episode_dirs.append((int(name), abs_path))
        episode_dirs.sort(key=lambda x: x[0])
    else:
        episode_dirs = []
        for ep in episodes:
            abs_path = os.path.join(flight_recorder_dir, str(ep))
            if os.path.isdir(abs_path):
                episode_dirs.append((ep, abs_path))

    for _, episode_dir in episode_dirs:
        if env_indices is None:
            for filename in os.listdir(episode_dir):
                if filename.startswith("flight_trace_env") and filename.endswith(".csv"):
                    yield os.path.join(episode_dir, filename), episode_dir
        else:
            for env_index in env_indices:
                filename = f"flight_trace_env{env_index}.csv"
                csv_path = os.path.join(episode_dir, filename)
                if os.path.isfile(csv_path):
                    yield csv_path, episode_dir


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, default="", help="Path to a single flight_trace_env*.csv")
    parser.add_argument("--flight-recorder-dir", type=str, default="", help="Path to flight_recorder root directory.")
    parser.add_argument("--episodes", type=str, default="", help="Comma-separated episode numbers, e.g. '0,40,82'. Empty means all.")
    parser.add_argument("--env-indices", type=str, default="", help="Comma-separated env indices, e.g. '0,1,3'. Empty means all.")
    parser.add_argument("--plot-agent-ids", type=str, default="", help="Comma-separated agent ids for plotting, e.g. 'A0100,A0200'.")
    parser.add_argument("--out-dir", type=str, default="", help="Directory for plot outputs (single-csv mode only).")
    args = parser.parse_known_args(args)[0]

    plot_agent_ids = _parse_str_list(args.plot_agent_ids)

    # Backward-compatible single csv mode
    if args.csv_path:
        csv_path = args.csv_path
        out_dir = args.out_dir or os.path.dirname(csv_path)
        recorder = FlightDataRecorder(save_dir=out_dir, plot_agent_ids=plot_agent_ids)
        recorder.plot_csv(csv_path, out_dir=out_dir)
        return

    # Batch mode for flight_recorder/<episode>/flight_trace_env*.csv
    if not args.flight_recorder_dir:
        raise ValueError("Either --csv-path or --flight-recorder-dir must be provided.")
    if not os.path.isdir(args.flight_recorder_dir):
        raise FileNotFoundError(f"Directory does not exist: {args.flight_recorder_dir}")

    episodes = _parse_int_list(args.episodes)
    env_indices = _parse_int_list(args.env_indices)
    recorder = FlightDataRecorder(save_dir=args.flight_recorder_dir, plot_agent_ids=plot_agent_ids)

    found_any = False
    for csv_path, episode_dir in _iter_selected_csv_paths(args.flight_recorder_dir, episodes, env_indices):
        found_any = True
        recorder.plot_csv(csv_path, out_dir=episode_dir)

    if not found_any:
        raise FileNotFoundError(
            "No matching csv files found. Please check --episodes / --env-indices / --flight-recorder-dir."
        )


if __name__ == "__main__":
    main([
        "--flight-recorder-dir", "../scripts/results/MultipleCombat/4v4/ShootMissile/HierarchySelfplay/mappoMoE/128-128-128{2-6-2}/run-20260310-Attack2Avoid1/flight_recorder",
        "--episodes", "0,40,82",
        "--env-indices", "0,1",
        "--plot-agent-ids", "A0100,A0200",
    ])
