import argparse
import os

from utils.flight_recorder import FlightDataRecorder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, required=True, help="Path to flight_trace.csv")
    parser.add_argument("--out-dir", type=str, default="", help="Directory for plot outputs")
    args = parser.parse_args()

    csv_path = args.csv_path
    out_dir = args.out_dir or os.path.dirname(csv_path)
    recorder = FlightDataRecorder(save_dir=out_dir)
    recorder.plot_csv(csv_path, out_dir=out_dir)


if __name__ == "__main__":
    main()
