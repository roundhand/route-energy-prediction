"""Generate small, entirely synthetic CSV inputs for end-to-end smoke tests."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def generate(output, trips_per_day=20):
    """Write the original CSV schema without any real vehicle or trajectory data."""
    rng = np.random.default_rng(42)
    output = Path(output)
    for day in pd.date_range("2023-09-11", "2023-09-15"):
        segments, routes = [], []
        for trip in range(trips_per_day):
            target = f"synthetic_vehicle_{trip % 4}"
            routes.append(
                dict(
                    target=target,
                    moveid_int=trip,
                    stime=str(day),
                    POI_time_sensitivity=["不敏感", "敏感", "高敏感"][trip % 3],
                    rush_hour=["平峰", "早高峰", "晚高峰"][trip % 3],
                )
            )
            for index in range(int(rng.integers(3, 8))):
                length = float(rng.uniform(80, 600))
                row = dict(
                    target=target,
                    moveid=trip,
                    entry_time=str(day + pd.Timedelta(minutes=trip * 10 + index)),
                    length=length,
                    energy_total=length * float(rng.uniform(0.0001, 0.0002)),
                    entry_angle=180.0,
                    exit_angle=180.0,
                    road_type=f"road_{(trip + index) % 7}",
                    entry_turn_type=[
                        "left_turn",
                        "right_turn",
                        "straight",
                        "trip_start",
                        "u_turn",
                        "unknown",
                    ][(trip + index) % 6],
                    exit_turn_type=[
                        "left_turn",
                        "right_turn",
                        "straight",
                        "trip_end",
                        "u_turn",
                        "unknown",
                    ][(trip + index) % 6],
                )
                for column in [
                    "traffic_speed_mean",
                    "traffic_speed_std",
                    "traffic_acceleration_mean",
                    "traffic_acceleration_std",
                    "traffic_strong_dec_ratio",
                    "traffic_dec_ratio",
                    "traffic_steady_ratio",
                    "traffic_acc_ratio",
                    "traffic_strong_acc_ratio",
                    "pseudo_acceleration",
                    "slope_mean",
                    "slope_std",
                ]:
                    row[column] = float(rng.uniform(0, 1))
                segments.append(row)
        folder = output / day.strftime("day_%Y-%m-%d")
        for subdirectory in ["segment_features_output", "features"]:
            (folder / subdirectory).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(segments).to_csv(
            folder / "segment_features_output/segment_features_clean_batch_0.csv", index=False
        )
        pd.DataFrame(routes).to_csv(folder / "features/poi_feature_0.csv", index=False)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/demo"))
    parser.add_argument("--trips-per-day", type=int, default=20)
    args = parser.parse_args(argv)
    if args.trips_per_day < 10:
        parser.error("trips-per-day must be at least 10 for valid train/validation/test splits")
    print(f"Synthetic data written to {generate(args.output, args.trips_per_day)}")


if __name__ == "__main__":
    main()
