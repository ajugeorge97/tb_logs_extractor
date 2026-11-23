
from pathlib import Path

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.util import tensor_util


def export_tensorboard_scalars(log_dir, output_dir="tb_export"):
    log_dir = Path(log_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted({
        path
        for path in log_dir.rglob("events.out.tfevents.*")
        if path.is_file()
    })

    all_records = []

    for run_dir in run_dirs:
        run_name = str(run_dir.relative_to(log_dir)) or "default"
        print(f"Processing: {run_name}")

        ea = event_accumulator.EventAccumulator(
            str(run_dir),
            size_guidance={
                event_accumulator.SCALARS: 0,
                event_accumulator.TENSORS: 0,
                event_accumulator.IMAGES: 1,
                event_accumulator.HISTOGRAMS: 1,
                event_accumulator.AUDIO: 1,
            },
        )
        ea.Reload()

        tags = ea.Tags()

        # Legacy scalar summaries
        for tag in tags.get("scalars", []):
            for event in ea.Scalars(tag):
                all_records.append({
                    "run": run_name,
                    "tag": tag,
                    "step": event.step,
                    "value": event.value,
                    "wall_time": event.wall_time,
                })

        # Modern tensor-backed scalar summaries
        for tag in tags.get("tensors", []):
            for event in ea.Tensors(tag):
                value = tensor_util.make_ndarray(event.tensor_proto)

                # Only export single numerical values
                if value.size != 1 or not np.issubdtype(
                    value.dtype, np.number
                ):
                    continue

                all_records.append({
                    "run": run_name,
                    "tag": tag,
                    "step": event.step,
                    "value": float(value.item()),
                    "wall_time": event.wall_time,
                })

    if not all_records:
        print("No scalar metrics found.")
        return None

    df = pd.DataFrame(all_records)
    df = df.sort_values(
        ["run", "tag", "step", "wall_time"]
    ).reset_index(drop=True)

    output_file = output_dir / "all_scalars.csv"
    df.to_csv(output_file, index=False)

    print(f"Exported {len(df):,} scalar records")
    print(f"Saved to: {output_file}")

    print("\nAvailable metrics:")
    print(df.groupby(["run", "tag"]).size().to_string())

    return df


# Change this to your TensorBoard log directory


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    parser.add_argument("--out_path", default="./tb_export")
    
    args = parser.parse_args()
    
    df = export_tensorboard_scalars(
    log_dir=args.path,
    output_dir=args.out_path,
)
