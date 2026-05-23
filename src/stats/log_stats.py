from omegaconf import OmegaConf
import os
import pprint

def log_stats(logs_raw: str, logs_repeats: str):
    # Get all files in the logs_raw directory
    raw_files_dir = os.listdir(logs_raw)
    repeats_files_dir = os.listdir(logs_repeats)

    assert len(raw_files_dir) == len(repeats_files_dir), "The number of raw log dirs and repeats log dirs should be the same."

    print(f"Number of raw log dirs: {len(raw_files_dir)}")
    print(f"Number of repeats log dirs: {len(repeats_files_dir)}")
    
    stats = dict()
    
    raw_files_path = [(os.path.join(logs_raw, f), f) for f in raw_files_dir]
    repeats_files_path = [(os.path.join(logs_repeats, f), f) for f in repeats_files_dir]
    
    for (raw_path, task), (repeats_path, _) in zip(raw_files_path, repeats_files_path):
        stats[task] = (len(os.listdir(raw_path)), len(os.listdir(repeats_path)))

    return stats

if __name__ == "__main__":
    config = OmegaConf.load("actions_config.yaml")
    pprint.pprint(log_stats(
        config.actions.stats.log_path_raw,
        config.actions.stats.log_path_repeats
    ))