from omegaconf import OmegaConf
import os
import shutil

def logs_copy(logs_src: str, logs_dst: str):
    # Get all files in the logs_raw directory
    src_files_dir = os.listdir(logs_src)
    
    src_files_path = [(os.path.join(logs_src, f), f) for f in src_files_dir]
    
    for (src_path, task) in src_files_path:
        dst_path = os.path.join(logs_dst, task)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path, exist_ok=True)
        for file in os.listdir(src_path):
            src_file_path = os.path.join(src_path, file)
            dst_file_path = os.path.join(dst_path, file)
            # if not os.path.exists(dst_file_path):
            shutil.copy2(src_file_path, dst_file_path)

    return True

if __name__ == "__main__":
    config = OmegaConf.load("actions_config.yaml")
    print("Successful copy: ",logs_copy(
        # logs_src="D:\\AU\Thesis\\Logs\\logs_2026-06-08_13-07-29_log_used_in_report_latest\\raw_calls\\repeats\\gpt-4.1-nano",
        logs_src="D:\\AU\\Thesis\\Logs\\logs_2026-06-11_18-22-09_v2_run_results_final\\raw_calls\\repeats\\gpt-4.1-nano",
        logs_dst=config.actions.stats.log_path_raw,
    ))
    print("Successful copy: ",logs_copy(
        # logs_src="D:\\AU\Thesis\\Logs\\logs_2026-06-08_13-07-29_log_used_in_report_latest\\repeats\\gpt-4.1-nano",
        logs_src="D:\\AU\\Thesis\\Logs\\logs_2026-06-11_18-22-09_v2_run_results_final\\repeats\\gpt-4.1-nano",
        logs_dst=config.actions.stats.log_path_repeats,
    ))