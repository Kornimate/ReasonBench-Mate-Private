import zipfile
import os
import datetime
from omegaconf import OmegaConf

def save_logs(log_dir: str, save_path: str):
    with zipfile.ZipFile(save_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, log_dir))
                
def main():
    config = OmegaConf.load("actions_config.yaml")
    LOG_DIRECTORY = config.actions.save.log_path
    SAVE_PATH = config.actions.save.save_path
    FILE_NAME = f"logs_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.zip"
    save_logs(
        LOG_DIRECTORY,
        os.path.join(SAVE_PATH, FILE_NAME)
    )

if __name__ == "__main__":
    main()