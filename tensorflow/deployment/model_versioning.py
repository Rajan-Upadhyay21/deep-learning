import os
from datetime import datetime


def create_versioned_model_path(base_dir="model_registry"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_path = os.path.join(base_dir, f"version_{timestamp}")
    os.makedirs(version_path, exist_ok=True)
    return version_path


def main():
    version_path = create_versioned_model_path()
    print("Created model version directory:", version_path)
    print("Use this directory to save exported models by version.")


if __name__ == "__main__":
    main()
