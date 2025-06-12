from importlib.metadata import version


def main():
    package_version = version("nortrip-python")
    print(f"Starting NORTRIP_model_python_v{package_version}...")
