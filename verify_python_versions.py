#!/usr/bin/env python3
import tomllib
from pathlib import Path


def check_python_versions():
    """Verify all pyproject.toml files have consistent Python versions."""
    repo_root = Path("/home/ubuntu/repos/cua")
    pyproject_files = list(repo_root.rglob("pyproject.toml"))
    
    print("Python version requirements in all pyproject.toml files:")
    print("=" * 60)
    
    for file_path in sorted(pyproject_files):
        try:
            with open(file_path, "rb") as f:
                data = tomllib.load(f)
            
            requires_python = data.get("project", {}).get("requires-python", "Not specified")
            relative_path = file_path.relative_to(repo_root)
            print(f"{relative_path}: {requires_python}")
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    print("\nSpecific files I modified:")
    print("=" * 30)
    modified_files = [
        "libs/python/computer-server/pyproject.toml",
        "libs/python/pylume/pyproject.toml"
    ]
    
    for file_name in modified_files:
        file_path = repo_root / file_name
        try:
            with open(file_path, "rb") as f:
                data = tomllib.load(f)
            requires_python = data.get("project", {}).get("requires-python", "Not specified")
            print(f"{file_name}: {requires_python}")
        except Exception as e:
            print(f"Error reading {file_name}: {e}")

if __name__ == "__main__":
    check_python_versions()
