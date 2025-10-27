import os
import shutil

# Get the parent directory (project root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print(f"Project root: {project_root}")

# Check what exists
print("\n=== Checking file structure ===")

# Check for models
models_dir = os.path.join(project_root, 'models')
print(f"Models directory exists: {os.path.exists(models_dir)}")
if os.path.exists(models_dir):
    print(f"  Files: {os.listdir(models_dir)}")

# Check for reports
reports_dir = os.path.join(project_root, 'reports')
print(f"Reports directory exists: {os.path.exists(reports_dir)}")
if os.path.exists(reports_dir):
    print(f"  Files: {os.listdir(reports_dir)}")

# Check for data
data_dir = os.path.join(project_root, 'data', 'processed')
print(f"Data directory exists: {os.path.exists(data_dir)}")
if os.path.exists(data_dir):
    print(f"  Files: {os.listdir(data_dir)}")