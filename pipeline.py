# run_pipeline.py
import subprocess
import sys

steps = [
    "get_data.py",
    "transform_data.py",
    "model_prep.py",
    "simulations.py",
    "latest_trades.py",
    "advanced_simul.py"
]

for step in steps:
    print(f"\n--- Running {step} ---")
    result = subprocess.run([
        sys.executable, "-u", step
    ], capture_output=True, text=True)
    #result = subprocess.run(["python", step], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors/Warnings:", result.stderr)
