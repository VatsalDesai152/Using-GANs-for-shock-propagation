import logging
import time
import subprocess
import sys
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_script(script_name):
    logging.info(f"--- Starting {script_name} ---")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name], 
            check=True,
            capture_output=True, 
            text=True
        )
        logging.info(f"Output of {script_name}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error occurred while running {script_name}")
        logging.error(f"Error output:\n{e.stderr}")
        sys.exit(1)
        
    duration = time.time() - start_time
    logging.info(f"--- {script_name} completed successfully in {duration:.2f} seconds ---")
    return result.stdout, duration

def main():
    logging.info("Starting T-GAT-AE Pipeline Orchestration")
    total_start_time = time.time()
    report = "# T-GAT-AE Final Orchestration Report\n\n"
    report += "This document provides a detailed breakdown of the end-to-end pipeline execution.\n\n"

    scripts = [
        "data_pipeline.py",
        "multi_seed_eval.py",
        "eval.py",
        "generate_latex.py",
        "generate_latex_aer.py"
    ]
    
    for script in scripts:
        report += f"## {script}\n"
        out, dur = run_script(script)
        report += f"**Execution Time:** {dur:.2f} seconds\n\n"
        report += "### Stdout Logs:\n"
        report += "```text\n"
        report += out
        report += "\n```\n\n"

    total_duration = time.time() - total_start_time
    logging.info(f"Pipeline Orchestration complete! Total time: {total_duration:.2f} seconds.")
    
    report += f"## Summary\n"
    report += f"**Total Orchestration Time:** {total_duration:.2f} seconds\n"
    report += "All scripts executed successfully.\n"
    
    with open("final_orchestration_report.md", "w") as f:
        f.write(report)
        
    logging.info("Wrote detailed breakdown to final_orchestration_report.md")

if __name__ == "__main__":
    main()
