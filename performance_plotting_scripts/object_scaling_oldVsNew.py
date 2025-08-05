#!/usr/bin/env python3
"""
GPU Performance Comparison Script

This script compares multiple GPU/CPU versions by plotting broad phase and narrow phase timing 
vs number of objects. It creates both line plots and bar plots with multiple lines/bars - one for each version.

Usage:
    python object_scaling_oldVsNew.py --folders <folder1> <folder2> ... --legends <legend1> <legend2> ...

Example:
    python object_scaling_oldVsNew.py --folders performance_jsons_bvh_1s performance_jsons_bvh_2s performance_jsons_cpu --legends "GPU Version 1" "GPU Version 2" "CPU Version"

The script expects the data to be formatted identically for all versions.
For CPU versions, the legend name must contain "cpu" (case insensitive).
"""

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import argparse
import numpy as np
from utils import get_data, calculate_actual_objects, get_corrected_timing, plot_broad_phase_timing_log_log_multiple, plot_hydroelastic_query_log_log_multiple, plot_hydroelastic_query_perf_speedup, plot_broad_phase_perf_speedup, plot_narrow_phase_query_perf_speedup, plot_broad_phase_perf_speedup_vs_num_elements, plot_hydroelastic_query_perf_speedup_vs_num_elements



def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compare multiple GPU/CPU versions with line plots and bar plots')
    parser.add_argument('--folders', nargs='+', required=True, help='Names of the performance data folders')
    parser.add_argument('--legends', nargs='+', required=True, help='Legend names for each version')
    args = parser.parse_args()
    
    # Validate that we have the same number of folders and legends
    if len(args.folders) != len(args.legends):
        print("Error: Number of folders must match number of legends")
        sys.exit(1)
    
    base_dir = os.path.dirname(os.getcwd())
    demo_name = "objects_scaling"
    spacings = ["0.1", "0.05"]
    num_gpp = ["1", "2", "5", "10", "20", "33", "50", "100", "200"]
    
    folder_names = args.folders
    legend_names = args.legends
    
    # Store all data in a nested dictionary: all_data[folder_name][spacing][num_gpp][data_type]
    gpu_data = {}
    cpu_data = {}
    
    for i, folder_name in enumerate(folder_names):
        legend_lower = legend_names[i].lower()
        if "cpu" in legend_lower:
            # CPU data
            if folder_name not in cpu_data:
                cpu_data[folder_name] = {}
            for spacing in spacings:
                if spacing not in cpu_data[folder_name]:
                    cpu_data[folder_name][spacing] = {}
                for gpp in num_gpp:
                    if gpp not in cpu_data[folder_name][spacing]:
                        cpu_data[folder_name][spacing][gpp] = {}
                    # Timing overall
                    json_path_timing_overall = f"{base_dir}/{folder_name}/{demo_name}_{spacing}_{gpp}_drake-cpu_timing_overall.json"
                    data_timing_overall = get_data(json_path_timing_overall)
                    cpu_data[folder_name][spacing][gpp]["timing_overall"] = data_timing_overall
                    
                    json_path_problem_size = f"{base_dir}/{folder_name}/{demo_name}_{spacing}_{gpp}_drake-cpu_problem_size.json"
                    data_problem_size = get_data(json_path_problem_size)
                    cpu_data[folder_name][spacing][gpp]["problem_size"] = data_problem_size
        else:
            # GPU data
            if folder_name not in gpu_data:
                gpu_data[folder_name] = {}
            for spacing in spacings:
                if spacing not in gpu_data[folder_name]:
                    gpu_data[folder_name][spacing] = {}
                for gpp in num_gpp:
                    if gpp not in gpu_data[folder_name][spacing]:
                        gpu_data[folder_name][spacing][gpp] = {}
                    # Timing overall
                    json_path_timing_overall = f"{base_dir}/{folder_name}/{demo_name}_{spacing}_{gpp}_sycl-gpu_timing_overall.json"
                    data_timing_overall = get_data(json_path_timing_overall)
                    gpu_data[folder_name][spacing][gpp]["timing_overall"] = data_timing_overall
                    # Kernel timing
                    json_path_kernel_timing = f"{base_dir}/{folder_name}/{demo_name}_{spacing}_{gpp}_sycl-gpu_timing.json"
                    data_kernel_timing = get_data(json_path_kernel_timing)
                    gpu_data[folder_name][spacing][gpp]["kernel_timing"] = data_kernel_timing
                    
                    json_path_problem_size = f"{base_dir}/{folder_name}/{demo_name}_{spacing}_{gpp}_sycl-gpu_problem_size.json"
                    data_problem_size = get_data(json_path_problem_size)
                    gpu_data[folder_name][spacing][gpp]["problem_size"] = data_problem_size
                    
    # Create plots directory
    plot_dir = "plots_gpu_comparison"
    if not os.path.exists(f"{base_dir}/{plot_dir}"):
        os.makedirs(f"{base_dir}/{plot_dir}")
    

    # Plot broad phase timing vs actual number of objects (log-log)
    fig, axes = plot_hydroelastic_query_perf_speedup(gpu_data, cpu_data, folder_names, legend_names, spacings, num_gpp)
    plt.savefig(f"{base_dir}/{plot_dir}/hydroelastic_query_perf_speedup.png", dpi=600)
    print(f"Saved hydroelastic query perf speedup plot to {base_dir}/{plot_dir}/hydroelastic_query_perf_speedup.png")
    plt.show()
    plt.close()
    
    fig, axes = plot_hydroelastic_query_perf_speedup_vs_num_elements(gpu_data, cpu_data, folder_names, legend_names, spacings, num_gpp)
    plt.savefig(f"{base_dir}/{plot_dir}/hydroelastic_query_perf_speedup_vs_num_elements.png", dpi=600)
    print(f"Saved hydroelastic query perf speedup vs num elements plot to {base_dir}/{plot_dir}/hydroelastic_query_perf_speedup_vs_num_elements.png")
    plt.show()
    plt.close()
    
    fig, axes = plot_broad_phase_perf_speedup(cpu_data, gpu_data, folder_names, legend_names, spacings, num_gpp)
    plt.savefig(f"{base_dir}/{plot_dir}/broad_phase_query_perf_speedup.png", dpi=600)
    print(f"Saved broad phase query perf speedup plot to {base_dir}/{plot_dir}/broad_phase_query_perf_speedup.png")
    plt.show()
    plt.close()
    
    fig, axes = plot_broad_phase_perf_speedup_vs_num_elements(cpu_data, gpu_data, folder_names, legend_names, spacings, num_gpp)
    plt.savefig(f"{base_dir}/{plot_dir}/broad_phase_query_perf_speedup_vs_num_elements.png", dpi=600)
    print(f"Saved broad phase query perf speedup vs num elements plot to {base_dir}/{plot_dir}/broad_phase_query_perf_speedup_vs_num_elements.png")
    plt.show()
    plt.close()
    
    fig, axes = plot_narrow_phase_query_perf_speedup(cpu_data, gpu_data, folder_names, legend_names, spacings, num_gpp)
    plt.savefig(f"{base_dir}/{plot_dir}/narrow_phase_query_perf_speedup.png", dpi=600)
    print(f"Saved narrow phase query perf speedup plot to {base_dir}/{plot_dir}/narrow_phase_query_perf_speedup.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()