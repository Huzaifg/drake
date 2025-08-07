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
from utils import get_data, plot_hydroelastic_query_perf_speedup_vs_num_elements_clutter, plot_hydroelastic_query_perf_speedup_clutter, plot_broad_phase_perf_speedup_clutter, plot_broad_phase_perf_speedup_vs_num_elements_clutter, plot_narrow_phase_query_perf_speedup_clutter, plot_advance_to_query_perf_speedup_vs_num_elements_clutter, plot_advance_to_query_perf_speedup_vs_obp_clutter



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
    demo_name = "clutter"
    objects_per_pile = ["1", "2", "5", "10", "20", "33", "50"]
    sphere_resolutions = ["0.0050", "0.0100", "0.0200", "0.0400"]
    
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
            for obp in objects_per_pile:
                if obp not in cpu_data[folder_name]:
                    cpu_data[folder_name][obp] = {}
                for sr in sphere_resolutions:
                    if sr not in cpu_data[folder_name][obp]:
                        cpu_data[folder_name][obp][sr] = {}
                    # Timing overall
                    json_path_timing_overall = f"{base_dir}/{folder_name}/{demo_name}_{obp}_1.0000_{sr}_3_drake-cpu_timing_overall.json"
                    data_timing_overall = get_data(json_path_timing_overall)
                    cpu_data[folder_name][obp][sr]["timing_overall"] = data_timing_overall
                    
                    json_path_problem_size = f"{base_dir}/{folder_name}/{demo_name}_{obp}_1.0000_{sr}_3_drake-cpu_problem_size.json"
                    data_problem_size = get_data(json_path_problem_size)
                    cpu_data[folder_name][obp][sr]["problem_size"] = data_problem_size
                    
                    json_path_advance_to = f"{base_dir}/{folder_name}/{demo_name}_{obp}_1.0000_{sr}_3_drake-cpu_timing_advance_to.json"
                    data_advance_to = get_data(json_path_advance_to)
                    cpu_data[folder_name][obp][sr]["advance_to"] = data_advance_to
        else:
            # GPU data
            if folder_name not in gpu_data:
                gpu_data[folder_name] = {}
            for obp in objects_per_pile:
                if obp not in gpu_data[folder_name]:
                    gpu_data[folder_name][obp] = {}
                for sr in sphere_resolutions:
                    if sr not in gpu_data[folder_name][obp]:
                        gpu_data[folder_name][obp][sr] = {}
                    # Timing overall
                    json_path_timing_overall = f"{base_dir}/{folder_name}/{demo_name}_{obp}_1.0000_{sr}_3_sycl-gpu_timing_overall.json"
                    data_timing_overall = get_data(json_path_timing_overall)
                    gpu_data[folder_name][obp][sr]["timing_overall"] = data_timing_overall
                    # Kernel timing
                    json_path_kernel_timing = f"{base_dir}/{folder_name}/{demo_name}_{obp}_1.0000_{sr}_3_sycl-gpu_timing.json"
                    data_kernel_timing = get_data(json_path_kernel_timing)
                    gpu_data[folder_name][obp][sr]["kernel_timing"] = data_kernel_timing
                    
                    json_path_problem_size = f"{base_dir}/{folder_name}/{demo_name}_{obp}_1.0000_{sr}_3_sycl-gpu_problem_size.json"
                    data_problem_size = get_data(json_path_problem_size)
                    gpu_data[folder_name][obp][sr]["problem_size"] = data_problem_size
                    
                    json_path_advance_to = f"{base_dir}/{folder_name}/{demo_name}_{obp}_1.0000_{sr}_3_sycl-gpu_timing_advance_to.json"
                    data_advance_to = get_data(json_path_advance_to)
                    gpu_data[folder_name][obp][sr]["advance_to"] = data_advance_to
                    
    # Create plots directory
    plot_dir = "plots_gpu_comparison_clutter_opt1"
    if not os.path.exists(f"{base_dir}/{plot_dir}"):
        os.makedirs(f"{base_dir}/{plot_dir}")
    

    # Plot broad phase timing vs actual number of objects (log-log)
    # fig, axes = plot_hydroelastic_query_perf_speedup_clutter(gpu_data, cpu_data, folder_names, legend_names, objects_per_pile, sphere_resolutions)
    # plt.savefig(f"{base_dir}/{plot_dir}/hydroelastic_query_perf_speedup.png", dpi=600)
    # print(f"Saved hydroelastic query perf speedup plot to {base_dir}/{plot_dir}/hydroelastic_query_perf_speedup.png")
    # plt.show()
    # plt.close()
    
    # fig, axes = plot_hydroelastic_query_perf_speedup_vs_num_elements_clutter(gpu_data, cpu_data, folder_names, legend_names, objects_per_pile, sphere_resolutions)
    # plt.savefig(f"{base_dir}/{plot_dir}/hydroelastic_query_perf_speedup_vs_num_elements.png", dpi=600)
    # print(f"Saved hydroelastic query perf speedup vs num elements plot to {base_dir}/{plot_dir}/hydroelastic_query_perf_speedup_vs_num_elements.png")
    # plt.show()
    # plt.close()
    
    # fig, axes = plot_broad_phase_perf_speedup_clutter(cpu_data, gpu_data, folder_names, legend_names, objects_per_pile, sphere_resolutions)
    # plt.savefig(f"{base_dir}/{plot_dir}/broad_phase_query_perf_speedup.png", dpi=600)
    # print(f"Saved broad phase query perf speedup plot to {base_dir}/{plot_dir}/broad_phase_query_perf_speedup.png")
    # plt.show()
    # plt.close()
    
    # fig, axes = plot_broad_phase_perf_speedup_vs_num_elements_clutter(cpu_data, gpu_data, folder_names, legend_names, objects_per_pile, sphere_resolutions)
    # plt.savefig(f"{base_dir}/{plot_dir}/broad_phase_query_perf_speedup_vs_num_elements.png", dpi=600)
    # print(f"Saved broad phase query perf speedup vs num elements plot to {base_dir}/{plot_dir}/broad_phase_query_perf_speedup_vs_num_elements.png")
    # plt.show()
    # plt.close()
    
    # fig, axes = plot_narrow_phase_query_perf_speedup_clutter(cpu_data, gpu_data, folder_names, legend_names, objects_per_pile, sphere_resolutions)
    # plt.savefig(f"{base_dir}/{plot_dir}/narrow_phase_query_perf_speedup.png", dpi=600)
    # print(f"Saved narrow phase query perf speedup plot to {base_dir}/{plot_dir}/narrow_phase_query_perf_speedup.png")
    # plt.show()
    # plt.close()
    
    fig, axes = plot_advance_to_query_perf_speedup_vs_num_elements_clutter(gpu_data, cpu_data, folder_names, legend_names, objects_per_pile, sphere_resolutions)
    plt.savefig(f"{base_dir}/{plot_dir}/advance_to_query_perf_speedup_vs_num_elements.png", dpi=600)
    print(f"Saved advance_to query perf speedup vs num elements plot to {base_dir}/{plot_dir}/advance_to_query_perf_speedup_vs_num_elements.png")
    plt.show()
    plt.close()
    
    fig, axes = plot_advance_to_query_perf_speedup_vs_obp_clutter(gpu_data, cpu_data, folder_names, legend_names, objects_per_pile, sphere_resolutions)
    plt.savefig(f"{base_dir}/{plot_dir}/advance_to_query_perf_speedup_vs_obp.png", dpi=600)
    print(f"Saved advance_to query perf speedup vs obp plot to {base_dir}/{plot_dir}/advance_to_query_perf_speedup_vs_obp.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()