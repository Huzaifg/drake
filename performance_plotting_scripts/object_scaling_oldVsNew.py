#!/usr/bin/env python3
"""
GPU Performance Comparison Script

This script compares two GPU versions by plotting broad phase and narrow phase timing 
vs number of objects. It creates both line plots and bar plots with two lines/bars - one for each GPU version.

Usage:
    python object_scaling_oldVsNew.py <folder1> <legend1> <folder2> <legend2>

Example:
    python object_scaling_oldVsNew.py performance_jsons_bvh_1s "GPU Version 1" performance_jsons_bvh_2s "GPU Version 2"

The script expects the data to be formatted identically for both GPU versions.
"""

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import argparse
from utils import get_data

def get_corrected_timing(timing_data, run_type):
    """
    Get corrected timing for sycl-cpu by excluding JIT compilation time.
    For sycl-cpu: (total_us - max_us) / (calls - 1)
    For others: use avg_us
    """
    if run_type == "sycl-cpu":
        total_us = timing_data.get("total_us", 0)
        max_us = timing_data.get("max_us", 0)
        calls = timing_data.get("calls", 1)
        if calls > 1:
            return (total_us - max_us) / (calls - 1)
        else:
            return timing_data.get("avg_us", 0)
    else:
        return timing_data.get("avg_us", 0)

def plot_broad_phase_timing_vs_num_objects(all_data, folder_names, legend_names, spacings, num_gpp):
    """
    Plots broad phase timing vs number of objects for two GPU versions.
    Args:
        all_data: Nested dict containing data for both GPU versions
        folder_names: List of two folder names
        legend_names: List of two legend names
        spacings: List of spacings (e.g., ["0.1", "0.05"])
        num_gpp: List of num_gpp values (e.g., ["1", "2", "5", "10", "20"])
    """
    import pandas as pd
    
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    
    # Create subplots for each spacing
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    # Prepare data for each spacing
    for i, spacing in enumerate(spacings):
        plot_data = []
        
        for folder_idx, folder_name in enumerate(folder_names):
            for gpp in num_gpp:
                # Get the broad phase timing data
                kernel_timing = all_data[folder_name][spacing][gpp]["kernel_timing"].get("kernel_timings", {})
                broad_data = kernel_timing.get("transform_and_broad_phase", {})
                broad_time = get_corrected_timing(broad_data, "sycl-gpu")
                
                plot_data.append({
                    "NumObjects": int(gpp),
                    "BroadPhaseTime": broad_time,
                    "GPUVersion": legend_names[folder_idx]
                })
        
        # Create DataFrame and plot
        df = pd.DataFrame(plot_data)
        
        # Plot on the current subplot
        current_ax = axes[i]
        sns.lineplot(data=df, x="NumObjects", y="BroadPhaseTime", hue="GPUVersion", 
                    marker="o", ax=current_ax)
        
        # Customize the subplot
        current_ax.set_xticks([int(gpp) for gpp in num_gpp])
        current_ax.set_xlabel("Number of Objects per Group", fontsize=14)
        current_ax.set_ylabel("Broad Phase Time (us)" if i == 0 else "", fontsize=14)
        title = "Sparse" if spacing == "0.1" else "Dense"
        current_ax.set_title(title, fontsize=15, fontweight='bold')
        
        # Only show legend on the first subplot
        if i == 0:
            current_ax.legend(fontsize=12, title_fontsize=13)
        else:
            current_ax.get_legend().remove() if current_ax.get_legend() else None
            
        current_ax.tick_params(axis='both', which='major', labelsize=12)
        current_ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes

def plot_broad_phase_timing_vs_num_objects_bar(all_data, folder_names, legend_names, spacings, num_gpp):
    """
    Plots broad phase timing vs number of objects for two GPU versions as bar plots.
    Args:
        all_data: Nested dict containing data for both GPU versions
        folder_names: List of two folder names
        legend_names: List of two legend names
        spacings: List of spacings (e.g., ["0.1", "0.05"])
        num_gpp: List of num_gpp values (e.g., ["1", "2", "5", "10", "20"])
    """
    import pandas as pd
    import numpy as np
    
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    
    # Create subplots for each spacing
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    # Prepare data for each spacing
    for i, spacing in enumerate(spacings):
        plot_data = []
        
        for folder_idx, folder_name in enumerate(folder_names):
            for gpp in num_gpp:
                # Get the broad phase timing data
                kernel_timing = all_data[folder_name][spacing][gpp]["kernel_timing"].get("kernel_timings", {})
                broad_data = kernel_timing.get("transform_and_broad_phase", {})
                broad_time = get_corrected_timing(broad_data, "sycl-gpu")
                
                plot_data.append({
                    "NumObjects": int(gpp),
                    "BroadPhaseTime": broad_time,
                    "GPUVersion": legend_names[folder_idx]
                })
        
        # Create DataFrame and plot
        df = pd.DataFrame(plot_data)
        
        # Plot on the current subplot
        current_ax = axes[i]
        
        # Create grouped bar plot
        x_pos = np.arange(len(num_gpp))
        width = 0.35
        
        for folder_idx, legend_name in enumerate(legend_names):
            folder_data = df[df["GPUVersion"] == legend_name]
            values = [folder_data[folder_data["NumObjects"] == int(gpp)]["BroadPhaseTime"].iloc[0] 
                     if len(folder_data[folder_data["NumObjects"] == int(gpp)]) > 0 else 0 
                     for gpp in num_gpp]
            
            current_ax.bar(x_pos + folder_idx * width, values, width, 
                          label=legend_name, alpha=0.8)
        
        # Customize the subplot
        current_ax.set_xticks(x_pos + width / 2)
        current_ax.set_xticklabels([int(gpp) for gpp in num_gpp])
        current_ax.set_xlabel("Number of Objects per Group", fontsize=14)
        current_ax.set_ylabel("Broad Phase Time (us)" if i == 0 else "", fontsize=14)
        title = "Sparse" if spacing == "0.1" else "Dense"
        current_ax.set_title(title, fontsize=15, fontweight='bold')
        
        # Only show legend on the first subplot
        if i == 0:
            current_ax.legend(fontsize=12, title_fontsize=13)
        else:
            current_ax.get_legend().remove() if current_ax.get_legend() else None
            
        current_ax.tick_params(axis='both', which='major', labelsize=12)
        current_ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes

def plot_narrow_phase_timing_vs_num_objects(all_data, folder_names, legend_names, spacings, num_gpp):
    """
    Plots narrow phase timing vs number of objects for two GPU versions.
    Args:
        all_data: Nested dict containing data for both GPU versions
        folder_names: List of two folder names
        legend_names: List of two legend names
        spacings: List of spacings (e.g., ["0.1", "0.05"])
        num_gpp: List of num_gpp values (e.g., ["1", "2", "5", "10", "20"])
    """
    import pandas as pd
    
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    
    # Create subplots for each spacing
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    # Prepare data for each spacing
    for i, spacing in enumerate(spacings):
        plot_data = []
        
        for folder_idx, folder_name in enumerate(folder_names):
            for gpp in num_gpp:
                # Get the narrow phase timing data
                kernel_timing = all_data[folder_name][spacing][gpp]["kernel_timing"].get("kernel_timings", {})
                narrow_data = kernel_timing.get("compute_contact_polygons", {})
                narrow_time = get_corrected_timing(narrow_data, "sycl-gpu")
                
                plot_data.append({
                    "NumObjects": int(gpp),
                    "NarrowPhaseTime": narrow_time,
                    "GPUVersion": legend_names[folder_idx]
                })
        
        # Create DataFrame and plot
        df = pd.DataFrame(plot_data)
        
        # Plot on the current subplot
        current_ax = axes[i]
        sns.lineplot(data=df, x="NumObjects", y="NarrowPhaseTime", hue="GPUVersion", 
                    marker="o", ax=current_ax)
        
        # Customize the subplot
        current_ax.set_xticks([int(gpp) for gpp in num_gpp])
        current_ax.set_xlabel("Number of Objects per Group", fontsize=14)
        current_ax.set_ylabel("Narrow Phase Time (us)" if i == 0 else "", fontsize=14)
        title = "Sparse" if spacing == "0.1" else "Dense"
        current_ax.set_title(title, fontsize=15, fontweight='bold')
        
        # Only show legend on the first subplot
        if i == 0:
            current_ax.legend(fontsize=12, title_fontsize=13)
        else:
            current_ax.get_legend().remove() if current_ax.get_legend() else None
            
        current_ax.tick_params(axis='both', which='major', labelsize=12)
        current_ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes

def plot_narrow_phase_timing_vs_num_objects_bar(all_data, folder_names, legend_names, spacings, num_gpp):
    """
    Plots narrow phase timing vs number of objects for two GPU versions as bar plots.
    Args:
        all_data: Nested dict containing data for both GPU versions
        folder_names: List of two folder names
        legend_names: List of two legend names
        spacings: List of spacings (e.g., ["0.1", "0.05"])
        num_gpp: List of num_gpp values (e.g., ["1", "2", "5", "10", "20"])
    """
    import pandas as pd
    import numpy as np
    
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    
    # Create subplots for each spacing
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    # Prepare data for each spacing
    for i, spacing in enumerate(spacings):
        plot_data = []
        
        for folder_idx, folder_name in enumerate(folder_names):
            for gpp in num_gpp:
                # Get the narrow phase timing data
                kernel_timing = all_data[folder_name][spacing][gpp]["kernel_timing"].get("kernel_timings", {})
                narrow_data = kernel_timing.get("compute_contact_polygons", {})
                narrow_time = get_corrected_timing(narrow_data, "sycl-gpu")
                
                plot_data.append({
                    "NumObjects": int(gpp),
                    "NarrowPhaseTime": narrow_time,
                    "GPUVersion": legend_names[folder_idx]
                })
        
        # Create DataFrame and plot
        df = pd.DataFrame(plot_data)
        
        # Plot on the current subplot
        current_ax = axes[i]
        
        # Create grouped bar plot
        x_pos = np.arange(len(num_gpp))
        width = 0.35
        
        for folder_idx, legend_name in enumerate(legend_names):
            folder_data = df[df["GPUVersion"] == legend_name]
            values = [folder_data[folder_data["NumObjects"] == int(gpp)]["NarrowPhaseTime"].iloc[0] 
                     if len(folder_data[folder_data["NumObjects"] == int(gpp)]) > 0 else 0 
                     for gpp in num_gpp]
            
            current_ax.bar(x_pos + folder_idx * width, values, width, 
                          label=legend_name, alpha=0.8)
        
        # Customize the subplot
        current_ax.set_xticks(x_pos + width / 2)
        current_ax.set_xticklabels([int(gpp) for gpp in num_gpp])
        current_ax.set_xlabel("Number of Objects per Group", fontsize=14)
        current_ax.set_ylabel("Narrow Phase Time (us)" if i == 0 else "", fontsize=14)
        title = "Sparse" if spacing == "0.1" else "Dense"
        current_ax.set_title(title, fontsize=15, fontweight='bold')
        
        # Only show legend on the first subplot
        if i == 0:
            current_ax.legend(fontsize=12, title_fontsize=13)
        else:
            current_ax.get_legend().remove() if current_ax.get_legend() else None
            
        current_ax.tick_params(axis='both', which='major', labelsize=12)
        current_ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compare two GPU versions with line plots and bar plots')
    parser.add_argument('folder1', help='Name of the first performance data folder')
    parser.add_argument('legend1', help='Legend name for the first GPU version')
    parser.add_argument('folder2', help='Name of the second performance data folder')
    parser.add_argument('legend2', help='Legend name for the second GPU version')
    
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.getcwd())
    demo_name = "objects_scaling"
    spacings = ["0.1", "0.05"]
    num_gpp = ["1", "2", "5", "10", "20"]
    
    folder_names = [args.folder1, args.folder2]
    legend_names = [args.legend1, args.legend2]
    
    # Store all data in a nested dictionary: all_data[folder_name][spacing][num_gpp][data_type]
    all_data = {folder_name: {} for folder_name in folder_names}
    
    for folder_name in folder_names:
        for spacing in spacings:
            all_data[folder_name][spacing] = {}
            for gpp in num_gpp:
                all_data[folder_name][spacing][gpp] = {}
                
                # Problem size
                json_path_problem_size = f"{base_dir}/{folder_name}/{demo_name}_{spacing}_{gpp}_sycl-gpu_problem_size.json"
                data_problem_size = get_data(json_path_problem_size)
                all_data[folder_name][spacing][gpp]["problem_size"] = data_problem_size
                
                # Timing overall
                json_path_timing_overall = f"{base_dir}/{folder_name}/{demo_name}_{spacing}_{gpp}_sycl-gpu_timing_overall.json"
                data_timing_overall = get_data(json_path_timing_overall)
                all_data[folder_name][spacing][gpp]["timing_overall"] = data_timing_overall
                
                # Kernel timing
                json_path_kernel_timing = f"{base_dir}/{folder_name}/{demo_name}_{spacing}_{gpp}_sycl-gpu_timing.json"
                data_kernel_timing = get_data(json_path_kernel_timing)
                all_data[folder_name][spacing][gpp]["kernel_timing"] = data_kernel_timing
    
    # Create plots directory
    plot_dir = "plots_gpu_comparison"
    if not os.path.exists(f"{base_dir}/{plot_dir}"):
        os.makedirs(f"{base_dir}/{plot_dir}")
    
    # Plot broad phase timing vs number of objects (line plots)
    fig, axes = plot_broad_phase_timing_vs_num_objects(all_data, folder_names, legend_names, spacings, num_gpp)
    plt.savefig(f"{base_dir}/{plot_dir}/broad_phase_timing_vs_num_objects_line.png", dpi=600)
    print(f"Saved broad phase line plot to {base_dir}/{plot_dir}/broad_phase_timing_vs_num_objects_line.png")
    plt.show()
    plt.close()
    
    # Plot broad phase timing vs number of objects (bar plots)
    fig, axes = plot_broad_phase_timing_vs_num_objects_bar(all_data, folder_names, legend_names, spacings, num_gpp)
    plt.savefig(f"{base_dir}/{plot_dir}/broad_phase_timing_vs_num_objects_bar.png", dpi=600)
    print(f"Saved broad phase bar plot to {base_dir}/{plot_dir}/broad_phase_timing_vs_num_objects_bar.png")
    plt.show()
    plt.close()
    
    # Plot narrow phase timing vs number of objects (line plots)
    fig, axes = plot_narrow_phase_timing_vs_num_objects(all_data, folder_names, legend_names, spacings, num_gpp)
    plt.savefig(f"{base_dir}/{plot_dir}/narrow_phase_timing_vs_num_objects_line.png", dpi=600)
    print(f"Saved narrow phase line plot to {base_dir}/{plot_dir}/narrow_phase_timing_vs_num_objects_line.png")
    plt.show()
    plt.close()
    
    # Plot narrow phase timing vs number of objects (bar plots)
    fig, axes = plot_narrow_phase_timing_vs_num_objects_bar(all_data, folder_names, legend_names, spacings, num_gpp)
    plt.savefig(f"{base_dir}/{plot_dir}/narrow_phase_timing_vs_num_objects_bar.png", dpi=600)
    print(f"Saved narrow phase bar plot to {base_dir}/{plot_dir}/narrow_phase_timing_vs_num_objects_bar.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()