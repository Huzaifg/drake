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
import numpy as np
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

def calculate_actual_objects(gpp):
    """
    Calculate the actual number of objects based on GPP (grippers per pepper).
    Each gripper has 2 bodies, each pepper has 1 body, and there's 1 table.
    Formula: 2 * gpp + gpp + 1 = 3 * gpp + 1
    """
    return 3 * int(gpp) + 1

def plot_broad_phase_timing_log_log(cpu_data, gpu_data, folder_names, legend_names, spacings, num_gpp):
    """
    Plots broad phase timing vs actual number of objects on a log-log scale for two GPU versions.
    Includes asymptotic complexity lines (O(n^2), O(nlogn), O(n), O(logn)).
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
            legend_lower = legend_names[folder_idx].lower()
            for gpp in num_gpp:
                # Get the broad phase timing data
                if(legend_lower == "cpu"):
                    broad_data = cpu_data[folder_name][spacing][gpp]["timing_overall"].get("timings", {}).get("BroadPhase", {})
                    broad_time = get_corrected_timing(broad_data, "drake-cpu")
                else:
                    kernel_timing = gpu_data[folder_name][spacing][gpp]["kernel_timing"].get("kernel_timings", {})
                    broad_data = kernel_timing.get("transform_and_broad_phase", {})
                    broad_time = get_corrected_timing(broad_data, "sycl-gpu")
                                    
                # Calculate actual number of objects
                actual_objects = calculate_actual_objects(gpp)
                
                plot_data.append({
                    "ActualObjects": actual_objects,
                    "BroadPhaseTime": broad_time,
                    "GPUVersion": legend_names[folder_idx]
                })
        
        # Create DataFrame and plot
        df = pd.DataFrame(plot_data)
        
        # Plot on the current subplot
        current_ax = axes[i]
        
        # Create log-log plot
        for gpu_version in legend_names:
            version_data = df[df["GPUVersion"] == gpu_version]
            current_ax.loglog(version_data["ActualObjects"], version_data["BroadPhaseTime"], 
                            marker="o", label=gpu_version, linewidth=2, markersize=6)
        
        # Get the actual data points to align asymptotic lines
        all_x = df["ActualObjects"].values
        all_y = df["BroadPhaseTime"].values
        
        # Find the minimum x and y values from actual data
        x_min_data = np.min(all_x)
        y_min_data = np.min(all_y)
        
        # Create x values for the asymptotic lines (same range as data)
        x_asymptotic = np.logspace(np.log10(x_min_data), np.log10(np.max(all_x)), 100)
        
        # Calculate scale factors to align with the data starting point
        # We want all asymptotic lines to start at the same y-value as the data
        scale_factors = {
            'O(n²)': y_min_data / (x_min_data**2),
            'O(n log n)': y_min_data / (x_min_data * np.log(x_min_data)),
            'O(n)': y_min_data / x_min_data,
            'O(log n)': y_min_data / np.log(x_min_data)
        }
        
        # Plot asymptotic lines with different colors and styles
        current_ax.loglog(x_asymptotic, scale_factors['O(n²)'] * x_asymptotic**2, 
                         '--', color='black', alpha=0.9, linewidth=3, label='O(n²)')
        current_ax.loglog(x_asymptotic, scale_factors['O(n log n)'] * x_asymptotic * np.log(x_asymptotic), 
                         '--', color='gray', alpha=0.9, linewidth=3, label='O(n log n)')
        current_ax.loglog(x_asymptotic, scale_factors['O(n)'] * x_asymptotic, 
                         '--', color='yellow', alpha=0.9, linewidth=3, label='O(n)')
        
        # Set more x-axis divisions
        current_ax.set_xscale('log')
        current_ax.xaxis.set_major_locator(plt.LogLocator(base=10, numticks=10))
        current_ax.xaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10), numticks=10))
        current_ax.grid(True, alpha=0.3, which='both')
        
        # Customize the subplot
        current_ax.set_xlabel("Number of Bodies", fontsize=14)
        current_ax.set_ylabel("Broad Phase Time (us)" if i == 0 else "", fontsize=14)
        title = "Sparse" if spacing == "0.1" else "Dense"
        current_ax.set_title(title, fontsize=15, fontweight='bold')
        
        # Only show legend on the first subplot
        if i == 0:
            current_ax.legend(fontsize=11, title_fontsize=12, loc='upper left', 
                           bbox_to_anchor=(0.02, 0.98), framealpha=0.9)
        else:
            current_ax.get_legend().remove() if current_ax.get_legend() else None
            
        current_ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    return fig, axes

def plot_broad_phase_timing_vs_num_objects(cpu_data, gpu_data, folder_names, legend_names, spacings, num_gpp):
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
            legend_lower = legend_names[folder_idx].lower()
            for gpp in num_gpp:
                # Get the broad phase timing data
                if(legend_lower == "cpu"):
                    broad_data = cpu_data[folder_name][spacing][gpp]["timing_overall"].get("timings", {}).get("BroadPhase", {})
                    broad_time = get_corrected_timing(broad_data, "drake-cpu")
                else:
                    kernel_timing = gpu_data[folder_name][spacing][gpp]["kernel_timing"].get("kernel_timings", {})
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

def plot_narrow_phase_timing_log_log(cpu_data, gpu_data, folder_names, legend_names, spacings, num_gpp):
    """
    Plots narrow phase timing vs actual number of objects on a log-log scale for two GPU versions.
    Includes asymptotic complexity lines (O(n^2), O(nlogn), O(n), O(logn)).
    """
    import pandas as pd

    sns.set_style("ticks")
    sns.set_palette("colorblind")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for i, spacing in enumerate(spacings):
        plot_data = []
        for folder_idx, folder_name in enumerate(folder_names):
            legend_lower = legend_names[folder_idx].lower()
            for gpp in num_gpp:
                if legend_lower == "cpu":
                    # CPU data
                    narrow_data = cpu_data[folder_name][spacing][gpp]["timing_overall"].get("timings", {}).get("NarrowPhase", {})
                    narrow_time = get_corrected_timing(narrow_data, "drake-cpu")
                else:
                    # GPU data
                    kernel_timing = gpu_data[folder_name][spacing][gpp]["kernel_timing"].get("kernel_timings", {})
                    narrow_data = kernel_timing.get("compute_contact_polygons", {})
                    narrow_time = get_corrected_timing(narrow_data, "sycl-gpu")
                actual_objects = calculate_actual_objects(gpp)
                plot_data.append({
                    "ActualObjects": actual_objects,
                    "NarrowPhaseTime": narrow_time,
                    "GPUVersion": legend_names[folder_idx]
                })

        df = pd.DataFrame(plot_data)
        current_ax = axes[i]
        for gpu_version in legend_names:
            version_data = df[df["GPUVersion"] == gpu_version]
            current_ax.loglog(version_data["ActualObjects"], version_data["NarrowPhaseTime"],
                             marker="o", label=gpu_version, linewidth=2, markersize=6)

        all_x = df["ActualObjects"].values
        all_y = df["NarrowPhaseTime"].values
        x_min_data = np.min(all_x)
        y_min_data = np.min(all_y)
        x_asymptotic = np.logspace(np.log10(x_min_data), np.log10(np.max(all_x)), 100)
        scale_factors = {
            'O(n²)': y_min_data / (x_min_data**2),
            'O(n log n)': y_min_data / (x_min_data * np.log(x_min_data)),
            'O(n)': y_min_data / x_min_data,
            'O(log n)': y_min_data / np.log(x_min_data)
        }
        current_ax.loglog(x_asymptotic, scale_factors['O(n²)'] * x_asymptotic**2, '--', color='black', alpha=0.9, linewidth=3, label='O(n²)')
        current_ax.loglog(x_asymptotic, scale_factors['O(n log n)'] * x_asymptotic * np.log(x_asymptotic), '--', color='gray', alpha=0.9, linewidth=3, label='O(n log n)')
        current_ax.loglog(x_asymptotic, scale_factors['O(n)'] * x_asymptotic, '--', color='yellow', alpha=0.9, linewidth=3, label='O(n)')

        current_ax.set_xscale('log')
        current_ax.xaxis.set_major_locator(plt.LogLocator(base=10, numticks=10))
        current_ax.xaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10), numticks=10))
        current_ax.grid(True, alpha=0.3, which='both')
        current_ax.set_xlabel("Number of Bodies", fontsize=14)
        current_ax.set_ylabel("Narrow Phase Time (us)" if i == 0 else "", fontsize=14)
        title = "Sparse" if spacing == "0.1" else "Dense"
        current_ax.set_title(title, fontsize=15, fontweight='bold')
        if i == 0:
            current_ax.legend(fontsize=11, title_fontsize=12, loc='upper left', bbox_to_anchor=(0.02, 0.98), framealpha=0.9)
        else:
            current_ax.get_legend().remove() if current_ax.get_legend() else None
        current_ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    return fig, axes

def plot_hydroelastic_query_log_log(gpu_data, cpu_data, folder_names, legend_names, spacings, num_gpp):
    """
    Plots total hydroelastic contact time (HydroelasticQuery) vs actual number of objects on a log-log scale for two GPU versions.
    Includes asymptotic complexity lines (O(n^2), O(nlogn), O(n), O(logn)).
    """
    import pandas as pd

    sns.set_style("ticks")
    sns.set_palette("colorblind")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for i, spacing in enumerate(spacings):
        plot_data = []
        for folder_idx, legend in enumerate(legend_names):
            for gpp in num_gpp:
                if "cpu" in legend.lower():
                    # CPU data
                    folder_name = folder_names[folder_idx]
                    timing_overall = cpu_data[folder_name][spacing][gpp]["timing_overall"].get("timings", {})
                    hq_data = timing_overall.get("HydroelasticQuery", {})
                    hq_time = get_corrected_timing(hq_data, "drake-cpu")
                else:
                    # GPU data
                    folder_name = folder_names[folder_idx]
                    timing_overall = gpu_data[folder_name][spacing][gpp]["timing_overall"].get("timings", {})
                    hq_data = timing_overall.get("HydroelasticQuery", {})
                    hq_time = get_corrected_timing(hq_data, "sycl-gpu")
                actual_objects = calculate_actual_objects(gpp)
                
                plot_data.append({
                    "ActualObjects": actual_objects,
                    "HydroelasticQueryTime": hq_time,
                    "GPUVersion": legend
                })

        df = pd.DataFrame(plot_data)
        current_ax = axes[i]
        for gpu_version in legend_names:
            version_data = df[df["GPUVersion"] == gpu_version]
            current_ax.loglog(version_data["ActualObjects"], version_data["HydroelasticQueryTime"],
                             marker="o", label=gpu_version, linewidth=2, markersize=6)

        all_x = df["ActualObjects"].values
        all_y = df["HydroelasticQueryTime"].values
        x_min_data = np.min(all_x)
        y_min_data = np.min(all_y)
        x_asymptotic = np.logspace(np.log10(x_min_data), np.log10(np.max(all_x)), 100)
        scale_factors = {
            'O(n²)': y_min_data / (x_min_data**2),
            'O(n log n)': y_min_data / (x_min_data * np.log(x_min_data)),
            'O(n)': y_min_data / x_min_data,
            'O(log n)': y_min_data / np.log(x_min_data)
        }
        current_ax.loglog(x_asymptotic, scale_factors['O(n²)'] * x_asymptotic**2, '--', color='black', alpha=0.9, linewidth=3, label='O(n²)')
        current_ax.loglog(x_asymptotic, scale_factors['O(n log n)'] * x_asymptotic * np.log(x_asymptotic), '--', color='gray', alpha=0.9, linewidth=3, label='O(n log n)')
        current_ax.loglog(x_asymptotic, scale_factors['O(n)'] * x_asymptotic, '--', color='yellow', alpha=0.9, linewidth=3, label='O(n)')

        current_ax.set_xscale('log')
        current_ax.xaxis.set_major_locator(plt.LogLocator(base=10, numticks=10))
        current_ax.xaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10), numticks=10))
        current_ax.grid(True, alpha=0.3, which='both')
        current_ax.set_xlabel("Number of Bodies", fontsize=14)
        current_ax.set_ylabel("HydroelasticQuery Time (us)" if i == 0 else "", fontsize=14)
        title = "Sparse" if spacing == "0.1" else "Dense"
        current_ax.set_title(title, fontsize=15, fontweight='bold')
        if i == 0:
            current_ax.legend(fontsize=11, title_fontsize=12, loc='upper left', bbox_to_anchor=(0.02, 0.98), framealpha=0.9)
        else:
            current_ax.get_legend().remove() if current_ax.get_legend() else None
        current_ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    return fig, axes

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compare two GPU versions with line plots and bar plots')
    parser.add_argument('folder1', help='Name of the first performance data folder')
    parser.add_argument('legend1', help='Legend name for the first GPU version')
    parser.add_argument('folder2', help='Name of the second performance data folder')
    parser.add_argument('legend2', help='Legend name for the second GPU version')
    # parser.add_argument('folder3', help='Name of the third performance data folder (CPU)')
    # parser.add_argument('legend3', help='Legend name for the CPU version')
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.getcwd())
    demo_name = "objects_scaling"
    spacings = ["0.1", "0.05"]
    num_gpp = ["1", "2", "5", "10", "20", "33"]
    
        # folder_names = [args.folder1, args.folder2, args.folder3]
        # legend_names = [args.legend1, args.legend2, args.legend3]
    folder_names = [args.folder1, args.folder2]
    legend_names = [args.legend1, args.legend2]
    # Store all data in a nested dictionary: all_data[folder_name][spacing][num_gpp][data_type]
    gpu_data = {folder_names[0]: {}}  # Only first folder is GPU
    cpu_data = {folder_names[1]: {}}  # Only second folder is CPU
    for i, folder_name in enumerate(folder_names):
        for spacing in spacings:
            for gpp in num_gpp:
                if "cpu" in legend_names[i].lower():
                    if spacing not in cpu_data[folder_name]:
                        cpu_data[folder_name][spacing] = {}
                    if gpp not in cpu_data[folder_name][spacing]:
                        cpu_data[folder_name][spacing][gpp] = {}
                    # Timing overall
                    json_path_timing_overall = f"{base_dir}/{folder_name}/{demo_name}_{spacing}_{gpp}_drake-cpu_timing_overall.json"
                    data_timing_overall = get_data(json_path_timing_overall)
                    cpu_data[folder_name][spacing][gpp]["timing_overall"] = data_timing_overall
                else:
                    if spacing not in gpu_data[folder_name]:
                        gpu_data[folder_name][spacing] = {}
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
                    
    # Create plots directory
    plot_dir = "plots_gpu_comparison"
    if not os.path.exists(f"{base_dir}/{plot_dir}"):
        os.makedirs(f"{base_dir}/{plot_dir}")
    
    # # Plot broad phase timing vs number of objects (line plots)
    # fig, axes = plot_broad_phase_timing_vs_num_objects(all_data, folder_names, legend_names, spacings, num_gpp)
    # plt.savefig(f"{base_dir}/{plot_dir}/broad_phase_timing_vs_num_objects_line.png", dpi=600)
    # print(f"Saved broad phase line plot to {base_dir}/{plot_dir}/broad_phase_timing_vs_num_objects_line.png")
    # plt.show()
    # plt.close()
    
    # # Plot broad phase timing vs number of objects (bar plots)
    # fig, axes = plot_broad_phase_timing_vs_num_objects_bar(all_data, folder_names, legend_names, spacings, num_gpp)
    # plt.savefig(f"{base_dir}/{plot_dir}/broad_phase_timing_vs_num_objects_bar.png", dpi=600)
    # print(f"Saved broad phase bar plot to {base_dir}/{plot_dir}/broad_phase_timing_vs_num_objects_bar.png")
    # plt.show()
    # plt.close()
    
    # # Plot narrow phase timing vs number of objects (line plots)
    # fig, axes = plot_narrow_phase_timing_vs_num_objects(all_data, folder_names, legend_names, spacings, num_gpp)
    # plt.savefig(f"{base_dir}/{plot_dir}/narrow_phase_timing_vs_num_objects_line.png", dpi=600)
    # print(f"Saved narrow phase line plot to {base_dir}/{plot_dir}/narrow_phase_timing_vs_num_objects_line.png")
    # plt.show()
    # plt.close()
    
    # # Plot narrow phase timing vs number of objects (bar plots)
    # fig, axes = plot_narrow_phase_timing_vs_num_objects_bar(all_data, folder_names, legend_names, spacings, num_gpp)
    # plt.savefig(f"{base_dir}/{plot_dir}/narrow_phase_timing_vs_num_objects_bar.png", dpi=600)
    # print(f"Saved narrow phase bar plot to {base_dir}/{plot_dir}/narrow_phase_timing_vs_num_objects_bar.png")
    # plt.show()
    # plt.close()

    # Plot broad phase timing vs actual number of objects (log-log)
    fig, axes = plot_broad_phase_timing_log_log(cpu_data, gpu_data, folder_names, legend_names, spacings, num_gpp)
    plt.savefig(f"{base_dir}/{plot_dir}/broad_phase_timing_vs_actual_num_objects_log_log.png", dpi=600)
    print(f"Saved broad phase log-log plot to {base_dir}/{plot_dir}/broad_phase_timing_vs_actual_num_objects_log_log.png")
    plt.show()
    plt.close()

    # Plot narrow phase timing vs actual number of objects (log-log)
    fig, axes = plot_narrow_phase_timing_log_log(cpu_data, gpu_data, folder_names, legend_names, spacings, num_gpp)
    plt.savefig(f"{base_dir}/{plot_dir}/narrow_phase_timing_vs_actual_num_objects_log_log.png", dpi=600)
    print(f"Saved narrow phase log-log plot to {base_dir}/{plot_dir}/narrow_phase_timing_vs_actual_num_objects_log_log.png")
    plt.show()
    plt.close()

    # Plot hydroelastic query timing vs actual number of objects (log-log)
    fig, axes = plot_hydroelastic_query_log_log(gpu_data, cpu_data, folder_names, legend_names, spacings, num_gpp)
    plt.savefig(f"{base_dir}/{plot_dir}/hydroelastic_query_vs_actual_num_objects_log_log.png", dpi=600)
    print(f"Saved hydroelastic query log-log plot to {base_dir}/{plot_dir}/hydroelastic_query_vs_actual_num_objects_log_log.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()