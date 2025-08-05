import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import json
"""
Relevant data:
- problem_size
For sycl-gpu and sycl-cpu:
  SYCFacesInserted : Number of faces after narrow phase
  SYCLCandidateTets : Number of candidate tetrahedra after broad phase
For drake-cpu:
  FacesInserted : Number of faces after narrow phase
  CandidateTets : Number of candidate tetrahedra after broad phase
  
  - Timing Overall
For sycl-gpu and sycl-cpu:
  HydroelasticQuery - avg_us : Avg time taken for full hydroelastic query
For drake-cpu:
  HydroelasticQuery - avg_us : Avg time taken for full hydroelastic query
  BroadPhase - avg_us : Avg time taken for broad phase
  NarrowPhase - avg_us : Avg time taken for narrow phase
  
  - kernel_timing
For sycl-gpu and sycl-cpu:
  - transform_and_broad_phase - avg_us : Avg time taken for transform and broad phase
  - compute_contact_polygons - avg_us : Avg time taken for narrow phase  
"""
def get_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def calculate_actual_objects(gpp):
    """
    Calculate the actual number of objects based on GPP (grippers per pepper).
    Each gripper has 2 bodies, each pepper has 1 body, and there's 1 table.
    Formula: 2 * gpp + gpp + 1 = 3 * gpp + 1
    """
    return 3 * int(gpp) + 1

def calculate_number_of_elements_objects_scaling(gpp, problem_size_data):
    num_floors = 1
    num_peppers = int(gpp)
    num_grippers = int(gpp)
    
    # Initialize element counts
    floor_elements = 0
    pepper_elements = 0
    left_gripper_elements = 0
    right_gripper_elements = 0
    
    # Get hydroelastic_bodies array
    hydroelastic_bodies = problem_size_data.get("hydroelastic_bodies", [])
    
    # Process each body in the array
    for body_data in hydroelastic_bodies:
        body_name = body_data.get("body", "")
        tetrahedra = int(body_data.get("tetrahedra", 0))
        
        if body_name == "Floor":
            floor_elements = tetrahedra * num_floors
        elif body_name == "yellow_bell_pepper_no_stem":
            pepper_elements = tetrahedra * num_peppers
        elif body_name == "left_finger_bubble":
            left_gripper_elements = tetrahedra * num_grippers
        elif body_name == "right_finger_bubble":
            right_gripper_elements = tetrahedra * num_grippers
    
    gripper_elements = left_gripper_elements + right_gripper_elements
    return floor_elements + pepper_elements + gripper_elements

def _slope_indicator(ax, x0, y0, exponent, label,
                     length_dec=0.6,             # ← shorter than before
                     **line_kw):
    """
    Add a faint slope reference O(n^exponent) starting at (x0, y0).

    length_dec : how many powers of ten the arrow spans along x.
    """
    x1 = x0 * 10 ** length_dec
    y1 = y0 * (x1 / x0) ** exponent

    defaults = dict(ls="--", lw=1.0, color="0.4", alpha=0.6, zorder=1,
                    solid_capstyle="butt")
    defaults.update(line_kw)

    ax.plot([x0, x1], [y0, y1], **defaults)
    ax.annotate(label, xy=(x1, y1), xytext=(4, -2),
                textcoords="offset points", fontsize=8,
                ha="left", va="center", color=defaults["color"])


def _slope_indicator_nlogn(ax, x0, y0, label=r"$n\log n$",
                           length_dec=0.6, **line_kw):
    """
    Draw a short dashed curve showing the asymptotic shape of n log n
    on log–log axes.  The curve starts at (x0, y0) and spans
    `length_dec` decades in x.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x0, y0 : float      start point in data coordinates
    label : str         text to annotate at the curve end
    length_dec : float  horizontal length in decades
    **line_kw :         forwarded to ax.plot
    """
    x1 = x0 * 10 ** length_dec
    xs = np.logspace(np.log10(x0), np.log10(x1), 32)

    # Scale factor k such that k * x0 * log(x0) == y0
    k = y0 / (x0 * np.log(x0))
    ys = k * xs * np.log(xs)

    defaults = dict(ls="--", lw=1.0, color="0.4",
                    alpha=0.6, zorder=1)
    defaults.update(line_kw)

    ax.plot(xs, ys, **defaults)
    ax.annotate(label, (xs[-1], ys[-1]),
                xytext=(4, -2), textcoords="offset points",
                ha="left", va="center", fontsize=8,
                color=defaults["color"])

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

def plot_candidate_tets_vs_resolution(all_data, run_types, resolutions, ax=None):
    """
    Plots CandidateTets vs Resolution for given run_types using seaborn.
    Args:
        all_data: Nested dict as in spatula.py
        run_types: List of run types (e.g., ["sycl-gpu", "drake-cpu"])
        resolutions: List of resolution values
        ax: Optional matplotlib axis to plot on
    """
    import pandas as pd
    # Prepare data for plotting
    plot_data = []
    for run_type in run_types:
        for res in resolutions:
            # sycl-gpu: 'SYCLCandidateTets', drake-cpu: 'CandidateTets'
            problem_sizes = all_data[run_type][res]["problem_size"].get("problem_sizes", {})
            if run_type.startswith("sycl"):
                tets = problem_sizes.get("SYCLCandidateTets", {}).get("avg", None)
            else:
                tets = problem_sizes.get("CandidateTets", {}).get("avg", None)
            plot_data.append({
                "Resolution": res,
                "CandidateTets": tets,
                "RunType": run_type
            })
    df = pd.DataFrame(plot_data)
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    if ax is None:
        fig, ax = plt.subplots()
    sns.lineplot(data=df, x="Resolution", y="CandidateTets", hue="RunType", marker="o", ax=ax)
    ax.set_xticks(resolutions)
    ax.set_xlabel("Mesh Res. (mm)", fontsize=16)
    ax.set_ylabel("Candidates - Broad Phase", fontsize=16)
    ax.legend(fontsize=12, title_fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=13)
    plt.tight_layout()
    return ax 



def plot_faces_inserted_vs_resolution(all_data, run_types, resolutions, ax=None):
    """
    Plots FacesInserted vs Resolution for given run_types using seaborn.
    Args:
        all_data: Nested dict as in spatula.py
        run_types: List of run types (e.g., ["sycl-gpu", "drake-cpu"])
        resolutions: List of resolution values
        ax: Optional matplotlib axis to plot on
    """
    import pandas as pd
    # Prepare data for plotting
    plot_data = []
    for run_type in run_types:
        for res in resolutions:
            # sycl-gpu: 'SYCFacesInserted', drake-cpu: 'FacesInserted'
            problem_sizes = all_data[run_type][res]["problem_size"].get("problem_sizes", {})
            if run_type.startswith("sycl"):
                tets = problem_sizes.get("SYCFacesInserted", {}).get("avg", None)
            else:
                tets = problem_sizes.get("FacesInserted", {}).get("avg", None)
            plot_data.append({
                "Resolution": res,
                "FacesInserted": tets,
                "RunType": run_type
            })
    df = pd.DataFrame(plot_data)
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    if ax is None:
        fig, ax = plt.subplots()
    sns.lineplot(data=df, x="Resolution", y="FacesInserted", hue="RunType", marker="o", ax=ax)
    ax.set_xticks(resolutions)
    ax.set_xlabel("Mesh Res. (mm)", fontsize=16)
    ax.set_ylabel("Faces Inserted - Narrow Phase", fontsize=16)
    ax.legend(fontsize=12, title_fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=13)
    plt.tight_layout()
    return ax 

def plot_faces_inserted_vs_num_gpp(all_data, run_types, spacings, num_gpp, ax=None):
    """
    Plots FacesInserted vs NumGpp for given run_types using seaborn.
    Args:
        all_data: Nested dict as in spatula.py
        run_types: List of run types (e.g., ["sycl-gpu", "drake-cpu"])
        spacings: List of spacings (e.g., ["0.1", "0.05"])
        num_gpp: List of num_gpp values (e.g., ["1", "2", "5", "10", "20"])
        ax: Optional matplotlib axis to plot on
    """
    import pandas as pd
    
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    
    # Create subplots if ax is not provided
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    else:
        # If ax is provided, we assume it's a single axis, so we can't create subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    # Prepare data for each spacing
    for i, spacing in enumerate(spacings):
        plot_data = []
        for run_type in run_types:
            for gpp in num_gpp:
                # Get the faces inserted data
                problem_sizes = all_data[run_type][spacing][gpp]["problem_size"].get("problem_sizes", {})
                if run_type.startswith("sycl"):
                    faces = problem_sizes.get("SYCFacesInserted", {}).get("avg", None)
                else:
                    faces = problem_sizes.get("FacesInserted", {}).get("avg", None)
                
                plot_data.append({
                    "NumGpp": int(gpp),
                    "FacesInserted": faces,
                    "RunType": run_type
                })
        
        # Create DataFrame and plot
        df = pd.DataFrame(plot_data)
        
        # Plot on the current subplot
        current_ax = axes[i]
        sns.lineplot(data=df, x="NumGpp", y="FacesInserted", hue="RunType", 
                    marker="o", ax=current_ax)
        
        # Customize the subplot
        current_ax.set_xticks([int(gpp) for gpp in num_gpp])
        current_ax.set_xlabel("Number of Objects per Group", fontsize=14)
        current_ax.set_ylabel("Faces Inserted - Narrow Phase" if i == 0 else "", fontsize=14)
        # title = "Sparse" if spacing == "0.1" else "Dense"
        title = ""
        if(spacing == "0.1"):
            title = "Sparse - 0.1"
        elif(spacing == "0.15"):
            title = "Sparse - 0.15"
        elif(spacing == "0.05"):
            title = "Dense - 0.05"
        current_ax.set_title(title, fontsize=15, fontweight='bold')
        
        # Only show legend on the first subplot
        if i == 0:
            current_ax.legend(fontsize=12, title_fontsize=13)
        else:
            current_ax.get_legend().remove() if current_ax.get_legend() else None
            
        current_ax.tick_params(axis='both', which='major', labelsize=12)
        current_ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return axes

def plot_candidate_tets_vs_num_gpp(all_data, run_types, spacings, num_gpp, ax=None):
    """
    Plots CandidateTets vs NumGpp for given run_types using seaborn.
    Args:
        all_data: Nested dict as in spatula.py
        run_types: List of run types (e.g., ["sycl-gpu", "drake-cpu"])
        spacings: List of spacings (e.g., ["0.1", "0.05"])
        num_gpp: List of num_gpp values (e.g., ["1", "2", "5", "10", "20"])
        ax: Optional matplotlib axis to plot on
    """
    import pandas as pd
    
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    
    # Create subplots if ax is not provided
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    else:
        # If ax is provided, we assume it's a single axis, so we can't create subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    # Prepare data for each spacing
    for i, spacing in enumerate(spacings):
        plot_data = []
        for run_type in run_types:
            for gpp in num_gpp:
                # Get the candidate tets data
                problem_sizes = all_data[run_type][spacing][gpp]["problem_size"].get("problem_sizes", {})
                if run_type.startswith("sycl"):
                    tets = problem_sizes.get("SYCLCandidateTets", {}).get("avg", None)
                else:
                    tets = problem_sizes.get("CandidateTets", {}).get("avg", None)
                
                plot_data.append({
                    "NumGpp": int(gpp),
                    "CandidateTets": tets,
                    "RunType": run_type
                })
        
        # Create DataFrame and plot
        df = pd.DataFrame(plot_data)
        
        # Plot on the current subplot
        current_ax = axes[i]
        sns.lineplot(data=df, x="NumGpp", y="CandidateTets", hue="RunType", 
                    marker="o", ax=current_ax)
        
        # Customize the subplot
        current_ax.set_xticks([int(gpp) for gpp in num_gpp])
        current_ax.set_xlabel("Number of Objects per Group", fontsize=14)
        current_ax.set_ylabel("Candidates - Broad Phase" if i == 0 else "", fontsize=14)
        # title = "Sparse" if spacing == "0.1" else "Dense"
        title = ""
        if(spacing == "0.1"):
            title = "Sparse - 0.1"
        elif(spacing == "0.15"):
            title = "Sparse - 0.15"
        elif(spacing == "0.05"):
            title = "Dense - 0.05"
        current_ax.set_title(title, fontsize=15, fontweight='bold')
        
        # Only show legend on the first subplot
        if i == 0:
            current_ax.legend(fontsize=12, title_fontsize=13)
        else:
            current_ax.get_legend().remove() if current_ax.get_legend() else None
            
        current_ax.tick_params(axis='both', which='major', labelsize=12)
        current_ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return axes

def plot_broad_narrow_misc_vs_num_gpp(all_data, run_types, spacings, num_gpp, ax=None):
    """
    Plots a grouped stacked bar plot comparing BroadPhase, NarrowPhase, and Misc times for sycl-gpu and drake-cpu.
    The full bar is HydroelasticQueryTime, with segments for BroadPhase, NarrowPhase, and Misc.
    Uses two subplots for different spacings.
    Args:
        all_data: Nested dict as in object_scaling.py
        run_types: List of run types (should include 'sycl-gpu' and 'drake-cpu')
        spacings: List of spacings (e.g., ["0.1", "0.05"])
        num_gpp: List of num_gpp values (e.g., ["1", "2", "5", "10", "20"])
        ax: Optional matplotlib axis to plot on
    """
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    
    # Create subplots if ax is not provided
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    else:
        # If ax is provided, we assume it's a single axis, so we can't create subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # Define colors for different timing components
    colors = {
        'BroadPhase': '#1f77b4',
        'NarrowPhase': '#ff7f0e',
        'Misc': '#2ca02c',
    }
    
    # Plot for each spacing
    for spacing_idx, spacing in enumerate(spacings):
        current_ax = axes[spacing_idx]
        
        bar_width = 0.35
        x = list(range(len(num_gpp)))
        run_type_offsets = {'sycl-gpu': -bar_width/2, 'drake-cpu': bar_width/2}
        
        for i, run_type in enumerate(['sycl-gpu', 'drake-cpu']):
            if run_type not in run_types:
                continue
                
            broad_vals, narrow_vals, misc_vals, total_vals = [], [], [], []
            
            for gpp in num_gpp:
                timings = all_data[run_type][spacing][gpp]["timing_overall"].get("timings", {})
                hq_data = timings.get("HydroelasticQuery", {})
                hq = get_corrected_timing(hq_data, run_type)
                
                if run_type == 'sycl-gpu':
                    kernel_timing = all_data[run_type][spacing][gpp]["kernel_timing"].get("kernel_timings", {})
                    broad_data = kernel_timing.get("transform_and_broad_phase", {})
                    narrow_data = kernel_timing.get("compute_contact_polygons", {})
                    broad = get_corrected_timing(broad_data, run_type)
                    narrow = get_corrected_timing(narrow_data, run_type)
                else:
                    broad_data = timings.get("BroadPhase", {})
                    narrow_data = timings.get("NarrowPhase", {})
                    broad = get_corrected_timing(broad_data, run_type)
                    narrow = get_corrected_timing(narrow_data, run_type)
                
                misc = max(hq - broad - narrow, 0)
                broad_vals.append(broad)
                narrow_vals.append(narrow)
                misc_vals.append(misc)
                total_vals.append(hq)
            # Plot stacked bars for this run_type
            xpos = [xi + run_type_offsets[run_type] for xi in x]
            
            # Create stacked bars
            b1 = current_ax.bar(xpos, broad_vals, bar_width, color=colors['BroadPhase'], 
                               label=f'BroadPhase' if spacing_idx == 0 and run_type == 'sycl-gpu' else None, 
                               hatch='//' if run_type == 'sycl-gpu' else None)
            
            b2 = current_ax.bar(xpos, narrow_vals, bar_width, bottom=broad_vals, color=colors['NarrowPhase'], 
                               label=f'NarrowPhase' if spacing_idx == 0 and run_type == 'sycl-gpu' else None, 
                               hatch='//' if run_type == 'sycl-gpu' else None)
            
            bottoms = [b + n for b, n in zip(broad_vals, narrow_vals)]
            b3 = current_ax.bar(xpos, misc_vals, bar_width, bottom=bottoms, color=colors['Misc'], 
                               label=f'Misc' if spacing_idx == 0 and run_type == 'sycl-gpu' else None, 
                               hatch='//' if run_type == 'sycl-gpu' else None)
        
        # Customize the subplot
        current_ax.set_xticks(x)
        current_ax.set_xticklabels(num_gpp)
        current_ax.set_xlabel("Number of Objects per Group", fontsize=14)
        current_ax.set_ylabel("Time (us)" if spacing_idx == 0 else "", fontsize=14)
        # title = "Sparse" if spacing == "0.1" else "Dense"
        title = ""
        if(spacing == "0.1"):
            title = "Sparse - 0.1"
        elif(spacing == "0.15"):
            title = "Sparse - 0.15"
        elif(spacing == "0.05"):
            title = "Dense - 0.05"
        current_ax.set_title(title, fontsize=15, fontweight='bold')
        
        # Add subtitle indicating bar grouping only on first subplot
        if spacing_idx == 0:
            current_ax.text(0.5, -0.15, "(left: sycl-gpu, right: drake-cpu)", 
                           transform=current_ax.transAxes, ha='center', fontsize=12)
        
        # Only show legend on the first subplot
        if spacing_idx == 0:
            handles, labels = current_ax.get_legend_handles_labels()
            seen = set()
            new_handles, new_labels = [], []
            for h, l in zip(handles, labels):
                if l not in seen and l:
                    new_handles.append(h)
                    new_labels.append(l)
                    seen.add(l)
            current_ax.legend(new_handles, new_labels, fontsize=12, title_fontsize=13)
        
        current_ax.tick_params(axis='both', which='major', labelsize=12)
        current_ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return axes


def plot_broad_narrow_misc_vs_num_elements(all_data, run_types, spacings, num_gpp, ax=None):
    """
    Plots a grouped stacked bar plot comparing BroadPhase, NarrowPhase, and Misc times for sycl-gpu and drake-cpu.
    The full bar is HydroelasticQueryTime, with segments for BroadPhase, NarrowPhase, and Misc.
    Uses two subplots for different spacings.
    Args:
        all_data: Nested dict as in object_scaling.py
        run_types: List of run types (should include 'sycl-gpu' and 'drake-cpu')
        spacings: List of spacings (e.g., ["0.1", "0.05"])
        num_gpp: List of num_gpp values (e.g., ["1", "2", "5", "10", "20"])
        ax: Optional matplotlib axis to plot on
    """
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    
    # Create subplots if ax is not provided
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    else:
        # If ax is provided, we assume it's a single axis, so we can't create subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    
    # Define colors for different timing components
    colors = {
        'BroadPhase': '#1f77b4',
        'NarrowPhase': '#ff7f0e',
        'Misc': '#2ca02c',
    }
    
    num_elements = [calculate_number_of_elements_objects_scaling(gpp, all_data[run_types[0]][spacings[0]][gpp]["problem_size"]) for gpp in num_gpp]
    
    # Plot for each spacing
    for spacing_idx, spacing in enumerate(spacings):
        current_ax = axes[spacing_idx]
        
        bar_width = 0.35
        x = list(range(len(num_elements)))
        run_type_offsets = {'sycl-gpu': -bar_width/2, 'drake-cpu': bar_width/2}
        
        for i, run_type in enumerate(['sycl-gpu', 'drake-cpu']):
            if run_type not in run_types:
                continue
                
            broad_vals, narrow_vals, misc_vals, total_vals = [], [], [], []
            
            for gpp in num_gpp:
                timings = all_data[run_type][spacing][gpp]["timing_overall"].get("timings", {})
                hq_data = timings.get("HydroelasticQuery", {})
                hq = get_corrected_timing(hq_data, run_type)
                
                if run_type == 'sycl-gpu':
                    kernel_timing = all_data[run_type][spacing][gpp]["kernel_timing"].get("kernel_timings", {})
                    broad_data = kernel_timing.get("transform_and_broad_phase", {})
                    narrow_data = kernel_timing.get("compute_contact_polygons", {})
                    broad = get_corrected_timing(broad_data, run_type)
                    narrow = get_corrected_timing(narrow_data, run_type)
                else:
                    broad_data = timings.get("BroadPhase", {})
                    narrow_data = timings.get("NarrowPhase", {})
                    broad = get_corrected_timing(broad_data, run_type)
                    narrow = get_corrected_timing(narrow_data, run_type)
                
                misc = max(hq - broad - narrow, 0)
                broad_vals.append(broad)
                narrow_vals.append(narrow)
                misc_vals.append(misc)
                total_vals.append(hq)
            # Plot stacked bars for this run_type
            xpos = [xi + run_type_offsets[run_type] for xi in x]
            
            # Create stacked bars
            b1 = current_ax.bar(xpos, broad_vals, bar_width, color=colors['BroadPhase'], 
                               label=f'BroadPhase' if spacing_idx == 0 and run_type == 'sycl-gpu' else None, 
                               hatch='//' if run_type == 'sycl-gpu' else None)
            
            b2 = current_ax.bar(xpos, narrow_vals, bar_width, bottom=broad_vals, color=colors['NarrowPhase'], 
                               label=f'NarrowPhase' if spacing_idx == 0 and run_type == 'sycl-gpu' else None, 
                               hatch='//' if run_type == 'sycl-gpu' else None)
            
            bottoms = [b + n for b, n in zip(broad_vals, narrow_vals)]
            b3 = current_ax.bar(xpos, misc_vals, bar_width, bottom=bottoms, color=colors['Misc'], 
                               label=f'Misc' if spacing_idx == 0 and run_type == 'sycl-gpu' else None, 
                               hatch='//' if run_type == 'sycl-gpu' else None)
        
        # Customize the subplot
        current_ax.set_xticks(x)
        
        # Format x-axis labels to be more readable
        def format_num_elements(num):
            if num >= 1000000:
                return f"{num/1000000:.1f}M"
            elif num >= 1000:
                return f"{num/1000:.0f}K"
            else:
                return str(num)
        
        formatted_labels = [format_num_elements(num) for num in num_elements]
        current_ax.set_xticklabels(formatted_labels, rotation=45, ha='right')
        current_ax.set_xlabel("Number of Elements", fontsize=14)
        current_ax.set_ylabel("Time (us)" if spacing_idx == 0 else "", fontsize=14)
        # title = "Sparse" if spacing == "0.1" else "Dense"
        title = ""
        if(spacing == "0.1"):
            title = "Sparse - 0.1"
        elif(spacing == "0.15"):
            title = "Sparse - 0.15"
        elif(spacing == "0.05"):
            title = "Dense - 0.05"
        current_ax.set_title(title, fontsize=15, fontweight='bold')
        
        # Add subtitle indicating bar grouping only on first subplot
        if spacing_idx == 0:
            current_ax.text(0.5, -0.15, "(left: sycl-gpu, right: drake-cpu)", 
                           transform=current_ax.transAxes, ha='center', fontsize=12)
        
        # Only show legend on the first subplot
        if spacing_idx == 0:
            handles, labels = current_ax.get_legend_handles_labels()
            seen = set()
            new_handles, new_labels = [], []
            for h, l in zip(handles, labels):
                if l not in seen and l:
                    new_handles.append(h)
                    new_labels.append(l)
                    seen.add(l)
            current_ax.legend(new_handles, new_labels, fontsize=12, title_fontsize=13)
        
        current_ax.tick_params(axis='both', which='major', labelsize=12)
        current_ax.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout(pad=2.0)
    return axes

def plot_narrow_phase_timing_vs_num_gpp(all_data, run_types, spacings, num_gpp, ax=None):
    """
    Plots NarrowPhase timing vs NumGpp for given run_types using seaborn line plots.
    Args:
        all_data: Nested dict as in object_scaling.py
        run_types: List of run types (e.g., ["sycl-gpu", "drake-cpu"])
        spacings: List of spacings (e.g., ["0.1", "0.05"])
        num_gpp: List of num_gpp values (e.g., ["1", "2", "5", "10", "20"])
        ax: Optional matplotlib axis to plot on
    """
    import pandas as pd
    
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    
    # Create subplots if ax is not provided
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    else:
        # If ax is provided, we assume it's a single axis, so we can't create subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    # Prepare data for each spacing
    for i, spacing in enumerate(spacings):
        plot_data = []
        for run_type in run_types:
            for gpp in num_gpp:
                # Get the narrow phase timing data
                if run_type == 'sycl-gpu':
                    kernel_timing = all_data[run_type][spacing][gpp]["kernel_timing"].get("kernel_timings", {})
                    narrow_data = kernel_timing.get("compute_contact_polygons", {})
                    narrow_time = get_corrected_timing(narrow_data, run_type)
                else:
                    timings = all_data[run_type][spacing][gpp]["timing_overall"].get("timings", {})
                    narrow_data = timings.get("NarrowPhase", {})
                    narrow_time = get_corrected_timing(narrow_data, run_type)
                
                plot_data.append({
                    "NumGpp": int(gpp),
                    "NarrowPhaseTime": narrow_time,
                    "RunType": run_type
                })
        
        # Create DataFrame and plot
        df = pd.DataFrame(plot_data)
        
        # Plot on the current subplot
        current_ax = axes[i]
        sns.lineplot(data=df, x="NumGpp", y="NarrowPhaseTime", hue="RunType", 
                    marker="o", ax=current_ax)
        
        # Customize the subplot
        current_ax.set_xticks([int(gpp) for gpp in num_gpp])
        current_ax.set_xlabel("Number of Objects per Group", fontsize=14)
        current_ax.set_ylabel("Narrow Phase Time (us)" if i == 0 else "", fontsize=14)
        # title = "Sparse" if spacing == "0.1" else "Dense"
        title = ""
        if(spacing == "0.1"):
            title = "Sparse - 0.1"
        elif(spacing == "0.15"):
            title = "Sparse - 0.15"
        elif(spacing == "0.05"):
            title = "Dense - 0.05"
        current_ax.set_title(title, fontsize=15, fontweight='bold')
        
        # Only show legend on the first subplot
        if i == 0:
            current_ax.legend(fontsize=12, title_fontsize=13)
        else:
            current_ax.get_legend().remove() if current_ax.get_legend() else None
            
        current_ax.tick_params(axis='both', which='major', labelsize=12)
        current_ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return axes

def plot_narrow_phase_timing_vs_candidate_tets(all_data, run_types, spacings, num_gpp, ax=None):
    """
    Plots NarrowPhase timing vs CandidateTets for given run_types using seaborn line plots.
    Shows the relationship between broad phase output and narrow phase performance.
    Args:
        all_data: Nested dict as in object_scaling.py
        run_types: List of run types (e.g., ["sycl-gpu", "drake-cpu"])
        spacings: List of spacings (e.g., ["0.1", "0.05"])
        num_gpp: List of num_gpp values (e.g., ["1", "2", "5", "10", "20"])
        ax: Optional matplotlib axis to plot on
    """
    import pandas as pd
    
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    
    # Create subplots if ax is not provided
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    else:
        # If ax is provided, we assume it's a single axis, so we can't create subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    # Prepare data for each spacing
    for i, spacing in enumerate(spacings):
        plot_data = []
        for run_type in run_types:
            for gpp in num_gpp:
                # Get the candidate tets data
                problem_sizes = all_data[run_type][spacing][gpp]["problem_size"].get("problem_sizes", {})
                if run_type.startswith("sycl"):
                    candidate_tets = problem_sizes.get("SYCLCandidateTets", {}).get("avg", None)
                else:
                    candidate_tets = problem_sizes.get("CandidateTets", {}).get("avg", None)
                
                # Get the narrow phase timing data
                if run_type == 'sycl-gpu':
                    kernel_timing = all_data[run_type][spacing][gpp]["kernel_timing"].get("kernel_timings", {})
                    narrow_data = kernel_timing.get("compute_contact_polygons", {})
                    narrow_time = get_corrected_timing(narrow_data, run_type)
                else:
                    timings = all_data[run_type][spacing][gpp]["timing_overall"].get("timings", {})
                    narrow_data = timings.get("NarrowPhase", {})
                    narrow_time = get_corrected_timing(narrow_data, run_type)
                
                plot_data.append({
                    "CandidateTets": candidate_tets,
                    "NarrowPhaseTime": narrow_time,
                    "RunType": run_type,
                    "NumGpp": int(gpp)  # Keep this for reference/debugging
                })
        
        # Create DataFrame and plot
        df = pd.DataFrame(plot_data)
        
        # Plot on the current subplot
        current_ax = axes[i]
        sns.lineplot(data=df, x="CandidateTets", y="NarrowPhaseTime", hue="RunType", 
                    marker="o", ax=current_ax)
        
        # Customize the subplot
        current_ax.set_xlabel("Candidate Tets - Broad Phase", fontsize=14)
        current_ax.set_ylabel("Narrow Phase Time (us)" if i == 0 else "", fontsize=14)
        # title = "Sparse" if spacing == "0.1" else "Dense"
        title = ""
        if(spacing == "0.1"):
            title = "Sparse - 0.1"
        elif(spacing == "0.15"):
            title = "Sparse - 0.15"
        elif(spacing == "0.05"):
            title = "Dense - 0.05"
        current_ax.set_title(title, fontsize=15, fontweight='bold')
        current_ax.set_xlim(0, 90000)  # Set consistent x-axis limits
        current_ax.xaxis.set_major_locator(plt.MultipleLocator(20000))
        current_ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
        # Only show legend on the first subplot
        if i == 0:
            current_ax.legend(fontsize=12, title_fontsize=13)
        else:
            current_ax.get_legend().remove() if current_ax.get_legend() else None
            
        current_ax.tick_params(axis='both', which='major', labelsize=12)
        current_ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return axes

def plot_timing_overall_vs_resolution(all_data, run_types, resolutions, ax=None):
    """
    Plots TimingOverall vs Resolution for given run_types using seaborn.
    Args:
        all_data: Nested dict as in spatula.py
        run_types: List of run types (e.g., ["sycl-gpu", "drake-cpu"])
        resolutions: List of resolution values
        ax: Optional matplotlib axis to plot on
    """
    import pandas as pd
    # Prepare data for plotting
    plot_data = []
    for run_type in run_types:
        for res in resolutions:
            timings = all_data[run_type][res]["timing_overall"].get("timings", {})
            # Use corrected timing for sycl-cpu, avg_us for others
            timing_data = timings.get("HydroelasticQuery", {})
            timing = get_corrected_timing(timing_data, run_type)
            plot_data.append({
                "Resolution": res,
                "TimingOverall": timing,
                "RunType": run_type
            })
    df = pd.DataFrame(plot_data)
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    if ax is None:
        fig, ax = plt.subplots()
    sns.lineplot(data=df, x="Resolution", y="TimingOverall", hue="RunType", marker="o", ax=ax)
    ax.set_xticks(resolutions)
    ax.set_xlabel("Mesh Res. (mm)", fontsize=16)
    ax.set_ylabel("HydroelasticQuery Time (us)", fontsize=16)
    ax.legend(fontsize=12, title_fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=13)
    plt.tight_layout()
    return ax


def plot_broad_narrow_misc_bar(all_data, run_types, resolutions, ax=None):
    """
    Plots a grouped stacked bar plot comparing BroadPhase, NarrowPhase, and Misc times for sycl-gpu and drake-cpu.
    The full bar is HydroelasticQueryTime, with segments for BroadPhase, NarrowPhase, and Misc.
    Only sycl-gpu and drake-cpu are compared.
    Uses correct timing keys for each run_type.
    Args:
        all_data: Nested dict as in spatula.py
        run_types: List of run types (should include 'sycl-gpu' and 'drake-cpu')
        resolutions: List of resolution values
        ax: Optional matplotlib axis to plot on
    """
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
    bar_width = 0.35
    x = list(range(len(resolutions)))
    colors = {
        'BroadPhase': '#1f77b4',
        'NarrowPhase': '#ff7f0e',
        'Misc': '#2ca02c',
    }
    run_type_offsets = {'sycl-gpu': -bar_width/2, 'drake-cpu': bar_width/2}
    legend_handles = {}
    for i, run_type in enumerate(['sycl-gpu', 'drake-cpu']):
        if run_type not in run_types:
            continue
        broad_vals, narrow_vals, misc_vals, total_vals = [], [], [], []
        for res in resolutions:
            timings = all_data[run_type][res]["timing_overall"].get("timings", {})
            hq_data = timings.get("HydroelasticQuery", {})
            hq = get_corrected_timing(hq_data, run_type)
            if run_type == 'sycl-gpu':
                kernel_timing = all_data[run_type][res]["kernel_timing"].get("kernel_timings", {})
                broad_data = kernel_timing.get("transform_and_broad_phase", {})
                narrow_data = kernel_timing.get("compute_contact_polygons", {})
                broad = get_corrected_timing(broad_data, run_type)
                narrow = get_corrected_timing(narrow_data, run_type)
            else:
                broad_data = timings.get("BroadPhase", {})
                narrow_data = timings.get("NarrowPhase", {})
                broad = get_corrected_timing(broad_data, run_type)
                narrow = get_corrected_timing(narrow_data, run_type)
            misc = max(hq - broad - narrow, 0)
            broad_vals.append(broad)
            narrow_vals.append(narrow)
            misc_vals.append(misc)
            total_vals.append(hq)
        xpos = [xi + run_type_offsets[run_type] for xi in x]
        b1 = ax.bar(xpos, broad_vals, bar_width, color=colors['BroadPhase'], label=f'BroadPhase' if run_type=='sycl-gpu' else None, hatch='//' if run_type=='sycl-gpu' else None)
        b2 = ax.bar(xpos, narrow_vals, bar_width, bottom=broad_vals, color=colors['NarrowPhase'], label=f'NarrowPhase' if run_type=='sycl-gpu' else None, hatch='//' if run_type=='sycl-gpu' else None)
        bottoms = [b+n for b, n in zip(broad_vals, narrow_vals)]
        b3 = ax.bar(xpos, misc_vals, bar_width, bottom=bottoms, color=colors['Misc'], label=f'Misc' if run_type=='sycl-gpu' else None, hatch='//' if run_type=='sycl-gpu' else None)
        
        

    ax.set_xticks(x)
    ax.set_xticklabels(resolutions)
    ax.set_xlabel("Mesh Res. (mm) \n(left: sycl-gpu, right: drake-cpu)", fontsize=16)
    ax.set_ylabel("Time (us)", fontsize=16)
    # Build legend (remove duplicate labels)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    new_handles, new_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen and l:
            new_handles.append(h)
            new_labels.append(l)
            seen.add(l)
    ax.legend(new_handles, new_labels, fontsize=12, title_fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=13)
    plt.tight_layout()
    return ax



def plot_broad_phase_timing_log_log_multiple(cpu_data, gpu_data, folder_names, legend_names, spacings, num_gpp):
    """
    Plots broad phase timing vs actual number of objects on a log-log scale for two GPU versions.
    Includes asymptotic complexity lines (O(n^2), O(nlogn), O(n), O(logn)).
    For CPU sims, legend name has to contain "cpu"
    Args:
        all_data: Nested dict containing data for both GPU versions
        folder_names: List of two folder names
        legend_names: List of two legend names
        spacings: List of spacings (e.g., ["0.1", "0.05"])
        num_gpp: List of num_gpp values (e.g., ["1", "2", "5", "10", "20"])
    """
    
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
                if("cpu" in legend_lower):
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
        current_ax.set_xlabel("Number of Geometries", fontsize=14)
        current_ax.set_ylabel("Broad Phase Time (us)" if i == 0 else "", fontsize=14)
        # title = "Sparse" if spacing == "0.1" else "Dense"
        title = ""
        if(spacing == "0.1"):
            title = "Sparse - 0.1"
        elif(spacing == "0.15"):
            title = "Sparse - 0.15"
        elif(spacing == "0.05"):
            title = "Dense - 0.05"
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


def plot_hydroelastic_query_log_log_multiple(gpu_data, cpu_data, folder_names, legend_names, spacings, num_gpp):
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
        current_ax.set_xlabel("Number of Geometries", fontsize=14)
        current_ax.set_ylabel("HydroelasticQuery Time (us)" if i == 0 else "", fontsize=14)
        # title = "Sparse" if spacing == "0.1" else "Dense"
        title = ""
        if(spacing == "0.1"):
            title = "Sparse - 0.1"
        elif(spacing == "0.15"):
            title = "Sparse - 0.15"
        elif(spacing == "0.05"):
            title = "Dense - 0.05"
        current_ax.set_title(title, fontsize=15, fontweight='bold')
        if i == 0:
            current_ax.legend(fontsize=11, title_fontsize=12, loc='upper left', bbox_to_anchor=(0.02, 0.98), framealpha=0.9)
        else:
            current_ax.get_legend().remove() if current_ax.get_legend() else None
        current_ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    return fig, axes

def plot_narrow_phase_query_perf_speedup(cpu_data, gpu_data,
                                         folder_names, legend_names,
                                         spacings, num_gpp):
    """
    Two‑row figure:
        • Row 0 – raw Narrow‑phase timings   (CPU + GPUs).
        • Row 1 – CPU / GPU speed‑up.

    CPU and GPU rows are matched on (Spacing, gpp) so the speed‑up
    column never turns into NaNs even when the “candidate‑tets”
    averages differ slightly.
    """
    # 0 ▸ visual defaults --------------------------------------------------
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    markers = ["o", "s", "D", "^", "v"]
    lstyles = ["-", "--", "-.", ":"]

    # 1 ▸ gather raw timings ----------------------------------------------
    rows_raw = []
    cpu_label = next(lbl for lbl in legend_names if "cpu" in lbl.lower())

    for f_idx, legend in enumerate(legend_names):
        store = cpu_data if legend == cpu_label else gpu_data
        for spacing in spacings:
            for gpp in num_gpp:
                if legend == cpu_label:                         # CPU
                    nph_dict = (store[folder_names[f_idx]][spacing][gpp]
                                ["timing_overall"].get("timings", {})
                                .get("NarrowPhase", {}))
                    nph_time = get_corrected_timing(nph_dict, "drake-cpu")
                    prob_sz  = (store[folder_names[f_idx]][spacing][gpp]
                                ["problem_size"].get("problem_sizes", {})
                                .get("CandidateTets", {}).get("avg", None))
                else:                                           # GPU
                    nph_dict = (store[folder_names[f_idx]][spacing][gpp]
                                ["kernel_timing"]
                                .get("kernel_timings", {})
                                .get("compute_contact_polygons", {}))
                    nph_time = get_corrected_timing(nph_dict, "sycl-gpu")
                    prob_sz  = (store[folder_names[f_idx]][spacing][gpp]
                                ["problem_size"].get("problem_sizes", {})
                                .get("SYCLCandidateTets", {}).get("avg", None))

                rows_raw.append(dict(
                    Legend      = legend,
                    Spacing     = spacing,
                    gpp         = gpp,               # <- control variable
                    TetsProcess = prob_sz,
                    NPTime_us   = nph_time
                ))

    # build tidy DataFrame, keep only finite values
    df_raw = (pd.DataFrame(rows_raw)
                .replace([np.inf, -np.inf], np.nan)
                .dropna(subset=["TetsProcess", "NPTime_us"]))
    df_raw = df_raw[df_raw["TetsProcess"] > 0]
    if df_raw.empty:
        raise ValueError("No finite narrow‑phase timings were found.")

    # 2 ▸ compute speed‑ups (keyed on Spacing + gpp) -----------------------
    gpu_legends = [l for l in legend_names if l != cpu_label]

    cpu_tbl = (df_raw[df_raw["Legend"] == cpu_label]
               .set_index(["Spacing", "gpp"])["NPTime_us"])

    rows_spd = []
    for legend in gpu_legends:
        sub = df_raw[df_raw["Legend"] == legend].copy()
        sub["CPU_us"] = cpu_tbl.reindex(
            sub.set_index(["Spacing", "gpp"]).index).values
        sub["SpeedUp"] = sub["CPU_us"] / sub["NPTime_us"]
        rows_spd.append(sub)

    df_spd = (pd.concat(rows_spd, ignore_index=True)
                .replace([np.inf, -np.inf], np.nan)
                .dropna(subset=["SpeedUp"]))

    # 3 ▸ global y‑limits --------------------------------------------------
    y_raw_min = df_raw["NPTime_us"].min() * 0.8
    y_raw_max = df_raw["NPTime_us"].max() * 1.25
    if df_spd.empty:                               # fall‑back baseline
        y_spd_min, y_spd_max = 0.8, 1.25
    else:
        y_spd_min = df_spd["SpeedUp"].min() * 0.8
        y_spd_max = df_spd["SpeedUp"].max() * 1.25

    # 4 ▸ figure grid ------------------------------------------------------
    n_cols = len(spacings)
    fig, axes = plt.subplots(2, n_cols, figsize=(5.0 * n_cols, 6.5),
                             sharex="col", sharey="row",
                             gridspec_kw=dict(hspace=0.10, wspace=0.15))
    if n_cols == 1:
        axes = np.array(axes).reshape(2, 1)

    spacing_titles = {}
    for spacing in spacings:
        if(spacing == "0.1"):
            spacing_titles[spacing] = "Sparse - 0.1"
        elif(spacing == "0.15"):
            spacing_titles[spacing] = "Sparse - 0.15"
        elif(spacing == "0.05"):
            spacing_titles[spacing] = "Dense - 0.05"

    # 5 ▸ row‑0 : raw timings ---------------------------------------------
    for c, spacing in enumerate(spacings):
        ax = axes[0, c]
        for i, legend in enumerate(legend_names):
            d = df_raw[(df_raw["Legend"] == legend) &
                       (df_raw["Spacing"] == spacing)]
            if d.empty:
                continue
            colour = "0.25" if legend == cpu_label else sns.color_palette()[i % 10]
            ax.loglog(d["TetsProcess"], d["NPTime_us"],
                      marker=markers[i % len(markers)],
                      ls=lstyles[i % len(lstyles)],
                      ms=5, lw=1.8, color=colour,
                      label=legend, zorder=3)

        ax.set_ylim(y_raw_min, y_raw_max)
        ax.set_title(spacing_titles.get(spacing, spacing),
                     fontsize=13, weight="bold")
        ax.grid(True, ls="-", lw=0.3, color="0.8", which="both")
        if c == 0:
            ax.set_ylabel("Narrow‑phase time [µs]", fontsize=12)
            ax.legend(frameon=False, fontsize=9, loc="upper left")

        # slope indicators
        xs = df_raw[df_raw["Spacing"] == spacing]["TetsProcess"].values
        ys = df_raw[df_raw["Spacing"] == spacing]["NPTime_us"].values
        x0 = np.percentile(xs, 25)
        if(spacing == "0.05"):
            y0 = np.percentile(ys, 65)
        else:
            y0 = np.percentile(ys, 80)
        _slope_indicator(ax, x0, y0,   2, r"$n^{2}$")
        _slope_indicator(ax, x0, y0/4, 1, r"$n$")

    # 6 ▸ row‑1 : speed‑up -------------------------------------------------
    for c, spacing in enumerate(spacings):
        ax = axes[1, c]
        for i, legend in enumerate(gpu_legends):
            d = df_spd[(df_spd["Legend"] == legend) &
                       (df_spd["Spacing"] == spacing)]
            if d.empty:
                continue
            ax.loglog(d["TetsProcess"], d["SpeedUp"],
                      marker=markers[i % len(markers)],
                      ls=lstyles[i % len(lstyles)],
                      ms=5, lw=1.8,
                      color=sns.color_palette()[i % 10],
                      label=legend, zorder=3)

        ax.axhline(1.0, color="0.3", lw=0.8, alpha=0.7)
        ax.set_ylim(y_spd_min, y_spd_max)
        ax.grid(True, ls="-", lw=0.3, color="0.8", which="both")
        ax.set_xlabel("Tets to process $n$", fontsize=12)
        if c == 0:
            ax.set_ylabel("CPU / GPU speed‑up", fontsize=12)
            ax.legend(frameon=False, fontsize=9, loc="upper left")

    return fig, axes

def plot_broad_phase_perf_speedup(cpu_data, gpu_data,
                                  folder_names, legend_names,
                                  spacings, num_gpp):
    """
    Two‑row grid (raw BroadPhase timings | GPU speed‑up).

    Identical y‑limits within each row, internal slope
    indicators, colour‑blind palette.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray shape (2, len(spacings))
    """
    # 0 ▸ cosmetics --------------------------------------------------------
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    markers  = ["o", "s", "D", "^", "v"]
    lstyles  = ["-", "--", "-.", ":"]

    # 1 ▸ assemble tidy raw‑timing table ----------------------------------
    rows_raw  = []
    cpu_label = next(lbl for lbl in legend_names if "cpu" in lbl.lower())

    for f_idx, legend in enumerate(legend_names):
        store = cpu_data if legend == cpu_label else gpu_data
        for spacing in spacings:
            for gpp in num_gpp:
                if legend == cpu_label:                # CPU path
                    bp_dict = (store[folder_names[f_idx]][spacing][gpp]
                               ["timing_overall"].get("timings", {})
                               .get("BroadPhase", {}))
                    bp_time = get_corrected_timing(bp_dict, "drake-cpu")
                else:                                  # GPU path
                    bp_dict = (store[folder_names[f_idx]][spacing][gpp]
                               ["kernel_timing"]
                               .get("kernel_timings", {})
                               .get("transform_and_broad_phase", {}))
                    bp_time = get_corrected_timing(bp_dict, "sycl-gpu")

                rows_raw.append(dict(
                    Legend     = legend,
                    Spacing    = spacing,
                    Bodies     = calculate_actual_objects(gpp),
                    BPTime_us  = bp_time
                ))

    df_raw = pd.DataFrame(rows_raw)

    # 2 ▸ compute speed‑ups ------------------------------------------------
    gpu_legends = [l for l in legend_names if l != cpu_label]
    rows_spd    = []

    cpu_tbl = (df_raw[df_raw["Legend"] == cpu_label]
               .set_index(["Spacing", "Bodies"])["BPTime_us"])

    for legend in gpu_legends:
        sub = df_raw[df_raw["Legend"] == legend].copy()
        sub["CPU_us"] = cpu_tbl.reindex(sub.set_index(["Spacing", "Bodies"]).index).values
        sub["SpeedUp"] = sub["CPU_us"] / sub["BPTime_us"]
        rows_spd.append(sub)

    df_spd = pd.concat(rows_spd, ignore_index=True)

    # 3 ▸ figure grid – sharey='row' keeps y equal in each row ------------
    n_cols = len(spacings)
    fig, axes = plt.subplots(2, n_cols, figsize=(5.0 * n_cols, 6.5),
                             sharex="col", sharey="row",
                             gridspec_kw=dict(hspace=0.10, wspace=0.15))
    if n_cols == 1:
        axes = np.array(axes).reshape(2, 1)

    # common y‑limits
    y_raw_min, y_raw_max = df_raw["BPTime_us"].min(), df_raw["BPTime_us"].max()
    y_spd_min, y_spd_max = df_spd["SpeedUp"].min(), df_spd["SpeedUp"].max()
    y_raw_min *= 0.8;  y_raw_max *= 1.25
    y_spd_min *= 0.8;  y_spd_max *= 1.25

    spacing_titles = {}
    for spacing in spacings:
        if(spacing == "0.1"):
            spacing_titles[spacing] = "Sparse - 0.1"
        elif(spacing == "0.15"):
            spacing_titles[spacing] = "Sparse - 0.15"
        elif(spacing == "0.05"):
            spacing_titles[spacing] = "Dense - 0.05"

    # 4 ▸ row‑0 : raw timings ---------------------------------------------
    for c, spacing in enumerate(spacings):
        ax = axes[0, c]
        for i, legend in enumerate(legend_names):
            d = df_raw[(df_raw["Legend"] == legend) &
                       (df_raw["Spacing"] == spacing)]
            if d.empty:
                continue
            color = "0.25" if legend == cpu_label else sns.color_palette()[i % 10]
            ax.loglog(d["Bodies"], d["BPTime_us"],
                      marker=markers[i % len(markers)],
                      ls=lstyles[i % len(lstyles)],
                      ms=5, lw=1.8, color=color,
                      label=legend, zorder=3)

        ax.set_ylim(y_raw_min, y_raw_max)
        ax.set_title(spacing_titles.get(spacing, spacing),
                     fontsize=13, weight="bold")
        ax.grid(True, ls="-", lw=0.3, color="0.8", which="both")
        if c == 0:
            ax.set_ylabel("BroadPhase time [µs]", fontsize=12)
            ax.legend(frameon=False, fontsize=9, loc="upper left")

        # internal slope indicators
        xs = df_raw[df_raw["Spacing"] == spacing]["Bodies"].values
        ys = df_raw[df_raw["Spacing"] == spacing]["BPTime_us"].values
        x0 = np.percentile(xs, 25)
        y0 = np.percentile(ys, 30)
        _slope_indicator_nlogn(ax, x0, y0)   # nlogn

    # 5 ▸ row‑1 : speed‑up -------------------------------------------------
    for c, spacing in enumerate(spacings):
        ax = axes[1, c]
        for i, legend in enumerate(gpu_legends):
            d = df_spd[(df_spd["Legend"] == legend) &
                       (df_spd["Spacing"] == spacing)]
            if d.empty:
                continue
            ax.loglog(d["Bodies"], d["SpeedUp"],
                      marker=markers[i % len(markers)],
                      ls=lstyles[i % len(lstyles)],
                      ms=5, lw=1.8,
                      color=sns.color_palette()[i % 10],
                      label=legend, zorder=3)

        ax.axhline(1.0, color="0.3", lw=0.8, alpha=0.7)
        ax.set_ylim(y_spd_min, y_spd_max)
        ax.grid(True, ls="-", lw=0.3, color="0.8", which="both")
        ax.set_xlabel("Number of Geometries $n$", fontsize=12)
        if c == 0:
            ax.set_ylabel("CPU / GPU speed‑up", fontsize=12)
            ax.legend(frameon=False, fontsize=9, loc="upper left")

    # 6 ▸ finish -----------------------------------------------------------
    return fig, axes

def plot_broad_phase_perf_speedup_vs_num_elements(cpu_data, gpu_data,
                                  folder_names, legend_names,
                                  spacings, num_gpp):
    """
    Two‑row grid (raw BroadPhase timings | GPU speed‑up).

    Identical y‑limits within each row, internal slope
    indicators, colour‑blind palette.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray shape (2, len(spacings))
    """
    # 0 ▸ cosmetics --------------------------------------------------------
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    markers  = ["o", "s", "D", "^", "v"]
    lstyles  = ["-", "--", "-.", ":"]

    # 1 ▸ assemble tidy raw‑timing table ----------------------------------
    rows_raw  = []
    cpu_label = next(lbl for lbl in legend_names if "cpu" in lbl.lower())

    for f_idx, legend in enumerate(legend_names):
        store = cpu_data if legend == cpu_label else gpu_data
        for spacing in spacings:
            for gpp in num_gpp:
                if legend == cpu_label:                # CPU path
                    bp_dict = (store[folder_names[f_idx]][spacing][gpp]
                               ["timing_overall"].get("timings", {})
                               .get("BroadPhase", {}))
                    bp_time = get_corrected_timing(bp_dict, "drake-cpu")
                else:                                  # GPU path
                    bp_dict = (store[folder_names[f_idx]][spacing][gpp]
                               ["kernel_timing"]
                               .get("kernel_timings", {})
                               .get("transform_and_broad_phase", {}))
                    bp_time = get_corrected_timing(bp_dict, "sycl-gpu")

                rows_raw.append(dict(
                    Legend     = legend,
                    Spacing    = spacing,
                    Bodies     = calculate_number_of_elements_objects_scaling(gpp, cpu_data[folder_names[f_idx]][spacing][gpp]["problem_size"]),
                    BPTime_us  = bp_time
                ))

    df_raw = pd.DataFrame(rows_raw)

    # 2 ▸ compute speed‑ups ------------------------------------------------
    gpu_legends = [l for l in legend_names if l != cpu_label]
    rows_spd    = []

    cpu_tbl = (df_raw[df_raw["Legend"] == cpu_label]
               .set_index(["Spacing", "Bodies"])["BPTime_us"])

    for legend in gpu_legends:
        sub = df_raw[df_raw["Legend"] == legend].copy()
        sub["CPU_us"] = cpu_tbl.reindex(sub.set_index(["Spacing", "Bodies"]).index).values
        sub["SpeedUp"] = sub["CPU_us"] / sub["BPTime_us"]
        rows_spd.append(sub)

    df_spd = pd.concat(rows_spd, ignore_index=True)

    # 3 ▸ figure grid – sharey='row' keeps y equal in each row ------------
    n_cols = len(spacings)
    fig, axes = plt.subplots(2, n_cols, figsize=(5.0 * n_cols, 6.5),
                             sharex="col", sharey="row",
                             gridspec_kw=dict(hspace=0.10, wspace=0.15))
    if n_cols == 1:
        axes = np.array(axes).reshape(2, 1)

    # common y‑limits
    y_raw_min, y_raw_max = df_raw["BPTime_us"].min(), df_raw["BPTime_us"].max()
    y_spd_min, y_spd_max = df_spd["SpeedUp"].min(), df_spd["SpeedUp"].max()
    y_raw_min *= 0.8;  y_raw_max *= 1.25
    y_spd_min *= 0.8;  y_spd_max *= 1.25

    spacing_titles = {}
    for spacing in spacings:
        if(spacing == "0.1"):
            spacing_titles[spacing] = "Sparse - 0.1"
        elif(spacing == "0.15"):
            spacing_titles[spacing] = "Sparse - 0.15"
        elif(spacing == "0.05"):
            spacing_titles[spacing] = "Dense - 0.05"

    # 4 ▸ row‑0 : raw timings ---------------------------------------------
    for c, spacing in enumerate(spacings):
        ax = axes[0, c]
        for i, legend in enumerate(legend_names):
            d = df_raw[(df_raw["Legend"] == legend) &
                       (df_raw["Spacing"] == spacing)]
            if d.empty:
                continue
            color = "0.25" if legend == cpu_label else sns.color_palette()[i % 10]
            ax.loglog(d["Bodies"], d["BPTime_us"],
                      marker=markers[i % len(markers)],
                      ls=lstyles[i % len(lstyles)],
                      ms=5, lw=1.8, color=color,
                      label=legend, zorder=3)

        ax.set_ylim(y_raw_min, y_raw_max)
        ax.set_title(spacing_titles.get(spacing, spacing),
                     fontsize=13, weight="bold")
        ax.grid(True, ls="-", lw=0.3, color="0.8", which="both")
        if c == 0:
            ax.set_ylabel("BroadPhase time [µs]", fontsize=12)
            ax.legend(frameon=False, fontsize=9, loc="upper left")

        # internal slope indicators
        xs = df_raw[df_raw["Spacing"] == spacing]["Bodies"].values
        ys = df_raw[df_raw["Spacing"] == spacing]["BPTime_us"].values
        x0 = np.percentile(xs, 25)
        y0 = np.percentile(ys, 30)
        _slope_indicator_nlogn(ax, x0, y0)   # nlogn

    # 5 ▸ row‑1 : speed‑up -------------------------------------------------
    for c, spacing in enumerate(spacings):
        ax = axes[1, c]
        for i, legend in enumerate(gpu_legends):
            d = df_spd[(df_spd["Legend"] == legend) &
                       (df_spd["Spacing"] == spacing)]
            if d.empty:
                continue
            ax.loglog(d["Bodies"], d["SpeedUp"],
                      marker=markers[i % len(markers)],
                      ls=lstyles[i % len(lstyles)],
                      ms=5, lw=1.8,
                      color=sns.color_palette()[i % 10],
                      label=legend, zorder=3)

        ax.axhline(1.0, color="0.3", lw=0.8, alpha=0.7)
        ax.set_ylim(y_spd_min, y_spd_max)
        ax.grid(True, ls="-", lw=0.3, color="0.8", which="both")
        ax.set_xlabel("Number of Elements $n$", fontsize=12)
        if c == 0:
            ax.set_ylabel("CPU / GPU speed‑up", fontsize=12)
            ax.legend(frameon=False, fontsize=9, loc="upper left")

    # 6 ▸ finish -----------------------------------------------------------
    return fig, axes

def plot_hydroelastic_query_perf_speedup(
        gpu_data, cpu_data, folder_names, legend_names, spacings, num_gpp):
    """
    Two‑row grid (raw timings | speed‑up) with:
      • identical y‑limits within each row (so left & right columns align),
      • internal O(n²) and O(n) slope indicators,
      • colour‑blind palette & compact legends.
    """
    # 0  Cosmetic defaults -------------------------------------------------
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    markers = ["o", "s", "D", "^", "v"]
    lstyles = ["-", "--", "-.", ":"]

    # 1  Build tidy table of raw timings ----------------------------------
    rows_raw = []
    cpu_label = next(lbl for lbl in legend_names if "cpu" in lbl.lower())

    for f_idx, legend in enumerate(legend_names):
        store = cpu_data if legend == cpu_label else gpu_data
        for spacing in spacings:
            for gpp in num_gpp:
                timing_dict = store[folder_names[f_idx]][spacing][gpp] \
                                  ["timing_overall"].get("timings", {})
                hq_time = get_corrected_timing(
                    timing_dict.get("HydroelasticQuery", {}),
                    "drake-cpu" if legend == cpu_label else "sycl-gpu")
                rows_raw.append(dict(Legend=legend,
                                     Spacing=spacing,
                                     Bodies=calculate_actual_objects(gpp),
                                     HQTime_us=hq_time))

    df_raw = pd.DataFrame(rows_raw)

    # 2  Speed‑up (= CPU / GPU) -------------------------------------------
    gpu_legends = [l for l in legend_names if l != cpu_label]
    rows_spd = []
    cpu_tbl = (df_raw[df_raw["Legend"] == cpu_label]
               .set_index(["Spacing", "Bodies"])["HQTime_us"])
    for legend in gpu_legends:
        sub = df_raw[df_raw["Legend"] == legend].copy()
        sub["CPU_us"] = cpu_tbl.reindex(sub.set_index(["Spacing", "Bodies"]).index).values
        sub["SpeedUp"] = sub["CPU_us"] / sub["HQTime_us"]
        rows_spd.append(sub)
    df_spd = pd.concat(rows_spd, ignore_index=True)

    # 3  Prepare grid – sharey='row' keeps y identical per row ------------
    n_cols = len(spacings)
    fig, axes = plt.subplots(2, n_cols, figsize=(5.0 * n_cols, 6.5),
                             sharex="col", sharey="row",
                             gridspec_kw=dict(hspace=0.10, wspace=0.15))
    if n_cols == 1:
        axes = np.array(axes).reshape(2, 1)

    # Pre‑compute common y‑limits
    y_raw_min, y_raw_max = df_raw["HQTime_us"].min(), df_raw["HQTime_us"].max()
    y_spd_min, y_spd_max = df_spd["SpeedUp"].min(), df_spd["SpeedUp"].max()

    # Nice padding
    y_raw_min *= 0.8
    y_raw_max *= 1.25
    y_spd_min *= 0.8
    y_spd_max *= 1.25

    spacing_titles = {}
    for spacing in spacings:
        if(spacing == "0.1"):
            spacing_titles[spacing] = "Sparse - 0.1"
        elif(spacing == "0.15"):
            spacing_titles[spacing] = "Sparse - 0.15"
        elif(spacing == "0.05"):
            spacing_titles[spacing] = "Dense - 0.05"

    # 4  Plot raw timings (row 0) -----------------------------------------
    for c, spacing in enumerate(spacings):
        ax = axes[0, c]
        for i, legend in enumerate(legend_names):
            d = df_raw[(df_raw["Legend"] == legend) & (df_raw["Spacing"] == spacing)]
            if d.empty:
                continue
            color = "0.25" if legend == cpu_label else sns.color_palette()[i % 10]
            ax.loglog(d["Bodies"], d["HQTime_us"],
                      marker=markers[i % len(markers)],
                      ls=lstyles[i % len(lstyles)],
                      ms=5, lw=1.8, color=color, label=legend, zorder=3)

        ax.set_ylim(y_raw_min, y_raw_max)
        ax.set_title(spacing_titles.get(spacing, spacing), fontsize=13, weight="bold")
        ax.grid(True, ls="-", lw=0.3, color="0.8", which="both")
        if c == 0:
            ax.set_ylabel("HydroelasticQuery time [µs]", fontsize=12)
            ax.legend(frameon=False, fontsize=9, loc="upper left")

        # Slope indicators – place safely inside limits
        xs = df_raw[df_raw["Spacing"] == spacing]["Bodies"].values
        ys = df_raw[df_raw["Spacing"] == spacing]["HQTime_us"].values
        x0 = np.percentile(xs, 25)
        y0 = np.percentile(ys, 30)
        _slope_indicator(ax, x0, y0, 2, r"$n^{2}$")          # quadratic
        _slope_indicator(ax, x0, y0 / 4, 1, r"$n$")          # linear (below)

    # 5  Plot speed‑up (row 1) --------------------------------------------
    for c, spacing in enumerate(spacings):
        ax = axes[1, c]
        for i, legend in enumerate(gpu_legends):
            d = df_spd[(df_spd["Legend"] == legend) & (df_spd["Spacing"] == spacing)]
            if d.empty:
                continue
            ax.loglog(d["Bodies"], d["SpeedUp"],
                      marker=markers[i % len(markers)],
                      ls=lstyles[i % len(lstyles)],
                      ms=5, lw=1.8, color=sns.color_palette()[i % 10],
                      label=legend, zorder=3)

        ax.axhline(1.0, color="0.3", lw=0.8, alpha=0.7)
        ax.set_ylim(y_spd_min, y_spd_max)
        ax.grid(True, ls="-", lw=0.3, color="0.8", which="both")
        ax.set_xlabel("Number of Geometries $n$", fontsize=12)
        if c == 0:
            ax.set_ylabel("CPU / GPU speed‑up", fontsize=12)
            ax.legend(frameon=False, fontsize=9, loc="upper left")

    # 6  Finish ------------------------------------------------------------
    return fig, axes


def plot_hydroelastic_query_perf_speedup_vs_num_elements(
        gpu_data, cpu_data, folder_names, legend_names, spacings, num_gpp):
    """
    Two‑row grid (raw timings | speed‑up) with:
      • identical y‑limits within each row (so left & right columns align),
      • internal O(n²) and O(n) slope indicators,
      • colour‑blind palette & compact legends.
    """
    # 0  Cosmetic defaults -------------------------------------------------
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    markers = ["o", "s", "D", "^", "v"]
    lstyles = ["-", "--", "-.", ":"]

    # 1  Build tidy table of raw timings ----------------------------------
    rows_raw = []
    cpu_label = next(lbl for lbl in legend_names if "cpu" in lbl.lower())

    for f_idx, legend in enumerate(legend_names):
        store = cpu_data if legend == cpu_label else gpu_data
        for spacing in spacings:
            for gpp in num_gpp:
                timing_dict = store[folder_names[f_idx]][spacing][gpp] \
                                  ["timing_overall"].get("timings", {})
                hq_time = get_corrected_timing(
                    timing_dict.get("HydroelasticQuery", {}),
                    "drake-cpu" if legend == cpu_label else "sycl-gpu")
                rows_raw.append(dict(Legend=legend,
                                     Spacing=spacing,
                                     Bodies=calculate_number_of_elements_objects_scaling(gpp, cpu_data[folder_names[f_idx]][spacing][gpp]["problem_size"]),
                                     HQTime_us=hq_time))

    df_raw = pd.DataFrame(rows_raw)

    # 2  Speed‑up (= CPU / GPU) -------------------------------------------
    gpu_legends = [l for l in legend_names if l != cpu_label]
    rows_spd = []
    cpu_tbl = (df_raw[df_raw["Legend"] == cpu_label]
               .set_index(["Spacing", "Bodies"])["HQTime_us"])
    for legend in gpu_legends:
        sub = df_raw[df_raw["Legend"] == legend].copy()
        sub["CPU_us"] = cpu_tbl.reindex(sub.set_index(["Spacing", "Bodies"]).index).values
        sub["SpeedUp"] = sub["CPU_us"] / sub["HQTime_us"]
        rows_spd.append(sub)
    df_spd = pd.concat(rows_spd, ignore_index=True)

    # 3  Prepare grid – sharey='row' keeps y identical per row ------------
    n_cols = len(spacings)
    fig, axes = plt.subplots(2, n_cols, figsize=(5.0 * n_cols, 6.5),
                             sharex="col", sharey="row",
                             gridspec_kw=dict(hspace=0.10, wspace=0.15))
    if n_cols == 1:
        axes = np.array(axes).reshape(2, 1)

    # Pre‑compute common y‑limits
    y_raw_min, y_raw_max = df_raw["HQTime_us"].min(), df_raw["HQTime_us"].max()
    y_spd_min, y_spd_max = df_spd["SpeedUp"].min(), df_spd["SpeedUp"].max()

    # Nice padding
    y_raw_min *= 0.8
    y_raw_max *= 1.25
    y_spd_min *= 0.8
    y_spd_max *= 1.25

    spacing_titles = {}
    for spacing in spacings:
        if(spacing == "0.1"):
            spacing_titles[spacing] = "Sparse - 0.1"
        elif(spacing == "0.15"):
            spacing_titles[spacing] = "Sparse - 0.15"
        elif(spacing == "0.05"):
            spacing_titles[spacing] = "Dense - 0.05"

    # 4  Plot raw timings (row 0) -----------------------------------------
    for c, spacing in enumerate(spacings):
        ax = axes[0, c]
        for i, legend in enumerate(legend_names):
            d = df_raw[(df_raw["Legend"] == legend) & (df_raw["Spacing"] == spacing)]
            if d.empty:
                continue
            color = "0.25" if legend == cpu_label else sns.color_palette()[i % 10]
            ax.loglog(d["Bodies"], d["HQTime_us"],
                      marker=markers[i % len(markers)],
                      ls=lstyles[i % len(lstyles)],
                      ms=5, lw=1.8, color=color, label=legend, zorder=3)

        ax.set_ylim(y_raw_min, y_raw_max)
        ax.set_title(spacing_titles.get(spacing, spacing), fontsize=13, weight="bold")
        ax.grid(True, ls="-", lw=0.3, color="0.8", which="both")
        if c == 0:
            ax.set_ylabel("HydroelasticQuery time [µs]", fontsize=12)
            ax.legend(frameon=False, fontsize=9, loc="upper left")

        # Slope indicators – place safely inside limits
        xs = df_raw[df_raw["Spacing"] == spacing]["Bodies"].values
        ys = df_raw[df_raw["Spacing"] == spacing]["HQTime_us"].values
        x0 = np.percentile(xs, 25)
        y0 = np.percentile(ys, 30)
        _slope_indicator(ax, x0, y0, 2, r"$n^{2}$")          # quadratic
        _slope_indicator(ax, x0, y0 / 4, 1, r"$n$")          # linear (below)

    # 5  Plot speed‑up (row 1) --------------------------------------------
    for c, spacing in enumerate(spacings):
        ax = axes[1, c]
        for i, legend in enumerate(gpu_legends):
            d = df_spd[(df_spd["Legend"] == legend) & (df_spd["Spacing"] == spacing)]
            if d.empty:
                continue
            ax.loglog(d["Bodies"], d["SpeedUp"],
                      marker=markers[i % len(markers)],
                      ls=lstyles[i % len(lstyles)],
                      ms=5, lw=1.8, color=sns.color_palette()[i % 10],
                      label=legend, zorder=3)

        ax.axhline(1.0, color="0.3", lw=0.8, alpha=0.7)
        ax.set_ylim(y_spd_min, y_spd_max)
        ax.grid(True, ls="-", lw=0.3, color="0.8", which="both")
        ax.set_xlabel("Number of Elements $n$", fontsize=12)
        if c == 0:
            ax.set_ylabel("CPU / GPU speed‑up", fontsize=12)
            ax.legend(frameon=False, fontsize=9, loc="upper left")

    # 6  Finish ------------------------------------------------------------
    return fig, axes
