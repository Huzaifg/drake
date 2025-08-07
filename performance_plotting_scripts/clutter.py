import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from utils import plot_faces_inserted_vs_obp, get_data, plot_candidate_tets_vs_obp, plot_broad_narrow_misc_vs_obp, plot_broad_narrow_misc_vs_num_elements_clutter
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


def main():
    base_dir = os.path.dirname(os.getcwd())
    demo_name = "clutter"
    objects_per_pile = ["1", "2", "5", "10", "20", "33", "50"]
    sphere_resolutions = ["0.0050", "0.0100", "0.0200", "0.0400"]
    
    
    run_types = ["sycl-gpu", "drake-cpu"]
    perf_folder = "performance_jsons_clutter_opt1"

    # Store all data in a nested dictionary: all_data[run_type][spacing][num_gpp][data_type]
    all_data = {run_type: {} for run_type in run_types}
    for run_type in run_types:
        for obp in objects_per_pile:
            all_data[run_type][obp] = {}
            for sr in sphere_resolutions:
                all_data[run_type][obp][sr] = {}
                # Problem size
                json_path_problem_size = f"{base_dir}/{perf_folder}/{demo_name}_{obp}_1.0000_{sr}_3_{run_type}_problem_size.json"
                data_problem_size = get_data(json_path_problem_size)
                all_data[run_type][obp][sr]["problem_size"] = data_problem_size
                
                # Timing overall
                json_path_timing_overall = f"{base_dir}/{perf_folder}/{demo_name}_{obp}_1.0000_{sr}_3_{run_type}_timing_overall.json"
                data_timing_overall = get_data(json_path_timing_overall)
                all_data[run_type][obp][sr]["timing_overall"] = data_timing_overall
                
                # Advance to timing
                json_path_advance_to = f"{base_dir}/{perf_folder}/{demo_name}_{obp}_1.0000_{sr}_3_{run_type}_timing_advance_to.json"
                data_advance_to = get_data(json_path_advance_to)
                all_data[run_type][obp][sr]["advance_to"] = data_advance_to

                if(run_type == "sycl-gpu" or run_type == "sycl-cpu"):
                    json_path_kernel_timing = f"{base_dir}/{perf_folder}/{demo_name}_{obp}_1.0000_{sr}_3_{run_type}_timing.json"
                    data_kernel_timing = get_data(json_path_kernel_timing)
                    all_data[run_type][obp][sr]["kernel_timing"] = data_kernel_timing
                   
    plot_faces_inserted_vs_obp(all_data, run_types, objects_per_pile, sphere_resolutions)
    plot_dir = "plots_clutter_opt1"
    if not os.path.exists(f"{base_dir}/{plot_dir}"):
        os.makedirs(f"{base_dir}/{plot_dir}")
    plt.savefig(f"{base_dir}/{plot_dir}/clutter_faces_inserted_vs_obp.png",dpi=600)
    print("Saved plot to ", f"{base_dir}/{plot_dir}/clutter_faces_inserted_vs_obp.png")
    plt.show()
    plt.close()
    
    plot_candidate_tets_vs_obp(all_data, run_types, objects_per_pile, sphere_resolutions)
    plot_dir = "plots_clutter_opt1"
    if not os.path.exists(f"{base_dir}/{plot_dir}"):
        os.makedirs(f"{base_dir}/{plot_dir}")
    plt.savefig(f"{base_dir}/{plot_dir}/clutter_candidate_tets_vs_obp.png",dpi=600)
    print("Saved plot to ", f"{base_dir}/{plot_dir}/clutter_candidate_tets_vs_obp.png")
    plt.show()
    plt.close()
    
    plot_broad_narrow_misc_vs_obp(all_data, run_types, objects_per_pile, sphere_resolutions)
    plot_dir = "plots_clutter_opt1"
    if not os.path.exists(f"{base_dir}/{plot_dir}"):
        os.makedirs(f"{base_dir}/{plot_dir}")
    plt.savefig(f"{base_dir}/{plot_dir}/clutter_broad_narrow_misc_vs_obp.png",dpi=600)
    print("Saved plot to ", f"{base_dir}/{plot_dir}/clutter_broad_narrow_misc_vs_obp.png")
    plt.show()
    plt.close()
    
    plot_broad_narrow_misc_vs_num_elements_clutter(all_data, run_types, objects_per_pile, sphere_resolutions)
    plot_dir = "plots_clutter_opt1"
    if not os.path.exists(f"{base_dir}/{plot_dir}"):
        os.makedirs(f"{base_dir}/{plot_dir}")
    plt.savefig(f"{base_dir}/{plot_dir}/clutter_broad_narrow_misc_vs_num_elements_clutter.png",dpi=600)
    print("Saved plot to ", f"{base_dir}/{plot_dir}/clutter_broad_narrow_misc_vs_num_elements_clutter.png")
    plt.show()
    plt.close()
        
if __name__ == "__main__":
    main()