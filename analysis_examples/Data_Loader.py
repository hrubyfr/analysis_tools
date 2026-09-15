#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os

import analysis_tools
import analysis_tools.beam_selection
from analysis_tools import DataLoader
from analysis_tools import BeamSelection, Cut, print_cherenkov_thresholds, SelectionMonitor
import time
import awkward as ak
from matplotlib.path import Path


# ### Open the file

# In[5]:


def process_run(run_number : int, FILE : Path):

    loader = DataLoader(FILE)


    # In[6]:


    # Check which particles are above Cherenkov threshold in each ACT for this run.
    # This tells you whether the ACT-based cuts in your selection are meaningful.
    # Look out for kaon runs in particular 
    # Be careful, particles will have lost momentum by the time they reach the ACTs, the values indicated here are indicatve only.   
    vme_run_info = loader.get_vme_analysis_run_info()
    print_cherenkov_thresholds(vme_run_info)


    # ### Define the PID selections based on the beam analysis information
    # Below is an example selections based on the nominal cuts lines derived in the VME beam analysis:
    # - act_eveto (upstream): electrons above threshold;  muons and pions below
    # - act_tagger (upstream): electrons and muons above threshold; pions below
    # - additional cut on TOF based on expected proton TOF
    # 
    # One should refine these cuts based on one's own analysis needs. 
    # One example of additional cut presented here is on the muon tagger charge which requires that the particle reaches the muon tagger, useful for selecting through-going particles. 

    # In[ ]:


    vme_scalar_results = loader.get_vme_analysis_scalar_results()

    # --- Define your particle selections ---
    # Every cut is a [variable, operator, value] triplet.
    # You can apply a cut to any VME variable 
    # Operators: ">", "<", ">=", "<=", "==", "!=", "between" (value must be [low, high] for "between").
    # Omit the TOF cut entirely if proton_tof_cut is 0 
    # This case of TOF separation unavailable happensfor negative polarity and low momentum runs in production 1.0,
    # To be improved in production 1.1 

    tof_cut    = vme_scalar_results['proton_tof_cut']
    if tof_cut == 0:
        print("WARNING: TOF separation unavailable for this run, setting TOF cut to default value of 999 ns.")
        tof_cut = 999

    eveto_cut  = vme_scalar_results['act_eveto_cut']
    tagger_cut = vme_scalar_results['act_tagger_cut']

    # You can select CROSSING MUONS with a cut on the muon tagger signal
    # at higher momentum. mu_tag_cut is the nominal cut line 
    mu_cut     = vme_scalar_results['mu_tag_cut']

    # I you want to apply a cut on a new variable, 

    # PIONS: fast particles that do not produce Cherenkov light in either ACT.
    pion_sel = BeamSelection.selection(
        "pion",
        ["vme_act_eveto",  "<", eveto_cut],
        ["vme_act_tagger", "<", tagger_cut],
        ["vme_tof_corr",        "<", tof_cut],
    )

    # MUONS: fast particles, below threshold in act_eveto, above threshold in act_tagger,
    #        with an additional cut on the total muon tagger signal.
    muon_sel = BeamSelection.selection(
        "muon",
        ["vme_act_eveto",    "<", eveto_cut],
        ["vme_act_tagger",   ">", tagger_cut],
        ["vme_tof_corr",          "<", tof_cut],
        #Only add the muon tag cut if you want CROSSING MUONS
        #["vme_mu_tag_total", ">", mu_cut],

    )

    # ELECTRONS: fast particles above threshold in the upstream ACT (act_eveto).
    ele_sel = BeamSelection.selection(
        "electron",
        ["vme_act_eveto", ">", eveto_cut],
        ["vme_tof_corr",       "<", tof_cut],
        # You can refine the electron selection with additional cuts on individual ACT PMT signals
        # You could also define new beam variables on which to cut (e.g. symmetry of the charge)
        # ["vme_act0_l_charge",       ">", 10],
    )

    # PROTONS: slow particles identified by their TOF falling in a window above the
    #          fast/slow separation value. Only meaningful when proton_tof_cut > 0.
    proton_sel = BeamSelection.selection(
        "proton",
        ["vme_tof_corr", "between", [tof_cut, tof_cut + 10]],
    )

    pion_sel.describe()


    # ### Loading the data in batches

    # In[ ]:



    #call these functions to apply data quality flags at time of loading
    loader.apply_mPMT_data_quality_cuts()
    loader.apply_vme_event_quality_cuts()
    loader.apply_t5_event_quality_cuts()

    #toggle for writing the output parquet file for selections
    write_parquet_file = False

    if write_parquet_file:
        # Enable parquet output for the selections you want to save.
        # Default filename is "<particle>.parquet". Pass a path to override.
        pion_sel.enable_parquet_output(f"run{run_number}_pions.parquet")
        muon_sel.enable_parquet_output(f"run{run_number}_muons.parquet")
        ele_sel.enable_parquet_output(f"run{run_number}_electrons.parquet")

    # Decide which selections you want to monitor live during loading. This is optional but useful for understanding cut lines.
    selections = [pion_sel, muon_sel, ele_sel, proton_sel] 
    # monitor    = SelectionMonitor(selections, update_every=10, vme_run_info=vme_run_info)

    start_time = time.time()
    n_windows_passing = 0
    total_entries = loader.file["WCTEReadoutWindows"].num_entries

    T5_branches = ["T5_event_nr", "T5_particle_nr", 
    "T5_HasValidHit", "T5_HasMultipleScintillatorsHit", "T5_HasOutOfTimeWindow", "T5_HasInTimeWindow", "T5_hit_is_in_bounds", "T5_hit_pos_x", "T5_hit_pos_y", "T5_hit_time", "T5_secondary_hit_is_in_bounds", "T5_secondary_hit_pos_x", "T5_secondary_hit_pos_y", "T5_secondary_hit_time"]

    T5_pion_batch = []
    T5_muon_batch = []
    T5_electron_batch = []
    T5_proton_batch = []

    folders = ["pions", "muons", "electrons", "protons"]
    file_names = [[f"run{run_number}_T5_pions.parquet", T5_pion_batch], [f"run{run_number}_T5_muons.parquet", T5_muon_batch], [f"run{run_number}_T5_electrons.parquet", T5_electron_batch], [f"run{run_number}_T5_protons.parquet", T5_proton_batch]]

    for i, (folder, (name, _)) in enumerate(zip(folders, file_names)):
        os.makedirs(os.path.join(os.getcwd(), "data", folder), exist_ok=True)
        file_names[i][0] = os.path.join(os.getcwd(), "data", folder, name)

    for i_batch, batch in enumerate(loader.iterate(verbose=False, step_size="100 MB")):
        print("On Event",n_windows_passing,"/",total_entries)
        n_windows_passing += len(batch)

        # monitor.update(batch)

        if write_parquet_file:
            for sel in selections:
                sel._write_to_parquet(batch[sel.mask(batch)])

        #alternatively, do analysis on individual bath
        pion_batch = batch[pion_sel.mask(batch)]
        muon_batch = batch[muon_sel.mask(batch)]
        electron_batch = batch[ele_sel.mask(batch)]
        proton_batch = batch[proton_sel.mask(batch)]
        T5_pion_batch.append(pion_batch[T5_branches])
        T5_muon_batch.append(muon_batch[T5_branches])
        T5_electron_batch.append(electron_batch[T5_branches])
        T5_proton_batch.append(proton_batch[T5_branches])

    # T5_pion_batch.append(vme_run_info)
    # T5_muon_batch.append(vme_run_info)
    # T5_electron_batch.append(vme_run_info)
    # T5_proton_batch.append(vme_run_info)

    for file_name, batch_list in file_names:
        ak.to_parquet(ak.concatenate(batch_list), file_name, compression="snappy")
        print(f"Saved {file_name} with {len(ak.concatenate(batch_list))} events ")


    if write_parquet_file:
        for sel in selections:
            sel.close_parquet_writer()

    # monitor.show()
    print(f"Loaded {n_windows_passing} events across {i_batch+1} batches  "
        f"({time.time() - start_time:.1f} s)")


    # In[ ]:


    # Read back the saved muon events and inspect them.
    # The same pattern works for pions (pion_sel._parquet_path) and electrons.
    # muons = ak.from_parquet(muon_sel._parquet_path)

    # print(f"Total muons saved : {len(muons)}")
    # print(f"Mean corrected TOF: {float(ak.mean(muons['vme_tof_corr'])):.2f} ns")
    # print(f"Mean act_eveto    : {float(ak.mean(muons['vme_act_eveto'])):.3f} PE")
    # print(f"Mean act_tagger   : {float(ak.mean(muons['vme_act_tagger'])):.3f} PE")


    # ## Demonstrate functions to get the configuration data from the file

    # Get the good PMT list (by mPMT slot and pmt position). Any hit which is not from a good channel is be masked. This is useful for determining which channels are reading out stably for comparison to monte carlo.

    # In[8]:


    #get the good PMTs slot and position
    good_wcte_mpmt_slots, good_wcte_pmt_pos = loader.get_good_wcte_pmts()
    print(good_wcte_mpmt_slots,good_wcte_pmt_pos)


    # In[9]:


    print("Get the configuration data from the merged file:")
    config = loader.get_configuration()
    print(config)

    print("Get the data quality metrics from the merged file:")
    dqm = loader.get_data_quality_metrics()
    print(dqm)

    print("Get the data quality metrics from the merged file:")
    dqm = loader.get_data_quality_metrics()
    print(dqm)

    print("Get the vme_analysis_scalar_results from the merged file:")
    vme_scalar_results = loader.get_vme_analysis_scalar_results()
    print(vme_scalar_results)

    print("Get the vme_analysis_scalar_results from the merged file:")
    vme_run_info = loader.get_vme_analysis_run_info()
    print(vme_run_info)


# In[ ]:

if __name__ == "__main__":
    run_number = 1610
    FILE = f"/eos/experiment/wcte/data/2025_commissioning/processed_offline_data/production_v1_0/{run_number}/WCTE_merged_production_R{run_number}.root"
    process_run(run_number, FILE)



