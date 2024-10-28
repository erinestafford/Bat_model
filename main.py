# Main file for running the ABM
import test_bat
import patches
from run_simulation import *
import numpy as np
import pickle
import os

# Example using some of the Australia data
#with open('test_big_patch_net.pkl', 'rb') as f:
#    patch_net = pickle.load(f)

#example on small synthetic grid
#with open('small_patch_prob_ex.pkl', 'rb') as f:
#    patch_net = pickle.load(f)

#example with Thai data
#with open('Thai_net.pkl', 'rb') as f:
#    patch_net = pickle.load(f)

#with open('small_synth_clustered.pkl', 'rb') as f:
#    patch_net = pickle.load(f)
#with open('small_synth_clustered_resources_uneven.pkl', 'rb') as f:
#    patch_net = pickle.load(f)
#with open('fragmentation_net_base.pkl', 'rb') as f:
with open('fragmentation_net_i1.pkl','rb') as f:
    patch_net = pickle.load(f)

#parameters to control simulation
simulation_parameters = {'pop':5000, #bat population
                         'patch_types_options': ['Roost', 'Residential', 'Orchard', 'Water Body', 'Forest', 'Dump'], #types of patches in grid
                         'num_p': len(patch_net.keys()), #number of grid patches
                         'gp':[0.5,0.5], #gender distribution of bat population ([male, female])
                         'grid_scale': [1,1], #scale of grid patches ([width,height]) in kilometers
                         'patch_type_forage_probs': np.array([0.0,0.05, 0.4, 0.4, 0.1, 0.05]), #initialization parameter for probability of foraging in each patch type
                         'sim_len': 24*365, #length of simulation in hours
                         'approx_time_pregnant': 300, #around day of simulation do the energy needs of the female bats change
                         'init_inf': 0, #initial number of bats inffected
                         'beta': 0.005, #transmission coefficient per contact (an hour of being in the same patch)
                         'gamma': 1.0/(10.0*24.0), #recover rate in hours
                         'stochastic_foraging': 0,
                         'bat_type': 0, #0 for fruit bat, 1 for insectivorous
                         'patch_pref': np.array([0,1])#'Orchard', 'Forest'])
                          }

#setting random seed for reproducibility
seed = float(os.environ['SLURM_ARRAY_TASK_ID'])
info = run_simulation(seed,patch_net,simulation_parameters)


with open('results/out_i1_' + str(int(seed)) + '.pkl', 'wb') as f:
    pickle.dump(info, f)
