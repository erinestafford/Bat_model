
import bats
import test_bat
import patches
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random


with open('test_big_patch_net.pkl', 'rb') as f:
    patch_net = pickle.load(f)

#with open('test_small_synth_patch_net.pkl', 'rb') as f:
#    patch_net = pickle.load(f)


simulation_parameters = {'pop':100,#14000
                         'patch_types_options': ['Roost', 'Residential', 'Orchard', 'Water Body', 'Forest', 'Dump'],
                         'num_p': len(patch_net.keys()),
                         'gp':[0.5,0.5],
                         'grid_scale': [1,1],
                         'patch_type_forage_probs': np.array([  0.,  62.,   2.,   4., 214.,  35.])/sum(np.array([  0.,  62.,   2.,   4., 214.,  35.])),#np.array([0.0, 0.53710742, 0.26605321, 0.06561312, 0.06561312, 0.06561313]),
                         'sim_len': 24*90,
                         'approx_time_pregnant': 1, #which day of simulation
                         'pregnancy_duration': 5*30*24, #5 months of pregnancy
                         'time_to_rear_young': 5*30*24 #5 months before self feeding
                          }

random.seed(1)

patches.initialize_patches(patch_net,simulation_parameters)
test_bat.initialize_bats(simulation_parameters)

for i in range(1,simulation_parameters['sim_len']):
    test_bat.update_bats(i)
    patches.update_patches(i)
#test_bat.bat_on_grid_over_time([0,2,3,4,5,6,7,8,9,10], [21,23])
#test_bat.view_individual_bat_behavior([0],0,simulation_parameters['sim_len'])
#plt.show()
test_bat.bat_on_grid_over_time(list(np.arange(simulation_parameters['pop'])), [23*2,21*2])
patches.rec_grid_over_time([23*2,21*2])
