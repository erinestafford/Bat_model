
import bats
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
                         'grid_scale': [1,1],
                         'patch_type_forage_probs': np.array([0.0, 0.53710742, 0.26605321, 0.06561312, 0.06561312, 0.06561313]),
                         'sim_len': 24*30}

patches.initialize_patches(patch_net,simulation_parameters)
bats.initialize_bats(simulation_parameters)

random.seed(1)
for i in range(1,simulation_parameters['sim_len']):
    bats.update_bats(i)
    patches.update_patches(i)
bats.view_individual_bat_behavior([0])
plt.show()
