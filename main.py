# Main file for running the ABM
import test_bat
import patches
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import time


# Example using some of the Australia data
with open('test_big_patch_net.pkl', 'rb') as f:
    patch_net = pickle.load(f)

#example on small synthetic grid
#with open('small_patch_prob_ex.pkl', 'rb') as f:
#    patch_net = pickle.load(f)


#parameters to control simulation
simulation_parameters = {'pop':1000, #bat population
                         'patch_types_options': ['Roost', 'Residential', 'Orchard', 'Water Body', 'Forest', 'Dump'], #types of patches in grid
                         'num_p': len(patch_net.keys()), #number of grid patches
                         'gp':[0.5,0.5], #gender distribution of bat population ([male, female])
                         'grid_scale': [1,1], #scale of grid patches ([width,height]) in kilometers
                         'patch_type_forage_probs': np.array([0., 0.1955836, 0.00630915, 0.0126183, 0.67507886,
                                                              0.11041009]), #initialization parameter for probability of foraging in each patch type
                         'sim_len': 24*30, #length of simulation in hours
                         'approx_time_pregnant': 1, #around day of simulation do the energy needs of the female bats change
                         'init_inf': 1, #initial number of bats inffected
                         'beta': 0.005, #transmission coefficient per contact (an hour of being in the same patch)
                         'gamma': 1.0/(10.0*24.0) #recover rate in hours
                          }

#setting random seed for reproducibility
random.seed(5)

#initializing patches and bats
patches.initialize_patches(patch_net,simulation_parameters)
test_bat.initialize_bats(simulation_parameters)

t1 = time.time() #start time of simulation

#update bats and grid for each time step
for i in range(1,simulation_parameters['sim_len']):
    test_bat.update_bats(i)
    patches.update_patches(i)

#end time of simulation
t2 = time.time()
print(t2-t1)

# shows the movement of listed bats to different patch types over the simulation
test_bat.view_individual_bat_behavior([0],0,simulation_parameters['sim_len'])
plt.show()

#determine probabilities of going to each patch from other patch types
test_ind = []
test_lab = np.array(patches.patches['patch_type'])
movement_dicts = {'Roost':np.zeros(6),'Residential':np.zeros(6), 'Orchard':np.zeros(6), 'Water Body':np.zeros(6), 'Forest': np.zeros(6), 'Dump':np.zeros(6)}
for j in range(simulation_parameters['pop']):
    test_ind=[i for i in test_bat.bats['th'][j] if i<simulation_parameters['num_p']]
    test_ind = np.array(test_ind).astype(int)
    test_hist = test_lab[test_ind]
    for i in range(len(test_ind) - 1):
        if test_ind[i] != test_ind[i + 1]:
            ind = np.where(test_hist[i + 1] == np.array(simulation_parameters['patch_types_options']))
            movement_dicts[test_hist[i]][ind] += 1
for k in movement_dicts.keys():
    movement_dicts[k] = movement_dicts[k]/sum(movement_dicts[k])

print(movement_dicts)

