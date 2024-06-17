
import bats
import test_bat
import patches
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random


#with open('test_big_patch_net.pkl', 'rb') as f:
#    patch_net = pickle.load(f)

with open('small_patch_prob_ex.pkl', 'rb') as f:
    patch_net = pickle.load(f)


simulation_parameters = {'pop':100,#14000
                         'patch_types_options': ['Roost', 'Residential', 'Orchard', 'Water Body', 'Forest', 'Dump'],
                         'num_p': len(patch_net.keys()),
                         'gp':[0.5,0.5],
                         'grid_scale': [1,1],
                         'patch_type_forage_probs': np.array([  0.,  62.,   2.,   4., 214.,  35.])/sum(np.array([  0.,  62.,   2.,   4., 214.,  35.])),#np.array([0.0, 0.53710742, 0.26605321, 0.06561312, 0.06561312, 0.06561313]),
                         'sim_len': 24*30,
                         'approx_time_pregnant': 1, #which day of simulation
                         'pregnancy_duration': 5*30*24, #5 months of pregnancy
                         'time_to_rear_young': 5*30*24 #5 months before self feeding
                          }

random.seed(5)

patches.initialize_patches(patch_net,simulation_parameters)
test_bat.initialize_bats(simulation_parameters)
#for i in range(1,simulation_parameters['sim_len']):
#    test_bat.update_bats(i)
#    patches.update_patches(i)
#    if test_bat.bats['states'][0]==1:
#        estars = test_bat.get_estars([0], i)
#        max_estars = np.max(estars, 1)
#        all_fr = 2 * test_bat.bats['fr'][:, i] / (1 + 0.25 ** (-test_bat.bats['energy'][:, i - 1] / 5000))
#        avail_rec_next = patches.get_patch_resources(test_bat.bats['loc'])
#        foraged_rec_next = min([all_fr[0], avail_rec_next[0]])
#        other_bats_in_loc=0
#        next_e_temp = test_bat.bats['bat_resource_conversion'] * (foraged_rec_next * test_bat.bats['fc']) * np.exp(-test_bat.bats['e_discount'] * test_bat.bats['tp'][0] - test_bat.bats['e_discount'] * other_bats_in_loc) - test_bat.bats['mr'][:, i]
#        if_switch = max_estars - next_e_temp[0]
#        if if_switch > 0:
#            estar_switch = estars[np.where(if_switch > 0), :] - next_e_temp[np.where(if_switch > 0)]
#            estar_switch = estar_switch.reshape(test_bat.bats['sp']['num_p'])
#            p_options = []
#            temp = np.where(estar_switch> 0)[0]
#            if len(temp) > 0 and np.max(patches.patches['resources'][temp]) > 0:
#                p_op_rec = patches.patches['resources'][temp]
#                p_op_smell = patches.get_patches_in_smell_range(temp, test_bat.bats['loc'][0])
#               w_temp = p_op_rec / sum(p_op_rec)
#                w_temp[np.where(p_op_smell > 0)] = w_temp[np.where(p_op_smell > 0)] * 10
#                w_temp = w_temp / sum(w_temp)
#                test = np.zeros(20 * 20)
#                test[temp] = w_temp
#                plt.imshow(test.reshape(20, 20))
#                plt.colorbar()

for i in range(1,simulation_parameters['sim_len']):
    test_bat.update_bats(i)
    patches.update_patches(i)
#test_bat.bat_on_grid_over_time([0,2,3,4,5,6,7,8,9,10], [21,23])
#test_bat.view_individual_bat_behavior([0],0,simulation_parameters['sim_len'])
#plt.show()

#determine probabilities of going to each patch from other patch types
test_ind = []
test_lab = np.array(patches.patches['patch_type'])
movement_dicts = {'Roost':np.zeros(6),'Residential':np.zeros(6), 'Orchard':np.zeros(6), 'Water Body':np.zeros(6), 'Forest': np.zeros(6), 'Dump':np.zeros(6)}
for j in range(simulation_parameters['pop']):
    test_ind=[i for i in test_bat.bats['th'][j] if i<400]
    test_ind = np.array(test_ind).astype(int)
    test_hist = test_lab[test_ind]
    for i in range(len(test_ind) - 1):
        if test_ind[i] != test_ind[i + 1]:
            ind = np.where(test_hist[i + 1] == np.array(simulation_parameters['patch_types_options']))
            movement_dicts[test_hist[i]][ind] += 1
for k in movement_dicts.keys():
    movement_dicts[k] = movement_dicts[k]/sum(movement_dicts[k])

movement_dicts
#test_bat.bat_on_grid_over_time(list(np.arange(simulation_parameters['pop'])), [23*2,21*2])
#patches.rec_grid_over_time([23*2,21*2])
