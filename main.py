
import bats
import patches
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.animation import FuncAnimation
import time

def animation_func(i):
    global scats
    print(i)
    # first remove all old scatters
    for scat in scats:
        scat.remove()
    scats = []
    colors = ['red', 'orange', 'blue', 'green']
    for p in range(simulation_parameters['num_p']):
        bats_in_patch = bats.get_bats_in_patch(p)
        [x_vals, y_vals] = patches.get_random_locs_in_patch(p, bats_in_patch)
        scats.append(ax.scatter(x_vals, y_vals, c=colors[p], s=5))
    bats.update_bats(i)
    patches.update_patches(i)

def animation_func2(i):
    global scats
    print(i)
    # first remove all old scatters
    for scat in scats:
        scat.remove()
    scats = []
    for b in bats.bats['id']:
        if bats.bats['loc'][b]<4:
            [x_val, y_val] = patches.get_random_locs_in_patch(bats.bats['loc'][b], 1)
            scats.append(ax.scatter(x_val, y_val, c=color[b], s=50))
    plt.axvline(x=1)
    plt.axhline(y=1)
    plt.title(str(np.round(i/24,2))+" Days")
    bats.update_bats(i)
    patches.update_patches(i)

def animation_func3(i):
    bats.update_bats(i)
    patches.update_patches(i)
    bats.get_bat_density_map(i)
    plt.title(str(np.round(i / 24, 2)) + " Days")

#patch_net = {0: {'Name': 'Roost', 'carrying_capacity': 1, 'init_resources': 0, 'resource_birth': 0,
#                 'patch_center': [0.5,0.5], 'Res':np.array([0,0])},
#             1: {'Name': 'Residential', 'carrying_capacity': 1000, 'init_resources': 1000, 'resource_birth': 0.05,
#                 'patch_center': [1.5,0.5], 'Res':np.array([0,1])},
#             2: {'Name': 'Residential 2', 'carrying_capacity': 1000, 'init_resources': 1000, 'resource_birth': 0.05,
#                 'patch_center': [2.5,0.5], 'Res':np.array([0,1])},
#             3: {'Name': 'Orchard', 'carrying_capacity': 1000, 'init_resources': 1000, 'resource_birth': 0.01,
#                 'patch_center': [0.5,1.5], 'Res':np.array([1,0])},
#             4: {'Name': 'Orchard 2', 'carrying_capacity': 1000, 'init_resources': 1000, 'resource_birth': 0.01,
#                 'patch_center': [1.5,1.5],  'Res':np.array([1,0])},
#             5: {'Name': 'Dump', 'carrying_capacity': 500, 'init_resources': 100, 'resource_birth': 0.5,
#                 'patch_center': [2.5,1.5],  'Res':np.array([0,1])}}

with open('test_big_patch_net.pkl', 'rb') as f:
    patch_net = pickle.load(f)

#bats convert resources very quickly, but can depend on food type (nector vs mang vs hard fruit)


simulation_parameters = {'pop':30,#14000
                         'patch_types_options': ['Roost', 'Residential', 'Orchard', 'Water Body', 'Forest', 'Dump'],
                         'num_p': len(patch_net.keys()),
                         'grid_scale': [1,1],
                         'patch_type_forage_probs': np.array([0.0, 0.53710742, 0.26605321, 0.06561312, 0.06561312, 0.06561313]),
                         'sim_len': 24*5}

patches.initialize_patches(patch_net,simulation_parameters)
bats.initialize_bats(simulation_parameters)
for i in range(1,simulation_parameters['sim_len']):
    bats.update_bats(i)
    patches.update_patches(i)
bats.view_individual_bat_behavior([0])
plt.show()
#patches.observe_patches()
#bats.see_time_spent()
print("done")


#from random import randint
#color = []
#n = simulation_parameters['pop']
#for i in range(n):
#    color.append('#%06X' % randint(0, 0xFFFFFF))

#fig, ax = plt.subplots()
#ax.set_xlim(0, 2)
#ax.set_ylim(0, 2)
#scats = []
#animation = FuncAnimation(fig, animation_func2,interval = 500, frames = simulation_parameters['sim_len'])
#plt.show()

#animation = FuncAnimation(fig, animation_func3,interval = 100, frames = simulation_parameters['sim_len'])
#plt.show()