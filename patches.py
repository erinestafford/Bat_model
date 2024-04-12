import bats
import matplotlib.pyplot as plt
import numpy as np
import random

class patch:
    pass

def initialize_patches(patch_net,simulation_parameters):  # need to input patch network
    global patches
    num_p = len(patch_net.keys())
    patches = {'id':np.arange(num_p),
               'resources': np.asarray([patch_net[i]['init_resources'] for i in range(num_p)]),
               'resource_regen_rate': np.asarray([patch_net[i]['resource_birth'] for i in range(num_p)]),
               'carrying_capacity': np.asarray([patch_net[i]['carrying_capacity'] for i in range(num_p)]),
               'resource_history': np.zeros((num_p,simulation_parameters['sim_len'])),
               'grid_scale':simulation_parameters['grid_scale'], #in km
               'patch_coord': np.asarray([patch_net[i]['patch_center'] for i in range(num_p)]),
               'patch_type':[patch_net[i]['Name'] for i in range(num_p)],
               'num_p': num_p,
               'res_type':[patch_net[i]['Res'] for i in range(num_p)],
               'color_map':{ 'Roost': np.array([255, 255, 255]),
             'Orchard': np.array([172, 188, 45]),
             'Forest': np.array([14, 121, 18]),
             'Forest': np.array([30, 191, 121]),
             'Residential': np.array([218, 92, 105]),
             'Dump': np.array([243, 171, 105]),
             'Water Body': np.array([77, 159, 220]),
}
                }
    patches['max_rec']=np.max(patches['resources'])
    for i in patch_net.keys():
        patches['resource_history'][i,0] = patches['resources'][i]

def update_patches(t):
    global patches
    new_resources = patches['resources'] * (patches['resource_regen_rate'] * (1 - patches['resources'] / patches['carrying_capacity']))
    patches['resources'] = patches['resources'] + new_resources
    if len(patches['resources'][np.where(patches['resources'] < 0)])>0:
        print('here')
    patches['resource_history'][:,t] = patches['resources']

def get_patch_names():
    global patches
    return patches['patch_type']

def get_patch_resources(patch_id):
    global patches
    return patches['resources'][int(patch_id)]

def get_patch_find_rec_prob(patch_id):
    global patches
    if patches['resources'][int(patch_id)]>np.mean(patches['resources']):
        return 1
    else:
        return np.exp(-patches['resources'][int(patch_id)]/np.mean(patches['resources']))


def get_patch_resource_types(patch_id):
    global patches
    return patches['res_type'][int(patch_id)]

def update_used_resources(patch_id, used_resources):
    global patches
    patches['resources'][int(patch_id)] -=used_resources

def get_time_to_next_patch(p1,p2):
    center1 = patches['patch_coord'][int(p1)]
    center2 = patches['patch_coord'][int(p2)]
    #randomly pick a location withing each grid space
    x1 = np.random.choice(np.linspace(center1[0]-patches['grid_scale'][0]/2,center1[0]+patches['grid_scale'][0]/2,20))
    x2 = np.random.choice(np.linspace(center2[0]-patches['grid_scale'][0]/2,center2[0]+patches['grid_scale'][0]/2,20))

    y1 = np.random.choice(np.linspace(center1[1] - patches['grid_scale'][1]/2, center1[1] + patches['grid_scale'][1]/2, 20))
    y2 = np.random.choice(np.linspace(center2[1] - patches['grid_scale'][1]/2, center2[1] + patches['grid_scale'][1]/2, 20))

    #get distance in hours between those points - https://en.wikipedia.org/wiki/Pteropus
    return (np.sqrt((x2-x1)**2 + (y2-y1)**2))/30


def get_loc_in_patch(p):
    center= patches['patch_coord'][int(p)]
    #randomly pick a location withing each grid space
    x = np.random.choice(np.linspace(center[0]-patches['grid_scale'][0]/2,center[0]+patches['grid_scale'][0]/2,20))
    y = np.random.choice(np.linspace(center[1] - patches['grid_scale'][1]/2, center[1] + patches['grid_scale'][1]/2, 20))

    #get distance in hours between those points - https://en.wikipedia.org/wiki/Pteropus
    return [x,y]

def get_time_to_next_point(p1,p2):
    center2 = patches['patch_coord'][int(p2)]
    #randomly pick a location withing each grid space
    x1 = p1[0]
    x2 = np.random.choice(np.linspace(center2[0]-patches['grid_scale'][0]/2,center2[0]+patches['grid_scale'][0]/2,20))

    y1 = p1[1]
    y2 = np.random.choice(np.linspace(center2[1] - patches['grid_scale'][1]/2, center2[1] + patches['grid_scale'][1]/2, 20))

    #get distance in hours between those points - https://en.wikipedia.org/wiki/Pteropus
    return (np.sqrt((x2-x1)**2 + (y2-y1)**2))/30

def get_time_between_points(p1,p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    return (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)) / 30

def get_random_locs_in_patch(p_id, num):
    center = patches['patch_coord'][int(p_id)]
    x_vals = random.choices(np.linspace(center[0] - patches['grid_scale'][0]/2, center[0] + patches['grid_scale'][0]/2, 20), k=num)
    y_vals = random.choices(np.linspace(center[1] - patches['grid_scale'][1]/2, center[1] + patches['grid_scale'][1]/2, 20), k=num)
    return [x_vals,y_vals]

def observe_patch(p_id):
    global patches
    plt.plot(np.arange(len(patches['resource_history'][p_id,:]))/24, patches['resource_history'][p_id,:])
    plt.xlabel('Days')
    plt.ylabel('Patch Resources')
    #plt.legend(patches['patch_type'][1:])
    plt.show()


def get_patch_type_ids(type):
    global patches
    inds = []
    count = 0
    for p in patches['patch_type']:
        if p==type:
            inds.append(count)
        count=count+1
    return inds

def get_patch_type_by_id(p_id):
    global patches
    return patches['patch_type'][p_id]

def get_patches_in_range(p_id, max_dist):
    #get center of patch p_id
    center = patches['patch_coord'][int(p_id)]
    x1 = center[0]
    y1 = center[1]
    #get all patches whose centers are with in max_dist
    patches_in_range=[]
    for p in patches['id']:
        center = patches['patch_coord'][int(p)]
        x2 = center[0]
        y2 = center[1]
        dist = (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)) / 30
        if dist <= max_dist:
            patches_in_range.append(p)
    return np.asarray(patches_in_range)
