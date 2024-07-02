import test_bat
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.animation as animation
import matplotlib.cm as cm
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

class patch:
    pass

def initialize_patches(patch_net,simulation_parameters):
    # initializing patch dictionary using patch network
    global patches
    num_p = len(patch_net.keys())
    patches = {'id':np.arange(num_p), #array of patch ids that are used for indexing
               'resources': np.asarray([patch_net[i]['init_resources'] for i in range(num_p)]), #stores current patch resources
               'resource_regen_rate': np.asarray([patch_net[i]['resource_birth'] for i in range(num_p)]), #stores resource regeneration rate for each patch
               'yearly_cc': np.asarray([patch_net[i]['carrying_capacity_yr'] for i in range(num_p)]), #yearly carrying capacity of patch
               'carrying_capacity': np.zeros((num_p,simulation_parameters['sim_len'])), #section of yearly_cc relevant to our simulation
               'resource_history': np.zeros((num_p,simulation_parameters['sim_len'])), #stores patch resources over the whole simulation
               'grid_scale':simulation_parameters['grid_scale'], #in km
               'patch_coord': np.asarray([patch_net[i]['patch_center'] for i in range(num_p)]), #patch location on grid
               'patch_type':[patch_net[i]['Name'] for i in range(num_p)], #type of each patch forest, water body, residential, etc.
               'num_p': num_p, #number of patches on grid
               'res_type':[patch_net[i]['Res'] for i in range(num_p)], #type of resources in each patch (not used now, but could be used to assign nutritional value)
               'color_map':{ 'Roost': np.array([255, 255, 255]),
                             'Orchard': np.array([172, 188, 45]),
                             'Forest': np.array([30, 191, 121]),
                             'Residential': np.array([218, 92, 105]),
                             'Dump': np.array([243, 171, 105]),
                             'Water Body': np.array([77, 159, 220])}, #for plotting
               'sp': simulation_parameters, #storing the overall simulationn parameters
               'tbp': np.zeros((num_p,num_p))#time between patches
                }
    #getting carying capacity for just the simulation time frame
    assign_seasonal_cc()
    #initializing resource history and pre-calculating time between patches
    for i in patch_net.keys():
        patches['resource_history'][i,0] = patches['resources'][i]
        patches['tbp'][i,:] = get_time_to_other_points(i)

def assign_seasonal_cc():
    # getting carying capacity for just the simulation time frame
    global patches
    temp = np.tile(patches['yearly_cc'],patches['sp']['sim_len']//365*24 + 1)
    patches['carrying_capacity'] = temp[:,0:patches['sp']['sim_len']]


def update_patches(t):
    #update  patch resources. Resources consumed by bats are removed in the bat class
    global patches
    new_resources = patches['resources'] * (patches['resource_regen_rate'] * (1 - patches['resources'] / patches['carrying_capacity'][:,t]))
    patches['resources'] = patches['resources'] + new_resources
    if len(patches['resources'][np.where(patches['resources'] < 0)])>0:
        patches['resources'][np.where(patches['resources'] < 0)] = 0
    patches['resource_history'][:,t] = patches['resources']

def get_patch_names():
    global patches
    return patches['patch_type']

def get_patch_resources(patch_id):
    global patches
    return patches['resources'][patch_id.astype(int)]

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
    patches['resources'][patch_id.astype(int)] =patches['resources'][patch_id.astype(int)]-used_resources

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


def get_loc_in_patch(p_ids):
    #randomly pick a location withing each grid space
    xs = []
    ys = []
    for p in p_ids:
        if p < patches['num_p']:
            center=patches['patch_coord'][p.astype(int)]
            xs.append(random.choices(np.linspace(center[0] - patches['grid_scale'][0] / 2, center[0] + patches['grid_scale'][0] / 2, 100)).pop())
            ys.append(random.choices(np.linspace(center[1] - patches['grid_scale'][0] / 2, center[1] + patches['grid_scale'][0] / 2, 100)).pop())
        else:
            xs.append(-5)
            ys.append(-5)

    #get distance in hours between those points - https://en.wikipedia.org/wiki/Pteropus
    return np.array([xs,ys])

def get_time_to_other_points(p_id): #TODO: doing using center and removing stochasticity for speed
    p = patches['patch_coord'][p_id]
    x = p[0]
    y = p[1]
    centers_xs = patches['patch_coord'][:,0]
    centers_ys = patches['patch_coord'][:,1]
    #other_xs = np.zeros(len(centers_xs))
    #other_ys = np.zeros(len(centers_ys))
    #for i in range(len(patches['id'])):
    #    other_xs[i] = random.choice(np.linspace(centers_xs[i]-patches['grid_scale'][0]/2,centers_xs[i]+patches['grid_scale'][0]/2,20))
    #    other_ys[i] = random.choice(np.linspace(centers_ys[i] - patches['grid_scale'][0] / 2, centers_ys[i] + patches['grid_scale'][0] / 2, 20))

    return (np.sqrt((centers_xs - x) ** 2 + (centers_ys - y) ** 2)) / 30.0#test_bat.bats['avg_speed']#(np.sqrt((other_xs - x) ** 2 + (other_ys - y) ** 2)) / test_bat.bats['avg_speed']

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
def get_patches_in_smell_range(p_ids, loc):
    #patches.get_patches_in_smell_range(temp, bats['loc'][to_switch[k]]) out of options which in range
    #get center of patch p_id
    in_range=np.zeros(len(p_ids))
    centers = patches['patch_coord'][p_ids.astype(int)]
    xs = centers[:,0]
    ys = centers[:,1]
    cur_center=patches['patch_coord'][int(loc)]
    x = cur_center[0]
    y = cur_center[1]
    dists = np.sqrt((xs-x)**2 + (ys-y)**2)
    max_dist=test_bat.bats['smell_dist']
    in_range[np.where(dists <= max_dist)] = 1
    return in_range
    
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


def rec_grid_over_time(d):
    global patches, rec, title, r,c
    r= d[0]
    c = d[1]

    fig, ax = plt.subplots()
    ax.set(xlim=[0, r], ylim=[0, c])
    ax.grid(which='major', alpha=0.5)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    rec = ax.imshow(patches['resource_history'][:,0].reshape(c,r))
    title = ax.text(0.5,1, "",transform=ax.transAxes, ha="center")

    ani = animation.FuncAnimation(fig=fig, func=lambda frame: animation_update(frame), frames=patches['sp']['sim_len'], interval=1)
    writervideo = animation.FFMpegWriter(fps=60)
    ani.save('test_patches.mp4', writer=writervideo)
    plt.close()
    #plt.show()

def animation_update(frame):
    global rec,title,r,c
    rec.set_array(patches['resource_history'][:, frame].reshape(c, r))
    title.set_text(np.round(frame/24,2))
    return rec
