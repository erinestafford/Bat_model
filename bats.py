import random
import patches
import matplotlib.pyplot as plt
import numpy as np


def initialize_bats(simulation_parameters):
    global bats
    n = simulation_parameters['pop']

    bats = {'sp': simulation_parameters,
            'id': np.arange(n),
            'states': np.zeros(n),  # 0=roosting, 1=foraging, 2=traveling, 3=dead
            'ts': np.zeros(n),  # time_in_cur_state
            'tp': np.zeros(n),  # time_in_cur_patch
            'tt': np.zeros(n),  # time_traveling
            'rest_time': 12,#average daylight hours
            'not_dead': np.ones(n),
            'out_sim': np.zeros(n),
            'td': np.zeros(n),
            'energy': np.zeros((n, simulation_parameters['sim_len'])),
            'fr': 335.4166667,#foraging rate in kJ/hr
            'mr': 146.16, #metabolic rate -  https://besjournals.onlinelibrary.wiley.com/doi/epdf/10.1046/j.1365-2435.2003.00706.x (low end of flying)
            'tc': 146.16*3, #NOT SURE travel cost (flying metabolic rate)  https://besjournals.onlinelibrary.wiley.com/doi/epdf/10.1046/j.1365-2435.2003.00706.x
            'r_mr': 7.308,#resting metabolic rate 1/20th of normal mr -  https://besjournals.onlinelibrary.wiley.com/doi/epdf/10.1046/j.1365-2435.2003.00706.x
            'bat_resource_conversion': 0.75,
            'loc': np.zeros(n),
            'next_loc': np.zeros(n),
            'time_to_roost': np.zeros(n),
            'dnp': np.zeros(n),
            'th': np.zeros((n, simulation_parameters['sim_len'])),
            'fh': [],
            'daily_diet_hist':np.zeros((n, 2)), #hard fruit, soft fruit
            'diet_rec': np.array([2/3,1/3]),
            'roost_locs': np.zeros(n),
            'max_dist_in_hr': 50.0/30.0 #50 km at 30 km per hour
     }

    assign_bats_to_roost()
    for b_id in bats['id']:
        bats['fh'].append(get_initial_forage_hist(b_id))

    bats['energy'][:,0] = 200*np.ones(n)


def assign_bats_to_roost():
    global bats
    roost_options = patches.get_patch_type_ids('Roost')
    if len(roost_options)==1:
        bats['roost_locs'] = roost_options*np.ones(bats['sp']['pop'])
    else:
        bats['roost_locs'] = random.choices(roost_options, k=bats['sp']['pop'])

    bats['loc'] = np.copy(bats['roost_locs'])
    bats['th'][:, 0] = np.copy(bats['roost_locs'])

def get_initial_forage_hist(b_id):
    global bats
    roost = bats['roost_locs'][b_id]
    p_types = bats['sp']['patch_types_options']
    p_probs = bats['sp']['patch_type_forage_probs']

    #get patches in foraging range
    patches_in_range = patches.get_patches_in_range(roost, bats['max_dist_in_hr'])
    #get patch types of patches in foraging range
    patch_probs = np.zeros(bats['sp']['num_p'])
    num_of_type = np.zeros(len(p_types))
    for p in patches_in_range:
        type = patches.get_patch_type_by_id(p)
        for t in range(len(p_types)):
            if type == p_types[t]:
                num_of_type[t] += 1


    for p in range(len(patches_in_range)):
        type = patches.get_patch_type_by_id(p)
        for t in range(len(p_types)):
            if type == p_types[t]:
                patch_probs[p] = p_probs[t]/num_of_type[t]

    return random.choices(patches_in_range, weights=patch_probs,k=48)

def get_random_patch_in_range_with_resources(b_id,t):
    global bats
    loc = bats['loc'][b_id]
    p_types = bats['sp']['patch_types_options']
    p_probs = bats['sp']['patch_type_forage_probs']

    #get patches in foraging range
    patches_in_range = patches.get_patches_in_range(loc, bats['energy'][b_id,t]*bats['tc'])
    patch_options = np.asarray([p for p in patches_in_range if patches.get_patch_resources(p)>0])

    return random.choices(patch_options,k=1).pop()


def get_estars(b_id,t):
    global bats
    loc = bats['loc'][b_id]
    energy = bats['energy'][b_id][t-1]
    time_in_cur_patch = bats['tp'][b_id]
    p = patches.get_loc_in_patch(loc)
    estars = []
    for i in range(bats['sp']['num_p']):
        tnp = patches.get_time_to_next_point(p,i)
        if i != loc  and tnp<=3:
            estars.append((energy - bats['tc'] * tnp) / (tnp + time_in_cur_patch))
        else:
            estars.append(-np.inf)
    return estars


def get_p_patches(b_id):  # get patch to start foraging in. Later patches chosen based on MVT
    global bats
    prop_patch = np.zeros(bats['sp']['num_p'])
    forage_hist = bats['fh'][b_id]
    for i in range(bats['sp']['num_p']):
        prop_patch[i] = len(np.where(np.asarray(forage_hist) == i)[0])
        if patches.get_patch_resources(i) <= 0:
            prop_patch[i] = 0

    if sum(prop_patch) <= 0:
        prop_patch = np.asarray([patches.get_patch_resources(i) for i in range(bats['sp']['num_p'])])
    if sum(prop_patch) <= 0:
        prop_patch = np.ones(bats['sp']['num_p'])

    return prop_patch / sum(prop_patch)


def change_patch(b_id, new_loc):
    global bats
    loc = bats['loc'][b_id]
    bats['dnp'][b_id] = patches.get_time_to_next_patch(loc, new_loc)
    bats['states'][b_id] = 2
    bats['next_loc'][b_id] = new_loc
    bats['loc'][b_id] = bats['sp']['num_p']
    bats['tp'][b_id] = 0
    bats['time_to_roost'][b_id] = patches.get_time_to_next_patch(new_loc, bats['roost_locs'][b_id])


def get_patch_level_forage_hist(b_id):
    global bats
    forage_hist = bats['fh'][b_id]
    time_hist = np.asarray(forage_hist)
    patch_hist = np.zeros(bats['sp']['num_p'])
    for i in range( bats['sp']['num_p']):
        patch_hist[i]=len(time_hist[np.where(time_hist == i)])
    return patch_hist


def forage(b_id, t):
    global bats
    loc = bats['loc'][b_id]
    energy = bats['energy'][b_id][t-1]
    metabolic_rate = bats['mr']
    foraging_rate = bats['fr']
    bats['tp'][b_id] += 1
    bats['ts'][b_id] += 1
    bats['fh'][b_id].append(loc)
    bats['fh'][b_id].pop(0)
    bats['th'][b_id, t] = loc
    rec_conv = bats['bat_resource_conversion']
    avail_rec = patches.get_patch_resources(loc)

    if avail_rec>foraging_rate:
        e_temp =rec_conv * foraging_rate- metabolic_rate#* energy#rec_conv * foraging_rate*(1/time_in_cur_patch) - metabolic_rate * energy
        bats['daily_diet_hist'][b_id] += patches.get_patch_resource_types(loc) * foraging_rate
        patches.update_used_resources(loc, foraging_rate)

        bats['energy'][b_id][t] = energy + e_temp
        next_e_temp = rec_conv * foraging_rate - metabolic_rate#* energy#rec_conv * foraging_rate*(1/(time_in_cur_patch+1)) - metabolic_rate * energy
    else:
        e_temp = rec_conv*avail_rec-metabolic_rate #* energy
        bats['daily_diet_hist'][b_id] += patches.get_patch_resource_types(loc) * avail_rec
        patches.update_used_resources(loc, avail_rec)

        bats['energy'][b_id][t] = energy + e_temp
        next_e_temp = 0


    prop_necessary_res_consumed = bats['daily_diet_hist'][b_id]/bats['diet_rec']

    #change in next step? - if time
    if bats['ts'][b_id] < 24 - bats['rest_time'] - bats['time_to_roost'][b_id]:
        estars = get_estars(b_id,t)
        estar_i = np.argmax(estars)

        if next_e_temp < estars[estar_i]:
            p_options = np.arange(bats['sp']['num_p'])
            temp_estar = np.asarray(estars)
            p_options = p_options[np.where(temp_estar > next_e_temp)]
            p_resources = np.asarray([patches.get_patch_resources(i) for i in p_options]) #only go to patches with resources
            p_options = p_options[np.where(p_resources > 0)]
            patch_hist = get_patch_level_forage_hist(b_id)
            patch_hist = patch_hist[p_options]
            if sum(patch_hist) > 0:
                patch_probs = patch_hist / sum(patch_hist)
            else:
                patch_probs = np.ones(len(p_options)) / len(p_options)
            if np.any(prop_necessary_res_consumed>=1) and sum(bats['daily_diet_hist'][b_id])<1: #take diet variability into account
                enough_res = np.where(prop_necessary_res_consumed>=1)
                # increase probability of needed resource being consumed
                for i in range(len(p_options)):
                    res = patches.get_patch_resource_types(p_options[i])
                    if res[enough_res]>0 and res[np.where(prop_necessary_res_consumed<1)]==0:
                        patch_probs[i] = 0
                if sum(patch_probs) <= 0:
                    patch_probs = np.ones(len(p_options)) / len(p_options)
                else:
                    patch_probs = patch_probs / sum(patch_probs)

            if len(p_options) >= 1: #check case where only pathces without resources available
                p_choice = random.choices(p_options, weights=patch_probs, k=1).pop()
                if patches.patches['patch_type'][p_choice]=='Roost':
                    print('foraging at home')
                change_patch(b_id, p_choice)
    else:
        bats['ts'][b_id]=0
        travel_to_roost(b_id,t)


def roost(b_id, t):
    global bats
    bats['daily_diet_hist'][b_id] = np.array([0,0])
    bats['ts'][b_id] += 1
    bats['th'][b_id, t] = bats['loc'][b_id]
    bats['energy'][b_id,t] = bats['energy'][b_id,t-1] - bats['r_mr']#*bats['mr']* energy


def travel(b_id, new_loc, t):
    global bats
    bats['next_loc'][b_id] = new_loc
    bats['loc'][b_id] = bats['sp']['num_p']
    bats['ts'][b_id] += 1
    bats['th'][b_id, t] = bats['loc'][b_id]
    if bats['dnp'][b_id]>=1:
        e_temp = -bats['tc']
        bats['energy'][b_id][t] = bats['energy'][b_id][t - 1] + e_temp
        bats['dnp'][b_id] -= 1
    elif bats['dnp'][b_id]>0:
        e_temp = -bats['tc']*bats['dnp'][b_id]
        bats['energy'][b_id][t] = bats['energy'][b_id][t - 1] + e_temp
        bats['dnp'][b_id]  = 0
    else:
        bats['dnp'][b_id] = 0
        arrive(b_id, t)


def arrive(b_id, t):
    global bats
    bats['loc'][b_id] = bats['next_loc'][b_id]
    bats['dnp'][b_id] = 0
    if bats['loc'][b_id] == bats['roost_locs'][b_id]:
        bats['ts'][b_id] = 0
        bats['states'][b_id] = 0
        roost(b_id, t)
    else:
        bats['states'][b_id] = 1
        forage(b_id, t)


def get_resource_consumption(location):
    global bats
    return bats['fr'] * len(bats['loc'][(bats['loc'] == location) & (bats['states']==1)])


def get_all_bat_locations():
    global bats
    return bats['loc']

def travel_to_roost(b_id, t):
    global bats
    bats['ts'][b_id] = 0
    bats['dnp'][b_id] = bats['time_to_roost'][b_id]
    bats['states'][b_id] = 2
    bats['next_loc'][b_id] = bats['roost_locs'][b_id]
    bats['loc'][b_id] = bats['sp']['num_p']
    bats['time_to_roost'][b_id] = 0
    travel(b_id, bats['next_loc'][b_id], t)


def start_foraging(b_id, t):
    global bats
    bats['ts'][b_id] = 0
    bats['states'][b_id] = 2
    patch_probs = get_p_patches(b_id)
    if sum(bats['energy'][b_id][t-24:t])<0: #if not getting enough energy go to other patches
        for p in bats['fh'][b_id][24:]:
            patch_probs[int(p)] = 0
    if sum(patch_probs)==0:
        patch_choice= get_random_patch_in_range_with_resources(b_id,t)
    else:
        patch_probs = patch_probs/sum(patch_probs)
        patch_choice = random.choices(np.arange(bats['sp']['num_p']), patch_probs).pop()

    bats['dnp'][b_id] = patches.get_time_to_next_patch(int(bats['loc'][b_id]), patch_choice)
    bats['time_to_roost'][b_id] = patches.get_time_to_next_patch(int(patch_choice), bats['roost_locs'][b_id])
    bats['next_loc'][b_id] = patch_choice
    bats['loc'][b_id] = bats['sp']['num_p']
    travel(b_id,bats['next_loc'][b_id], t)


def update_bats(t):
    global bats
    # update foraging bats
    if t%24==0:
        [travel_to_roost(b, t) for b in bats['id'] if (bats['next_loc'][b]!= bats['roost_locs'][b])]
    else:
        foraging = bats['id'][np.where(bats['states'] == 1)]
        roosting = bats['id'][np.where(bats['states'] == 0)]
        traveling = bats['id'][np.where(bats['states'] == 2)]

        if len(foraging)>0:
            not_time_to_roost = bats['id'][np.where(bats['ts'] < 24 - bats['rest_time'] - bats['time_to_roost'])]
            keep_foraging = np.asarray([b_id for b_id in foraging if b_id in not_time_to_roost])
            [forage(b, t) for b in keep_foraging]
            done_foraging = np.asarray([b_id for b_id in foraging if b_id not in not_time_to_roost])
            [travel(b,bats['roost_locs'][b] , t) for b in done_foraging]

        if len(roosting) > 0:
            keep_roosting = [b_id for b_id in roosting if bats['ts'][b_id] < bats['rest_time']]
            [roost(b, t) for b in keep_roosting]
            done_roosting = [b for b in roosting if b not in keep_roosting]
            [start_foraging(b, t) for b in done_roosting]

        if len(traveling)>0:
            keep_traveling = [b_id for b_id in traveling if bats['dnp'][b_id] > 0]
            [travel(b, bats['next_loc'][b], t) for b in keep_traveling]
            done_traveling = [b for b in traveling if b not in keep_traveling]
            [arrive(b, t) for b in done_traveling]



def get_bats_in_patch(p):
    global bats
    return len(bats['id'][bats['loc']==p])

def view_individual_bat_behavior(b_ids):
    global bats
    p_types = bats['sp']['patch_types_options']

    for b in b_ids:
        temp = np.asarray(bats['th'][b])
        type_list = np.zeros(len(temp))
        for t in range(len(temp)):
            if temp[t] == bats['sp']['num_p']:
                type_list[t] = len(p_types)
            else:
                for p in range(len(p_types)):
                    if patches.get_patch_type_by_id(int(temp[t])) == p_types[p]:
                        type_list[t] = p
        plt.plot(np.arange(bats['sp']['sim_len'])/24, type_list)
    plt.yticks(np.arange(len(bats['sp']['patch_types_options'])+1), bats['sp']['patch_types_options']+['Traveling'])
    plt.xlabel("Days")
    plt.show()

def see_time_spent():
    global bats
    p_types = bats['sp']['patch_types_options']
    time_props = np.zeros((len(bats['id']),len(p_types)))

    for b in bats['id']:
        temp = np.asarray(bats['th'][b])
        temp_types = np.asarray([patches.get_patch_type_by_id(int(p)) for p in temp if p<bats['sp']['num_p']])
        time_list = np.zeros(len(p_types))
        for t in temp_types:
            for p in range(len(p_types)):
                if t==p_types[p]:
                    time_list[p]+=1
        time_props[b, :] = time_list/ len(temp)
    mean_times = np.mean(time_props, 0)
    plt.scatter(np.arange(len(p_types)), mean_times, color='r')
    foraging_places = np.zeros(len(p_types))
    foraging_places[1:] = mean_times[1:] / sum(mean_times[1:])
    plt.scatter(np.arange(len(p_types)), foraging_places, color='b')
    plt.xticks(np.arange(len(p_types)), p_types)
    plt.legend(['total', 'foraging'])
    plt.ylabel('Proportion of time spent in patch')
    plt.show()
    print(foraging_places)

def get_bat_density_map(t):
    global bats
    bat_mat = np.zeros(bats['sp']['num_p'])
    for p in range(bats['sp']['num_p']):
        bat_mat[p]=len(bats['th'][:, t][np.where(bats['th'][:, t]==p)])
    plt.imshow(bat_mat.reshape((2,3)))



