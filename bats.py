import random
import patches
import matplotlib.pyplot as plt
import numpy as np


def initialize_bats(simulation_parameters):
    global bats, sim_time
    n = simulation_parameters['pop']
    sim_time = 1

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
            'energy_cp': np.zeros(n),#energy ggained from current patch
            'e_discount':0.01,
            'fr': 29.16666667*np.ones(n),#foraging rate in g/hr
            'fc': 11.5, #food to energy conversion in kJ/g
            'mr': 146.16*np.ones(n), #metabolic rate -  https://besjournals.onlinelibrary.wiley.com/doi/epdf/10.1046/j.1365-2435.2003.00706.x (low end of flying)
            'tc': 146.16*2*np.ones(n), #NOT SURE travel cost (flying metabolic rate)  https://besjournals.onlinelibrary.wiley.com/doi/epdf/10.1046/j.1365-2435.2003.00706.x
            'r_mr': 30*np.ones(n),#7.308,#resting metabolic rate 1/20th of normal mr -  https://besjournals.onlinelibrary.wiley.com/doi/epdf/10.1046/j.1365-2435.2003.00706.x
            'bat_resource_conversion': 0.75,
            'loc': np.zeros(n),
            'next_loc': np.zeros(n),
            'next_state': np.zeros(n),
            'time_to_roost': np.zeros(n),
            'dnp': np.zeros(n),
            'th': np.zeros((n, simulation_parameters['sim_len'])),
            'fh': [],
            'daily_diet_hist':np.zeros((n, 2)), #hard fruit, soft fruit
            'diet_rec': np.array([2/3,1/3]),
            'roost_locs': np.zeros(n),
            'max_dist_in_hr': (50.0/30.0)*np.ones(n), #50 km at 30 km per hour
            'max_food_before_roost': 29.16666667*11.5*5, #3 foraging trips
            'food_before_roost': np.zeros(n),
            'smell_dist':10.0/30.0, #max smell dist in hr
            'avg_speed':30.0,
            'gender': np.zeros(n) #0 -> Male, 1 -> female
     }

    assign_bats_to_roost()
    assign_gender()
    for b_id in bats['id']:
        bats['fh'].append(get_initial_forage_hist(b_id))

    bats['energy'][:,0] = 1000*np.ones(n)


def assign_gender():
    global bats
    pm = bats['sp']['gp'][0]
    pf = bats['sp']['gp'][1]
    bats['gender'] = random.choices([0.0,1.0],weights=[pm,pf],k = bats['sp']['pop'])
    #fr, r_mr, mr,max_dist_in_hr
    #increasing these for females
    bats['fr'] += bats['gender']*bats['fr']
    #bats['r_mr'] += bats['gender']*bats['r_mr']/2
    #bats['mr'] += bats['gender']* bats['mr']/2

    #decreasing this for females
    bats['max_dist_in_hr'] -=bats['gender']*((40/30)*np.ones(bats['sp']['pop']))

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
    patches_in_range = patches.get_patches_in_range(roost, bats['max_dist_in_hr'][b_id])

    #get patch types of patches in foraging range
    patch_probs = np.zeros(len(patches_in_range))
    num_of_type = np.zeros(len(p_types))
    for p in patches_in_range:
        type = patches.get_patch_type_by_id(p)
        if type == 'Roost':#new
            type = 'Forest'
        for t in range(len(p_types)):
            if type == p_types[t]:
                num_of_type[t] += 1


    for p in range(len(patches_in_range)):
        type = patches.get_patch_type_by_id(p)
        if type == 'Roost':#new
            type = 'Forest'
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
    patches_in_range = patches.get_patches_in_range(loc, bats['max_dist_in_hr'][b_id])
    patch_options = np.asarray([p for p in patches_in_range if patches.get_patch_resources(p)>0])
    patch_rec=patches.patches['resources'][patch_options]
    return random.choices(patch_options,k=1, weights = patch_rec/np.sum(patch_rec)).pop()


def get_estars(b_id,t):
    global bats
    loc = bats['loc'][b_id]
    energy = bats['energy_cp'][b_id]
    time_in_cur_patch = bats['tp'][b_id]
    p = patches.get_loc_in_patch(loc)
    estars = []
    for i in range(bats['sp']['num_p']):
        tnp = patches.get_time_to_next_point(p,i)
        if i != loc  and tnp<=3:
            estars.append((energy - bats['tc'][b_id] * tnp) / (tnp + time_in_cur_patch))
        else:
            estars.append(-np.inf)
    return estars


def get_p_patches(b_id):  # get patch to start foraging in. Later patches chosen based on MVT
    global bats
    patch_rec = patches.patches['resources']
    patch_prob = patch_rec/sum(patch_rec)

    visit_patch = np.zeros(bats['sp']['num_p'])
    forage_hist = bats['fh'][b_id]
    for i in range(bats['sp']['num_p']):
        visit_patch[i] = len(np.where(np.asarray(forage_hist) == i)[0])

    patch_prob[np.where(visit_patch>0)] =patch_prob[np.where(visit_patch>0)]*visit_patch[np.where(visit_patch>0)]
    #get bats per patch
    b_per_p = [get_bats_in_patch(int(p)) for p in range(bats['sp']['num_p'])]
    patch_prob = patch_prob*(1-(1/bats['sp']['pop'])*np.ones(len(b_per_p))*b_per_p)
    patch_prob=patch_prob/np.sum(patch_prob)

    return patch_prob


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
    metabolic_rate = bats['mr'][b_id]
    foraging_rate = bats['fr'][b_id]*bats['fc']

    bats['fh'][b_id].append(loc)
    bats['fh'][b_id].pop(0)
    bats['th'][b_id, t] = loc
    rec_conv = bats['bat_resource_conversion']
    avail_rec = patches.get_patch_resources(loc)

    #food_p = patches.get_patch_find_rec_prob(loc)
    found_food = 1#random.choices([0, 1], weights=[1 - food_p, food_p]).pop()

    if avail_rec>=bats['fr'][b_id]:
        e_temp =found_food *rec_conv * foraging_rate*np.exp(-bats['e_discount']*bats['tp'][b_id])- metabolic_rate#* energy#rec_conv * foraging_rate*(1/time_in_cur_patch) - metabolic_rate * energy
        #bats['daily_diet_hist'][b_id] += patches.get_patch_resource_types(loc) * foraging_rate
        bats['energy_cp'][b_id] += e_temp
        patches.update_used_resources(loc, bats['fr'][b_id])
        bats['energy'][b_id][t] = energy + e_temp
        bats['food_before_roost'][b_id] += foraging_rate


        if patches.get_patch_resources(loc) >= bats['fr'][b_id]:
            next_e_temp = found_food*rec_conv * foraging_rate*np.exp(-bats['e_discount']*(bats['tp'][b_id]+1)) - metabolic_rate#* energy#rec_conv * foraging_rate*(1/(time_in_cur_patch+1)) - metabolic_rate * energy
        else:
            next_e_temp = found_food*rec_conv*patches.get_patch_resources(loc)*bats['fc']*np.exp(-bats['e_discount']*(bats['tp'][b_id]+1))-metabolic_rate
    else:
        e_temp = found_food*rec_conv*avail_rec*bats['fc']*np.exp(-bats['e_discount']*(bats['tp'][b_id]))-metabolic_rate #* energy
        #bats['daily_diet_hist'][b_id] += patches.get_patch_resource_types(loc) * avail_rec*bats['fc']
        bats['energy_cp'][b_id] += e_temp
        patches.update_used_resources(loc, avail_rec)

        bats['energy'][b_id][t] = energy + e_temp
        bats['food_before_roost'][b_id] += avail_rec
        next_e_temp = 0

    bats['tp'][b_id] += 1
    bats['ts'][b_id] += 1
    #prop_necessary_res_consumed = bats['daily_diet_hist'][b_id]/bats['diet_rec']

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
            p_resources = np.asarray([patches.get_patch_resources(i) for i in p_options])  # only go to patches with resources

            #using patch history
            patch_hist = get_patch_level_forage_hist(b_id)
            patch_hist = patch_hist[p_options]

            #using smell dist
            patches_in_range = patches.get_patches_in_range(loc, bats['smell_dist'])

            p_rec = p_resources/np.sum(p_resources)
            p_rec[np.where(patch_hist>0)] =p_rec[np.where(patch_hist>0)]*patch_hist[np.where(patch_hist>0)]

            p_rec[np.nonzero(np.in1d(p_options,patches_in_range))] = p_rec[np.nonzero(np.in1d(p_options,patches_in_range))] * 10
            patch_probs = p_rec/np.sum(p_rec)

            if len(p_options) >= 1: #check case where only pathces without resources available
                p_choice = random.choices(p_options, weights=patch_probs, k=1).pop()
                change_patch(b_id, p_choice)
    else:
        bats['ts'][b_id]=0
        travel_to_roost(b_id,t)


def roost(b_id, t):
    global bats
    bats['daily_diet_hist'][b_id] = np.array([0,0])
    bats['ts'][b_id] += 1
    bats['th'][b_id, t] = bats['loc'][b_id]
    bats['energy'][b_id,t] = bats['energy'][b_id,t-1] - bats['r_mr'][b_id]#*bats['mr']* energy
    bats['food_before_roost'][b_id]=0


def travel(b_id, new_loc, t):
    global bats
    bats['energy_cp'][b_id] = 0
    bats['next_loc'][b_id] = new_loc
    bats['loc'][b_id] = bats['sp']['num_p']
    bats['ts'][b_id] += 1
    bats['th'][b_id, t] = bats['loc'][b_id]
    if bats['dnp'][b_id]>=1:
        e_temp = -bats['tc'][b_id]
        bats['energy'][b_id][t] = bats['energy'][b_id][t - 1] + e_temp
        bats['dnp'][b_id] -= 1
    elif bats['dnp'][b_id]>0:
        e_temp = -bats['tc'][b_id]*bats['dnp'][b_id]
        bats['energy'][b_id][t] = bats['energy'][b_id][t - 1] + e_temp
        bats['dnp'][b_id]  = 0
    else:
        bats['dnp'][b_id] = 0
        arrive(b_id, t)


def arrive(b_id, t):
    global bats
    bats['loc'][b_id] = bats['next_loc'][b_id]
    bats['dnp'][b_id] = 0
    bats['states'][b_id] = bats['next_state'][b_id]
    if bats['loc'][b_id] == bats['roost_locs'][b_id] and bats['states'][b_id]==0:
        bats['ts'][b_id] = 0
        roost(b_id, t)
    elif bats['loc'][b_id] == bats['roost_locs'][b_id] and bats['states'][b_id]==3:
        feed_young(b_id,t)
    else:
        forage(b_id, t)


def get_all_bat_locations():
    global bats
    return bats['loc']

def travel_to_roost(b_id, t):
    global bats
    bats['ts'][b_id] = 0
    bats['dnp'][b_id] = bats['time_to_roost'][b_id]
    bats['states'][b_id] = 2
    bats['next_state'][b_id] = 0
    bats['next_loc'][b_id] = bats['roost_locs'][b_id]
    bats['loc'][b_id] = bats['sp']['num_p']
    bats['time_to_roost'][b_id] = 0
    travel(b_id, bats['next_loc'][b_id], t)

def travel_to_roost_feed_young(b_id, t):
    global bats
    bats['dnp'][b_id] = bats['time_to_roost'][b_id]
    bats['states'][b_id] = 2
    bats['next_state'][b_id] = 3
    bats['next_loc'][b_id] = bats['roost_locs'][b_id]
    bats['loc'][b_id] = bats['sp']['num_p']
    bats['time_to_roost'][b_id] = 0
    travel(b_id, bats['next_loc'][b_id], t)

def feed_young(b_id, t):
    global bats
    bats['dnp'][b_id] = bats['time_to_roost'][b_id]
    bats['states'][b_id] = 2
    bats['next_state'][b_id] = 1
    bats['loc'][b_id] = bats['roost_locs'][b_id]
    bats['food_before_roost'][b_id]=0
    bats['th'][b_id,t] = bats['roost_locs'][b_id]

    #go back to foraging -- if needed
    patch_probs = get_p_patches(b_id)
    patch_choice = random.choices(np.arange(bats['sp']['num_p']), patch_probs).pop()

    bats['dnp'][b_id] = patches.get_time_to_next_patch(int(bats['roost_locs'][b_id]), patch_choice)
    bats['time_to_roost'][b_id] = bats['dnp'][b_id]
    bats['next_loc'][b_id] = patch_choice
    bats['loc'][b_id] = bats['sp']['num_p']

def start_foraging(b_id, t):
    global bats
    bats['ts'][b_id] = 0
    bats['states'][b_id] = 2
    bats['next_state'][b_id] = 1
    patch_probs = get_p_patches(b_id)
    patch_choice = random.choices(np.arange(bats['sp']['num_p']), patch_probs).pop()
    bats['dnp'][b_id] = patches.get_time_to_next_patch(int(bats['loc'][b_id]), patch_choice)
    bats['time_to_roost'][b_id] = patches.get_time_to_next_patch(int(patch_choice), bats['roost_locs'][b_id])
    bats['next_loc'][b_id] = patch_choice
    bats['loc'][b_id] = bats['sp']['num_p']
    travel(b_id,bats['next_loc'][b_id], t)


def update_bats(t):
    global bats
    # update foraging bats
    for b_id in bats['id']: #find where energy and th not being updated
        bats['energy'][b_id][t]=bats['energy'][b_id][t-1]
        bats['th'][b_id][t] = bats['th'][b_id][t - 1]

    if t%24==0:
        [travel_to_roost(b, t) for b in bats['id'] if (bats['next_loc'][b]!= bats['roost_locs'][b])]
    else:
        foraging = bats['id'][np.where(bats['states'] == 1)]
        roosting = bats['id'][np.where(bats['states'] == 0)]
        traveling = bats['id'][np.where(bats['states'] == 2)]

        if len(foraging)>0:
            not_time_to_roost = bats['id'][np.where(bats['ts'] < 24 - bats['rest_time'] - bats['time_to_roost'])]
            max_food = bats['id'][np.where(bats['food_before_roost']*bats['gender']>=bats['max_food_before_roost'])]
            keep_foraging = np.asarray([b_id for b_id in foraging if (b_id in not_time_to_roost) and (b_id not in max_food)])
            [forage(b, t) for b in keep_foraging]
            done_foraging = np.asarray([b_id for b_id in foraging if b_id not in not_time_to_roost])
            [travel(b,bats['roost_locs'][b] , t) for b in done_foraging]
            [travel_to_roost_feed_young(b_id, t) for b_id in max_food if b_id in foraging]

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
    plt.title(str(t/24) + " Days")
    plt.imshow(bat_mat.reshape((10,10)))
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.show()
    return bat_mat



