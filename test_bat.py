import random
import patches
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import matplotlib.cm as cm
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


def initialize_bats(simulation_parameters):
    global bats, sim_time, scat
    n = simulation_parameters['pop']
    sim_time = 1

    bats = {'sp': simulation_parameters,
            'id': np.arange(n),
            'states': np.zeros(n),  # 0=roosting, 1=foraging, 2=traveling, 3=feeding children, 4=dead
            'tp': np.zeros(n),  # time_in_cur_patch
            'tr':np.zeros(n),  #time in current roost
            'tt': np.zeros(n),  # time_traveling
            'energy': np.zeros((n, simulation_parameters['sim_len'])),
            'dist_traveled':np.zeros((n, simulation_parameters['sim_len'])),
            'energy_next': np.zeros(n),
            'energy_cp': np.zeros(n),  #energy gained from current patch
            'energy_cr': np.zeros(n),  # energy gained from current roost
            'e_discount':0.05,
            'fr': 29.16666667*np.ones((n, simulation_parameters['sim_len'])),  #foraging rate in g/hr
            'fc': 17,  #food to energy conversion in kJ/g
            'mr': (146.16/2)*np.ones((n, simulation_parameters['sim_len'])), #146.16 #metabolic rate -  https://besjournals.onlinelibrary.wiley.com/doi/epdf/10.1046/j.1365-2435.2003.00706.x (low end of flying)
            'tc': 146.16*np.ones(n),  #NOT SURE travel cost (flying metabolic rate)  https://besjournals.onlinelibrary.wiley.com/doi/epdf/10.1046/j.1365-2435.2003.00706.x
            'r_mr': 7.308*np.ones((n, simulation_parameters['sim_len'])),  #7.308,#resting metabolic rate 1/20th of normal mr -  https://besjournals.onlinelibrary.wiley.com/doi/epdf/10.1046/j.1365-2435.2003.00706.x
            'bat_resource_conversion': 0.75,
            'loc': np.zeros(n),
            'next_loc': np.zeros(n),
            'prev_loc': np.zeros(n),
            'next_state': np.zeros(n),
            'time_to_roost': np.zeros(n),
            'dnp': np.zeros(n),
            'th': np.zeros((n, simulation_parameters['sim_len'])),
            'fh': [],
            'daily_diet_hist':np.zeros((n, 2)),  #hard fruit, soft fruit
            'diet_rec': np.array([2/3,1/3]),
            'roost_locs': np.zeros(n),
            'max_dist_in_hr': (50.0/30.0)*np.ones((n, simulation_parameters['sim_len'])),  #50 km at 30 km per hour
            'max_food_before_roost': 29.16666667*11.5*5,  #3 foraging trips
            'food_before_roost': np.zeros(n),
            'smell_dist':10.0/30.0,  #max smell dist in hr
            'gender': np.zeros(n),  #0 -> Male, 1 -> female
            'young_state':np.zeros((n, simulation_parameters['sim_len'])),  #0 -> no young, #1 has young
            'preg_state': np.zeros((n, simulation_parameters['sim_len'])),  # 0 -> not pregnant, #1 pregnant
            'avg_speed': 30.0,
            'visits_per_patch':np.zeros((n,simulation_parameters['num_p'])),
            'roost_each_day':np.zeros((n, int(simulation_parameters['sim_len']/24))),
            'd_th': -1000.0 #bats death threshold
            }

    assign_bats_to_roost()
    assign_gender()
    for b_id in bats['id']:
        temp=np.asarray(get_initial_forage_hist(b_id))
        bats['fh'].append(temp)
        for p in range(bats['sp']['num_p']):
            bats['visits_per_patch'][b_id,p] += len(np.where(temp==p)[0])
    bats['fh']=np.asarray(bats['fh'])
    bats['energy'][:,0] = 2000*np.ones(n)

def assign_gender():
    global bats
    pm = bats['sp']['gp'][0]
    pf = bats['sp']['gp'][1]
    bats['gender'] = np.array(random.choices([0.0,1.0],weights=[pm,pf],k = bats['sp']['pop']))

    #for female bats - set times pregnant, times with young, and times without
    fb = bats['id'][np.where(bats['gender']==1)]
    #'approx_time_pregnant': 1,  # which day of simulation
    t_preg = np.round(np.abs(np.random.normal(bats['sp']['approx_time_pregnant'] * 24, 5 * 24, len(fb))))
    t_preg = t_preg.astype(int)
    #'pregnancy_duration': 5 * 30 * 24,  # 5 months of pregnancy
    preg_dur = np.round(np.abs(np.random.normal(bats['sp']['pregnancy_duration'], 7*24, len(fb))))
    preg_dur = preg_dur.astype(int)

    #'time_to_rear_young': 5 * 30 * 24  # 5 months before self feeding
    young_dur = np.round(np.abs(np.random.normal(bats['sp']['time_to_rear_young'], 7*24, len(fb))))
    young_dur = young_dur.astype(int)
    count = 0
    for b in fb:
        t1 = t_preg[count]
        if t1<bats['sp']['sim_len']:
            t2 = t_preg[count]+preg_dur[count]
            if t2<bats['sp']['sim_len']:
                bats['preg_state'][b][t1:t2 + 1] = 1
                t3 = t_preg[count] + preg_dur[count] + young_dur[count]
                if t3<bats['sp']['sim_len']:
                    bats['young_state'][b][t2:t3 + 1] = 1
                else:
                    bats['young_state'][b][t2:] = 1
            else:
                bats['preg_state'][b][t1:] = 1
        count = count + 1

    #fr, r_mr, mr,max_dist_in_hr
    #increasing these for females - with young
    bats['fr'] = bats['fr']+bats['preg_state']*bats['fr']+bats['young_state']*bats['fr'] #doubling foraging rate when pregnant or feeding
    bats['r_mr'] = bats['r_mr']+bats['young_state']*bats['r_mr']/2+bats['preg_state']*bats['r_mr']/2 #increasing resting metabolic rate for mothers
    #bats['mr'] = bats['mr'] + bats['young_state']*bats['mr']/2+ bats['preg_state']*bats['mr']/2 #increasing normal metabolic rate for mothers

    #decreasing this for females - with young
    bats['max_dist_in_hr'] =bats['max_dist_in_hr'] - bats['young_state']*((40/30))

def assign_bats_to_roost():
    global bats
    roost_options = patches.get_patch_type_ids('Roost')
    bats['roost_options']=roost_options
    if len(roost_options)==1:
        bats['roost_locs'] = roost_options*np.ones(bats['sp']['pop'])
    else:
        bats['roost_locs'] = np.array(random.choices(roost_options, k=bats['sp']['pop']))

    bats['roost_locs']= bats['roost_locs'].astype(float)
    bats['loc'] = np.copy(bats['roost_locs'])
    bats['th'][:, 0] = np.copy(bats['roost_locs'])
    bats['roost_each_day'][:,0]= np.copy(bats['roost_locs'])

def get_initial_forage_hist(b_id):
    global bats
    roost = bats['roost_locs'][b_id]
    p_types = bats['sp']['patch_types_options']
    p_probs = bats['sp']['patch_type_forage_probs']

    #get patches in foraging range
    patches_in_range = patches.get_patches_in_range(roost, bats['max_dist_in_hr'][b_id,0])

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


def update_bats(t):
    global bats
    bats['prev_loc']=bats['loc']
    #check_dead(t)
    if t%12==0 and t%24!=0: #start foraging
        time_to_forage(t)
    elif t%24==0: #start roosting
        time_to_roost(t)
    else: #continue what  you were doing
        continue_on(t)

def check_dead(t):
    global bats
    dead = bats['id'][np.where(bats['energy'][:,t-1] <= bats['d_th'])]
    if len(dead)>0:
        bats['states'][dead] = 4
        bats['energy'][dead,t:] = np.nan
        bats['loc'][dead] = np.nan
        bats['next_loc'][dead] = np.nan
        bats['prev_loc'][dead] = np.nan
        bats['next_state'][dead] = np.nan
        bats['time_to_roost'][dead]= np.nan
        bats['dnp'][dead]= np.nan
        bats['th'][dead,t:]= np.nan
        bats['roost_locs'][dead]=np.nan

def time_to_roost(t):
    global bats
    #Determine if individual bats should change roost (does it need to be a group?)
    # Need energy gained in current roost -> bats['energy_cr']
    # Need expected energy gain (hour or foraging) -> record this when calculated in previous foraging step? bats['energy_next']
    # Need time in current roost -> bats['tr']
    # Need travel cost to new roost and time - check all roost options
    not_dead= bats['id'][np.where(bats['states']!=4)]
    bats_to_switch = []
    bats_to_switch_unique=[]
    estars=np.zeros((len(not_dead),len(bats['roost_options'])))
    count=0
    for i in bats['roost_options']:
        travel_locs = bats['prev_loc'][not_dead].copy()
        nl = bats['next_loc'][not_dead]
        travel_locs[np.where(travel_locs == bats['sp']['num_p'])]=nl[np.where(travel_locs == bats['sp']['num_p'])]
        tts= np.array([patches.get_time_to_next_patch(p,i) for p in travel_locs])#calculate travel_times
        estars[:,count] =(bats['energy_cr'][not_dead] - bats['tc'][not_dead]*tts)/(bats['tr'][not_dead] + tts)
        bats_to_switch.append(np.where(estars[:,count]>bats['energy_next'][not_dead])[0])
        rl = bats['roost_locs'][not_dead]
        bats_to_switch[count] = bats_to_switch[count][np.where(rl[bats_to_switch[count]] != i)]
        for b in bats_to_switch[count]:
            if b not in bats_to_switch_unique:
                bats_to_switch_unique.append(b)
        count=count+1

    for b in bats_to_switch_unique:
        b_op=[]
        for p in range(len(bats['roost_options'])):
            if b in bats_to_switch[p]:
                b_op.append(p)
        bats['roost_locs'][b] = bats['roost_options'][random.choice(b_op)]
        bats['tr'][b]=0
        bats['energy_cr'][b]=0


    bats['next_loc'][not_dead] = bats['roost_locs'][not_dead].copy()
    bats['next_state'][not_dead] = np.zeros(len(not_dead))
    not_roosting = bats['id'][np.where(bats['loc'] != bats['roost_locs'])]
    already_roosting = bats['id'][np.where(bats['loc'] == bats['roost_locs'])]

    if len(not_roosting)>0:
        temp=bats['time_to_roost'][not_roosting]
        bats['dnp'][not_roosting] = temp #if heading to roost, this decreases
        bats['states'][not_roosting] = 2*np.ones(len(not_roosting))
        bats['next_state'][not_roosting] = np.zeros(len(not_roosting))
        bats['next_loc'][not_roosting] = bats['roost_locs'][not_roosting]
        travel(not_roosting, t)
    if len(already_roosting) > 0:
        bats['th'][already_roosting, t] = bats['roost_locs'][already_roosting]
        bats['energy'][already_roosting, t] = bats['energy'][already_roosting, t - 1] - bats['r_mr'][already_roosting,t]  # *bats['mr']* energy
        bats['food_before_roost'][already_roosting] = 0
    bats['roost_each_day'][:, int(t/24)] = np.copy(bats['roost_locs'])

def continue_on(t):
    foraging = bats['id'][np.where(bats['states'] == 1)]
    roosting = bats['id'][np.where(bats['states'] == 0)]
    traveling = bats['id'][np.where(bats['states'] == 2)]
    feeding_young = bats['id'][np.where(bats['states'] == 3)]
    if len(foraging)>0:
        forage(foraging,t)
    if len(roosting) > 0:
        roost(roosting, t)
    if len(traveling) >0:
        travel(traveling,t)
    if len(feeding_young) >0:
        feed_young(feeding_young,t)

def travel(B_arr,t):
    global bats
    bats['energy_cp'][B_arr] = np.zeros(len(B_arr))
    # energy updates
    tc = bats['tc'][B_arr] + 1*bats['tc'][B_arr]*bats['gender'][B_arr]
    bats['energy'][B_arr,t] =bats['energy'][B_arr,t-1] - tc * np.stack((bats['dnp'][B_arr], np.ones(len(B_arr)))).min(axis=0)
    bats['th'][B_arr, t] = bats['sp']['num_p']

    #other updates
    bats['dnp'][B_arr] = bats['dnp'][B_arr] - np.ones(len(B_arr))

    #done traveling
    arrived=B_arr[np.where(bats['dnp'][B_arr] <= 0)]
    bats['states'][arrived] = bats['next_state'][arrived]
    bats['loc'][arrived] = bats['next_loc'][arrived]
    bats['dnp'][arrived] = 0


def roost(b_arr,t):
    global bats
    bats['energy'][b_arr, t] =bats['energy'][b_arr, t-1]-bats['r_mr'][b_arr,t]
    bats['th'][b_arr, t] = bats['roost_locs'][b_arr]
    bats['tr'][b_arr] +=1

def forage(b_arr,t):
    global bats
    locs = bats['loc'][b_arr]
    temp = np.roll(bats['fh'][b_arr,:],-1)
    temp[:,-1]=locs
    bats['fh'][b_arr]=temp
    bats['th'][b_arr, t] = locs
    rec_conv = bats['bat_resource_conversion']
    avail_rec = patches.get_patch_resources(locs)
    other_bats_in_loc = np.array([len(np.where(locs==loc)[0]) for loc in locs])-1

    all_fr = 2*bats['fr'][b_arr,t]/(1+0.25**(-bats['energy'][b_arr, t - 1]/5000))
    foraged_rec = np.array([min([all_fr[i], avail_rec[i]]) for i in range(len(b_arr))]) #2*29.16666667 is max fr
    e_temp2 = rec_conv * (foraged_rec* bats['fc']) * np.exp(-bats['e_discount'] * bats['tp'][b_arr] - bats['e_discount'] * other_bats_in_loc) - bats['mr'][b_arr, t]
    bats['energy_cp'][b_arr] += e_temp2
    bats['energy_cr'][b_arr] += e_temp2
    patches.update_used_resources(bats['loc'][b_arr], foraged_rec)
    bats['energy'][b_arr, t] = bats['energy'][b_arr, t - 1] + e_temp2
    bats['food_before_roost'][b_arr] += bats['fr'][b_arr, t]

    #bp_ind = np.where(avail_rec >=bats['fr'][b_arr,t])
   # b_plenty = b_arr[bp_ind]

    #bl_ind = np.where(avail_rec < np.max(bats['fr'][b_arr,t]))
    #b_lacking = b_arr[bl_ind]

    #if len(b_plenty) > 0:
    #    e_temp_bp = e_temp2[bp_ind]#rec_conv*(bats['fr'][b_plenty,t]*bats['fc'])* np.exp(-bats['e_discount'] * bats['tp'][b_plenty] - bats['e_discount']*other_bats_in_loc[bp_ind]) - bats['mr'][b_plenty,t]
    #    bats['energy_cp'][b_plenty] += e_temp_bp
    #    bats['energy_cr'][b_plenty] += e_temp_bp
    #    patches.update_used_resources(bats['loc'][b_plenty], bats['fr'][b_plenty,t])
    #    bats['energy'][b_plenty,t] = bats['energy'][b_plenty, t - 1] + e_temp_bp
    #    bats['food_before_roost'][b_plenty] += bats['fr'][b_plenty,t]

    #if len(b_lacking)>0:
    #    e_temp = e_temp2[bl_ind]#rec_conv * (avail_rec[np.where(avail_rec <np.max(bats['fr'][b_arr,t]))] * bats['fc']) * np.exp(-bats['e_discount'] * bats['tp'][b_lacking] - bats['e_discount']*other_bats_in_loc[bl_ind]) - bats['mr'][b_lacking,t]
    #    bats['energy_cp'][b_lacking] += e_temp
    #    bats['energy_cr'][b_lacking] += e_temp
    #    patches.update_used_resources(bats['loc'][b_lacking], bats['fr'][b_lacking,t])
    #    bats['energy'][b_lacking, t] = bats['energy'][b_lacking, t - 1] + e_temp
    #    bats['food_before_roost'][b_lacking] += avail_rec[np.where(avail_rec < np.max(bats['fr'][b_arr,t]))]

    #next expected resources
    #next_e_temp = np.zeros(len(b_arr))
    #avail_rec_next = patches.get_patch_resources(locs)
    #bp_ind_next = np.where(avail_rec_next >= bats['fr'][b_arr,t])
    #bl_ind_next = np.where(avail_rec_next < bats['fr'][b_arr,t])
    #b_plenty = b_arr[bp_ind_next]
    #b_lacking = b_arr[bl_ind_next]
    #next_e_temp[np.where(avail_rec_next >= bats['fr'][b_arr,t])] = rec_conv * (bats['fr'][b_plenty,t] * bats['fc']) * np.exp(
    #    -bats['e_discount'] * bats['tp'][b_plenty] - bats['e_discount']*other_bats_in_loc[bp_ind_next]) - bats['mr'][b_plenty,t]
    #next_e_temp[np.where(avail_rec_next < bats['fr'][b_arr,t])] = rec_conv * (avail_rec_next[np.where(avail_rec_next < bats['fr'][b_arr,t])] * bats['fc']) * np.exp(
    #    -bats['e_discount'] * bats['tp'][b_lacking] - bats['e_discount']*other_bats_in_loc[bl_ind_next]) - bats['mr'][b_lacking,t]

    #all_fr = bats['fr'][b_arr, t]  # 2*bats['fr'][b_arr,t]/(1+0.25**(-bats['energy'][b_arr, t - 1]/10000))
    #foraged_rec = np.array([min([all_fr[i], avail_rec[i]]) for i in range(len(b_arr))])
    #e_temp2 = rec_conv * (foraged_rec* bats['fc']) * np.exp(-bats['e_discount'] * bats['tp'][b_arr] - bats['e_discount'] * other_bats_in_loc) - bats['mr'][b_arr, t]
    #          rec_conv * (bats['fr'][b_plenty, t] * bats['fc']) * np.exp(-bats['e_discount'] * bats['tp'][b_plenty] - bats['e_discount'] * other_bats_in_loc[bp_ind]) - bats['mr'][b_plenty, t]

    #bats['energy_cp'][b_arr] += e_temp
    #bats['energy_cr'][b_arr] += e_temp
    #patches.update_used_resources(bats['loc'][b_arr], bats['fr'][b_arr, t])
    #bats['energy'][b_arr, t] = bats['energy'][b_arr, t - 1] + e_temp
    #bats['food_before_roost'][b_arr] += bats['fr'][b_arr, t]

    #next expected resources
    avail_rec_next = patches.get_patch_resources(locs)
    foraged_rec_next = np.array([min([all_fr[i], avail_rec_next[i]]) for i in range(len(b_arr))])
    next_e_temp = rec_conv * (foraged_rec_next * bats['fc']) * np.exp(
        -bats['e_discount'] * bats['tp'][b_arr] - bats['e_discount'] * other_bats_in_loc) - bats['mr'][b_arr, t]

    bats['tp'][b_arr] += 1
    bats['energy_next'][b_arr] = next_e_temp
    make_decisions(b_arr, next_e_temp, t)





def make_decisions(b_arr,next_e_temp,t):
    global bats
    can_forage = b_arr[np.where(bats['time_to_roost'][b_arr]<=((24-t)%24)-1)]
    cf_ind = np.where(bats['time_to_roost'][b_arr] <= ((24 - t) % 24) - 1)
    #out of keep foraging, which need to feed young?
    not_max_food = b_arr[np.where(bats['food_before_roost'][b_arr]*bats['young_state'][b_arr,t] < bats['max_food_before_roost'])]
    nmf_ind = np.where(bats['food_before_roost'][b_arr] *bats['young_state'][b_arr,t] < bats['max_food_before_roost'])
    max_food = b_arr[np.where(bats['food_before_roost'][b_arr] *bats['young_state'][b_arr,t] >= bats['max_food_before_roost'])]
    keep_foraging = np.intersect1d(not_max_food, can_forage)
    kf_ind = np.intersect1d(cf_ind, nmf_ind)
    feed_young = np.intersect1d(max_food, can_forage)
    go_home = b_arr[np.where(bats['time_to_roost'][b_arr]>=((24-t)%24)-1)]

    #for bats that will continue foraging, make OFT decision
    if len(keep_foraging)>0:
        estars = get_estars(keep_foraging, t)
        max_estars = np.max(estars,1)
        if_switch = max_estars-next_e_temp[kf_ind]
        to_switch=keep_foraging[np.where(if_switch>0)]
        if len(to_switch)>0:
            estar_switch =estars[np.where(if_switch>0),:]-next_e_temp[np.where(if_switch>0)].reshape((len(to_switch),1))
            estar_switch = estar_switch.reshape((len(to_switch),bats['sp']['num_p']))
            p_options=[]
            for k in range(len(to_switch)):
                #options for bat
                temp=np.where(estar_switch[k,:]>0)[0]
                if len(temp)>0 and np.max(patches.patches['resources'][temp])>0:
                    #resources in options
                    p_op_rec=patches.patches['resources'][temp]

                    #num visits to options
                    p_op_vis=bats['visits_per_patch'][to_switch[k],temp]

                    #options in smell dist
                    p_op_smell=patches.get_patches_in_smell_range(temp, bats['loc'][to_switch[k]])

                    #get_probs
                    w_temp = p_op_rec/sum(p_op_rec)
                    w_temp[np.where(p_op_vis > 0)] = w_temp[np.where(p_op_vis > 0)] * p_op_vis[np.where(p_op_vis > 0)]
                    w_temp[np.where(p_op_smell > 0)] = w_temp[np.where(p_op_smell > 0)] * 10
                    w_temp = w_temp/sum(w_temp)
                    p_choice = random.choices(temp, weights=w_temp, k=1).pop()
                    # to visualize patch choice
                    #test = np.zeros(20*20)
                    #test[temp] = w_temp
                    #plt.imshow(test.reshape(20,20))
                    #plt.colorbar()
                    bats['visits_per_patch'][to_switch[k], p_choice] += 1
                    change_patch(to_switch[k], p_choice,t)
    if len(feed_young)>0:
        travel_to_roost_feed_young(feed_young, t)
    if len(go_home)>0:
        travel_to_roost(go_home,t)

def change_patch(b_id, new_loc,t):
    global bats
    loc = bats['loc'][b_id]
    bats['dnp'][b_id] = patches.get_time_to_next_patch(loc, new_loc)
    bats['dist_traveled'][b_id, t] += bats['dnp'][b_id] * bats['avg_speed']
    bats['states'][b_id] = 2
    bats['next_loc'][b_id] = new_loc
    bats['loc'][b_id] = bats['sp']['num_p']
    bats['tp'][b_id] = 0
    bats['time_to_roost'][b_id] = patches.get_time_to_next_patch(new_loc, bats['roost_locs'][b_id])


def travel_to_roost(b_id, t):
    global bats
    bats['dnp'][b_id.astype(int)] = bats['time_to_roost'][b_id.astype(int)]
    bats['dist_traveled'][b_id, t] += bats['dnp'][b_id] * bats['avg_speed']
    bats['states'][b_id.astype(int)] = 2
    bats['next_state'][b_id.astype(int)] = 0
    bats['next_loc'][b_id.astype(int)] = bats['roost_locs'][b_id.astype(int)]
    bats['loc'][b_id.astype(int)] = bats['sp']['num_p']

def travel_to_roost_feed_young(b_id, t):
    global bats
    bats['dnp'][b_id.astype(int)] = bats['time_to_roost'][b_id.astype(int)]
    bats['dist_traveled'][b_id, t] += bats['dnp'][b_id] * bats['avg_speed']
    bats['states'][b_id.astype(int)] = 2
    bats['next_state'][b_id.astype(int)] = 3
    bats['next_loc'][b_id.astype(int)] = bats['roost_locs'][b_id.astype(int)]
    bats['loc'][b_id.astype(int)] = bats['sp']['num_p']
    bats['time_to_roost'][b_id.astype(int)] = 0

def feed_young(b_id, t):
    global bats
    b_ids=b_id.astype(int)
    bats['loc'][b_ids] = bats['roost_locs'][b_ids]
    bats['food_before_roost'][b_ids] = 0
    bats['th'][b_ids, t] = bats['roost_locs'][b_ids]
    if 1 >= ((24 - t) % 24) - 1:
        bats['states'][b_ids] = 0
    else:
        bats['states'][b_ids] = 2
        bats['next_state'][b_ids] = 1
        # go back to foraging -- if needed
        for k in range(len(b_ids)):
            # resources in options
            p_op_rec = patches.patches['resources']

            # num visits to options
            p_op_vis = bats['visits_per_patch'][b_ids[k], :]

            # get probs
            w_temp = p_op_rec / sum(p_op_rec)
            w_temp[np.where(p_op_vis > 0)] = w_temp[np.where(p_op_vis > 0)] * p_op_vis[np.where(p_op_vis > 0)]
            w_temp = w_temp / sum(w_temp)
            p_choice = random.choices(patches.patches['id'], weights=w_temp, k=1).pop()
            change_patch(b_ids[k], p_choice,t)


def get_patch_level_forage_hist(b_id):
    global bats
    patch_hist=[]
    for b in b_id:
        patch_hist.append([len(bats['fh'][b,np.where(bats['fh'][b,:] == i)][0]) for i in range( bats['sp']['num_p'])])
    return np.array(patch_hist)


def get_estars(b_arr,t):
    global bats
    locs = bats['loc'][b_arr]
    energy = bats['energy_cp'][b_arr].reshape((len(b_arr),1))* np.ones((1, bats['sp']['num_p']))
    time_in_cur_patch = bats['tp'][b_arr].reshape((len(b_arr),1))* np.ones((1, bats['sp']['num_p']))
    for b in range(len(b_arr)):
        time_in_cur_patch[b, int(locs[b])] = 1.0
    t_all_p=np.array([patches.get_time_to_other_points(loc) for loc in locs]).reshape((len(b_arr),bats['sp']['num_p']))
    tc = bats['tc'][b_arr].reshape((len(b_arr), 1)) * np.ones((1, bats['sp']['num_p']))*t_all_p
    estars = (energy - tc)/(t_all_p + time_in_cur_patch)
    for b in range(len(b_arr)):
        estars[b, int(locs[b])] = -np.inf
    return estars


def get_p_patches(rec_prob,b_id):  # get patch to start foraging in. Later patches chosen based on MVT
    global bats
    #factor in previous visits to patches with resources
    fh = bats['fh'][b_id,:]
    patch_prob=rec_prob.copy()
    visit_patch =np.array([len(fh[np.where(fh==i)]) for i in range(bats['sp']['num_p'])])
    temp=patch_prob[np.where(visit_patch>0)]*visit_patch[np.where(visit_patch>0)]
    patch_prob[np.where(visit_patch>0)] =temp
    #factor in aversion to other bats
    #b_per_p = [get_bats_in_patch(int(p)) for p in range(bats['sp']['num_p'])]
    #patch_prob = patch_prob*(1-(1/bats['sp']['pop'])*np.ones(len(b_per_p))*b_per_p)
    patch_prob=patch_prob/np.sum(patch_prob)
    return patch_prob

def get_bats_in_patch(p):
    global bats
    return len(bats['id'][bats['loc']==p])

def time_to_forage(t):
    #bats will start foraging for the day
    not_dead = bats['id'][np.where(bats['states']!=4)]
    patch_rec = patches.patches['resources']
    rec_prob = patch_rec / sum(patch_rec)
    choices = np.zeros(bats['sp']['pop'])
    dists = np.zeros(bats['sp']['pop'])
    d_r = np.zeros(bats['sp']['pop'])
    for b_id in not_dead:
        patch_probs = get_p_patches(rec_prob,b_id)
        choice = random.choices(np.arange(bats['sp']['num_p']),weights=patch_probs, k=1).pop()
        bats['visits_per_patch'][b_id, choice] +=1
        if np.isnan(bats['roost_locs'][b_id]):
            print('here')
        dists[b_id] = patches.get_time_to_next_patch(int(bats['roost_locs'][b_id]), choice)
        choices[b_id]=int(choice)
        d_r[b_id] = patches.get_time_to_next_patch(int(choice), bats['roost_locs'][b_id])
        bats['time_to_roost'][b_id] = patches.get_time_to_next_patch(choice, bats['roost_locs'][b_id])

    bats['time_to_roost']=d_r
    bats['dnp'] = dists
    bats['next_loc']=choices
    bats['loc']= bats['sp']['num_p']*np.ones(bats['sp']['pop'])
    bats['next_state'] = np.ones(bats['sp']['pop'])
    bats['states'] = 2*np.ones(bats['sp']['pop'])
    travel(np.arange(bats['sp']['pop']), t)


def view_individual_bat_behavior(b_ids,st,et):
    global bats
    p_types = bats['sp']['patch_types_options']

    for b in b_ids:
        temp = np.asarray(bats['th'][b,st:et])
        type_list = np.zeros(len(temp))
        for t in range(len(temp)):
            if temp[t] == bats['sp']['num_p']:
                type_list[t] = len(p_types)
            else:
                for p in range(len(p_types)):
                    if patches.get_patch_type_by_id(int(temp[t])) == p_types[p]:
                        type_list[t] = p
        plt.plot(np.arange(st,et)/24, type_list)
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

def get_bat_density_map(t,d):
    global bats
    bat_mat = np.zeros(bats['sp']['num_p'])
    for p in range(bats['sp']['num_p']):
        bat_mat[p]=len(bats['th'][:, t][np.where(bats['th'][:, t]==p)])
    plt.title(str(t/24) + " Days")
    plt.imshow(bat_mat.reshape((d[0],d[1])))
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.show()
    return bat_mat

def bat_on_grid_over_time(b_arr, d):
    global bats, scat, title, colors#, rec, n_row, n_c
    n_row = d[0]
    n_c = d[1]

    fig, ax = plt.subplots()
    ax.set(xlim=[-5, n_c+5], ylim=[-5, n_row+5])
    ax.grid(which='major', alpha=0.5)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    sort_index = np.argsort(bats['th'][b_arr, 0])
    sorted_locs = np.sort(bats['th'][b_arr, 0])
    rand_locs = patches.get_loc_in_patch(sorted_locs)
    #sizes = np.array([len(sorted_locs[np.where(sorted_locs  == l)]) for l in unique_locs])
    #all_sizes = np.array([sizes[np.where(unique_locs==i)] for i in sorted_locs]).flatten()*50
    colors = cm.rainbow(np.linspace(0, 1, len(b_arr)))
    scat = ax.scatter(rand_locs[0,:], rand_locs[1,:], c = colors[sort_index], s=1)
    #rec = ax.imshow(patches.patches['resource_history'][:,0].reshape(n_row,n_c))
    title = ax.text(0.5,1, "",transform=ax.transAxes, ha="center")

    ani = animation.FuncAnimation(fig=fig, func=lambda frame: animation_update(frame, b_arr), frames=bats['sp']['sim_len'], interval=1)
    writervideo = animation.FFMpegWriter(fps=60)
    ani.save('test_bat.mp4', writer=writervideo)
    plt.close()
    #plt.show()

def animation_update(frame, b_arr):
    global scat,title, colors#, rec, n_row, n_c
    sort_index = np.argsort(bats['th'][b_arr, frame])
    sorted_locs = np.sort(bats['th'][b_arr, frame])
    rand_locs = patches.get_loc_in_patch(sorted_locs)
    #sizes = np.array([len(sorted_locs[np.where(sorted_locs == l)]) for l in unique_locs])
    #all_sizes = np.array([sizes[np.where(unique_locs == i)] for i in sorted_locs]).flatten() * 50
    data = np.stack([rand_locs[0,:], rand_locs[1,:]]).T
    scat.set_offsets(data)
    #scat.set_sizes(all_sizes)
    scat.set_facecolor(colors[sort_index])
    #rec.set_array(patches.patches['resource_history'][:, frame].reshape(n_row, n_c))
    title.set_text(np.round(frame/24,2))
    return scat

def see_individual_roosting_behavior(b_ids, roost_locs):
    global bats
    for b in b_ids:
        temp = np.asarray(bats['roost_each_day'][b, :])
        type_list = np.zeros(len(temp))
        for t in range(len(temp)):
            for p in range(len(roost_locs)):
                if temp[t] == roost_locs[p]:
                    type_list[t] = p
        plt.scatter(np.arange(bats['sp']['sim_len']/24),type_list)
    plt.xlabel("Days")
    plt.yticks(np.arange(len(roost_locs)))
    plt.show()