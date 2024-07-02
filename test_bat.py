import random
import patches
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import matplotlib.cm as cm
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

# code for updating bats
#TODO: young maturation in to population

def initialize_bats(simulation_parameters):
    #global parameters for dictionary of individuals and points for scatter plot animations
    global bats, points

    n = simulation_parameters['pop']
    sim_len = simulation_parameters['sim_len']
    #setting up dictionary to use to describe the bats
    bats = {'sp': simulation_parameters,  #storing the overall simulationn parameters
            'id': np.arange(n), #array of bat ids that are used for indexing
            'states': np.zeros(n),  # activity states for bats in the population 0=roosting, 1=foraging, 2=traveling, 3=feeding children, 4=dead
            'disease_states': np.zeros(n),  # disease states for bats in the population, 0=susceptible, 1=infected, 2=recovered
            'cumulative_infected': np.zeros(sim_len), # keeping track of how many bats have been infected
            'tp': np.zeros(n),  # the time bats have spent in their current patches (resets after changing location)
            'tr':np.zeros(n),  #the time bats have spent in their current roosts (resets after roost switching)
            'tt': np.zeros(n),  #the time bats have spent traveling when switching location (used to determine if theyve arrived)
            'energy': np.zeros((n, sim_len)), #stores each bats energy at each time in the simulation
            'dist_traveled':np.zeros((n, sim_len)), #stores the distance traveled by each bat at each interval in the simulation
            'energy_next': np.zeros(n), # used to make foraging decisions, energy gained in the next step if continuing foraging in current patch
            'energy_cp': np.zeros(n),  # energy gained from current patch
            'energy_cr': np.zeros(n),  # energy gained from current roost
            'e_discount':0.05, # discount rate for energy gained by staying in the same patch (used to simulate need for diversity in diet)
            'fr': 29.16666667*np.ones((n,sim_len)),  #foraging rate in g/hr
            'fc': 17,  #food to energy conversion in kJ/g
            'mr': (146.16/2)*np.ones((n,sim_len)),  # Guess for metabolic rate
            'tc': 146.16*np.ones(n),  #Guess for travel cost (flying metabolic rate)
            'r_mr': 7.308*np.ones((n, sim_len)),  # Guess for resting metabolic rate 1/20th of normal mr
            'bat_resource_conversion': 0.75, # Guess for proportion of potential energy gained from consumed resource
            'loc': np.zeros(n), #current location of each bat in the grid (location for traveling is > num_p)
            'next_loc': np.zeros(n), # location that bats are traveling to (used when traveling)
            'prev_loc': np.zeros(n), # keeps track of where the bats were before updating
            'next_state': np.zeros(n), # keeps track of which activity bats will be doing upon arrival after traveling
            'time_to_roost': np.zeros(n), # keeps track of how far a bat is from its roost
            'dnp': np.zeros(n), #the distance to the next patch (used when traveling)
            'th': np.zeros((n, sim_len)),# total history of locations of each bat across the simulation
            'fh': [], #foraging history, assigned later in initialization
            'roost_locs': np.zeros(n), # current roost of each bat in the population
            'max_dist_in_hr': 30.0*np.ones((n, sim_len)), #GUESS: average speed of 30 km per hour
            'max_food_before_roost': 29.16666667*11.5*5, #GUESS: maximum amount off food bats with pups collect before returning to roost
            'food_before_roost': np.zeros(n), # keeps track of how much food bats have collected/consumed before roosting
            'smell_dist':10.0/30.0,  #GUESS:  maximum distance bats can smell measured in travel time to loaction
            'sex': np.zeros(n),  # 0 -> Male, 1 -> female
            'young_state':np.zeros((n, sim_len)),  #0 -> no young, 1 -> has young
            'preg_state': np.zeros((n,sim_len)),  # 0 -> not pregnant, 1 -> pregnant
            'pregnancy_duration': 5 * 30 * 24,  # Guess: approximately 5 months of pregnancy
            'time_to_rear_young': 5 * 30 * 24,  #Guess: approximately 5 months before self feeding
            'avg_speed': 30.0, #Guess: average speed of bat flight when traveling
            'visits_per_patch': np.zeros((n,simulation_parameters['num_p'])), #for each bat, records number of times each patch was cisited
            'roost_each_day': np.zeros((n, int(sim_len/24))), # records chosen roosts for each bat each day
            'd_th': -1000.0,  #Guess: starvation threshold that leads to death
            'max_energy': 5000 #Guess: maximum energy level of bats
            }

    # functions for assigning initial values
    assign_bats_to_roost() # initial roost is chosen randomly from given options
    assign_sex() #bats are assigned male or female based on distribution in simulation parameters
    initialize_disease() # bats are randomly chosen to be infected based on the given number of initial infected bats
    initialize_foraging_history() # an initial foraging history is assigned to each bat based on the given proportion of forgaging time spent in each patch type
    bats['energy'][:,0] = 2000*np.ones(n) # initial energy storage of bats

def initialize_foraging_history():
    #an initial foraging history is assigned to each bat and the visits per patch and foraging history parameters are updated
    global bats
    for b_id in bats['id']:
        temp=np.asarray(get_initial_forage_hist(b_id))
        bats['fh'].append(temp)
        for p in range(bats['sp']['num_p']):
            bats['visits_per_patch'][b_id,p] += len(np.where(temp==p)[0])
    bats['fh']=np.asarray(bats['fh'])

def initialize_disease():
    # bats are randomly chosen from the given number of initial infected to be infected.
    # disease states and the cumulative number of infected bats is updated
    global bats
    init_inf = random.choices(bats['id'], k=bats['sp']['init_inf'])
    bats['disease_states'][init_inf] = 1
    bats['cumulative_infected'][0] = bats['sp']['init_inf']

def assign_sex():
    # function to assign sex to bat population as well as pre-define pregnancy and young-rearing states
    global bats

    #assign sex based on given distribution
    pm = bats['sp']['gp'][0]
    pf = bats['sp']['gp'][1]
    bats['sex'] = np.array(random.choices([0.0,1.0],weights=[pm,pf],k = bats['sp']['pop']))

    #for female bats - set times pregnant, times with young, and times without
    fb = bats['id'][np.where(bats['sex']==1)]
    #'approx_time_pregnant': 1,  # which day of simulation within 5 days of set day
    t_preg = np.round(np.abs(np.random.normal(bats['sp']['approx_time_pregnant'] * 24, 5 * 24, len(fb))))
    t_preg = t_preg.astype(int)
    #'pregnancy_duration': 5 * 30 * 24,  duration of pregnancy is within a week of average time
    preg_dur = np.round(np.abs(np.random.normal(bats['pregnancy_duration'], 7*24, len(fb))))
    preg_dur = preg_dur.astype(int)

    #'time_to_rear_young': 5 * 30 * 24, duration of feeding young is within a week of average time
    young_dur = np.round(np.abs(np.random.normal(bats['time_to_rear_young'], 7*24, len(fb))))
    young_dur = young_dur.astype(int)

    # using durations above to pre-define states before running simulation
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
    #increasing these for females with young
    bats['fr'] = bats['fr']+bats['preg_state']*bats['fr']+bats['young_state']*bats['fr'] #doubling foraging rate when pregnant or feeding
    bats['r_mr'] = bats['r_mr']+bats['young_state']*bats['r_mr']/2+bats['preg_state']*bats['r_mr']/2 #increasing resting metabolic rate for mothers

    #decreasing this for females - with young
    bats['max_dist_in_hr'] =bats['max_dist_in_hr'] - bats['young_state']*10

def assign_bats_to_roost():
    #assigning initial roost to bats in population based on pre-defined roost options
    global bats
    roost_options = patches.get_patch_type_ids('Roost')
    bats['roost_options']=np.array(roost_options)
    if len(roost_options)==1:
        bats['roost_locs'] = roost_options*np.ones(bats['sp']['pop'])
    else:
        bats['roost_locs'] = np.array(random.choices(roost_options, k=bats['sp']['pop']))

    bats['roost_locs']= bats['roost_locs'].astype(float)
    bats['loc'] = np.copy(bats['roost_locs']) #bats start the simulation in the roosts
    bats['th'][:, 0] = np.copy(bats['roost_locs']) # updating the total history
    bats['roost_each_day'][:,0]= np.copy(bats['roost_locs']) #updating the roosting history

def get_initial_forage_hist(b_id):
    # assigning the initial foraging history
    global bats
    roost = bats['roost_locs'][b_id]
    p_types = bats['sp']['patch_types_options']
    p_probs = bats['sp']['patch_type_forage_probs']

    #get patches in foraging range from roost
    patches_in_range = patches.get_patches_in_range(roost, bats['max_dist_in_hr'][b_id,0])

    #get patch types of patches in foraging range
    num_of_type = np.zeros(len(p_types))
    for p in patches_in_range:
        type = patches.get_patch_type_by_id(p)
        if type == 'Roost':#new
            type = 'Forest'
        for t in range(len(p_types)):
            if type == p_types[t]:
                num_of_type[t] += 1

    # calculate probabilities of choosing patches in each range based on patch type and number of that type
    patch_probs = np.zeros(len(patches_in_range))
    for p in range(len(patches_in_range)):
        type = patches.get_patch_type_by_id(p)
        if type == 'Roost':#new
            type = 'Forest'
        for t in range(len(p_types)):
            if type == p_types[t]:
                patch_probs[p] = p_probs[t]/num_of_type[t]
    # return random choices based on probabiliites
    return random.choices(patches_in_range, weights=patch_probs,k=48)


def update_bats(t):
    # update function for each time step
    global bats
    # update previous locations to be current locations
    bats['prev_loc']=bats['loc']
    #check if any of the bats have starved, and if so, remove them
    check_dead(t)

    #update steps for activity
    if t%12==0 and t%24!=0: # if it's sunset, start foraging
        time_to_forage(t)
    elif t%24==0: #if it's sunrise, go back to the roost
        time_to_roost(t)
    else: # otherwise continue what  you were doing
        continue_on(t)
    # update step for the disease
    update_disease(t)

def update_disease(t):
    #update function for the disease
    global bats
    # get indices of sick bats
    sick_ind = np.where(bats['disease_states']==1)
    #get locations of sick bats
    sick_locs = bats['loc'][sick_ind]
    # determine which bats are in the same locations as the sick bats
    bats_in_sick_locs = []
    for i in np.unique(sick_locs):
        bats_in_sick_locs+=list(bats['id'][np.where(bats['loc']==i)])
    bats_in_sick_locs = np.array(bats_in_sick_locs)
    # get indices of susceptible bats
    sus_bats = bats['id'][np.where(bats['disease_states'] == 0)]
    # determine which of the bats in the same location as sick are susceptible (can be infected)
    can_infect = np.intersect1d(bats_in_sick_locs,sus_bats)
    if len(can_infect)>0: #if there are bats that can be infected
        # infect bats based on beta as probability of infection per contact (defined as being in same patch for an hour)
        infect_decisions = random.choices([1,0], weights = [bats['sp']['beta'], 1-bats['sp']['beta']],k = len(can_infect))
        #updates based on decisions
        bats['disease_states'][can_infect] = infect_decisions
        bats['cumulative_infected'][t] = bats['cumulative_infected'][t-1]+sum(infect_decisions)

    if len(sick_ind) > 0: #sick bats may recover based on gamma
        recover_decisions = random.choices([2, 1], weights=[bats['sp']['gamma'], 1 - bats['sp']['gamma']], k=len(sick_ind))
        bats['disease_states'][sick_ind] = recover_decisions


def check_dead(t):
    #check if any bats died from starvation
    global bats
    dead = bats['id'][np.where(bats['energy'][:,t-1] <= bats['d_th'])]
    #for bats that died, remove from simulation
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
    # for the bats that are still alive, make a decision about which roost to go to and head there
    not_dead= bats['id'][np.where(bats['states']!=4)]

    bats_to_switch = [] #bats that may switch to each roost
    bats_to_switch_unique=[]

    travel_locs = bats['prev_loc'][not_dead].copy()  # location before heading to roost
    nl = bats['next_loc'][not_dead]  # locations traveling bats are headed to

    # for bats currently traveling, assume location is where they were headed
    travel_locs[np.where(travel_locs == bats['sp']['num_p'])] = nl[np.where(travel_locs == bats['sp']['num_p'])]

    # calculate E^* for each bat and each roost option
    estars=np.zeros((len(not_dead),len(bats['roost_options'])))
    count=0
    for i in bats['roost_options']:# for each of the roost options
        # calculate travel times between roost i and current locations of bats
        tts= np.array([patches.get_time_to_next_patch(p,i) for p in travel_locs])
        # calculate E^* for decision making
        estars[:,count] =(bats['energy_cr'][not_dead] - bats['tc'][not_dead]*tts)/(bats['tr'][not_dead] + tts)
        # ignore the bats that already roost here
        estars[np.where(bats['roost_locs'][not_dead] == i), count] = -np.inf
        count=count+1


    #for each bat
    for b in not_dead:
        temp = np.where(estars[b,:] > bats['energy_next'][b])[0]
        #if  E* > E'[t+1] anywhere
        if len(temp)>0:
            #choose a new roost from available options
            bats['roost_locs'][b] = bats['roost_options'][random.choice(temp)]
            bats['tr'][b] = 0
            bats['energy_cr'][b] = 0

    #update location to move to
    bats['next_loc'][not_dead] = bats['roost_locs'][not_dead].copy()
    #update next state to roosting
    bats['next_state'][not_dead] = np.zeros(len(not_dead))

    #check if bats have already traveled to a roost
    not_roosting = bats['id'][np.where(bats['loc'] != bats['roost_locs'])]
    already_roosting = bats['id'][np.where(bats['loc'] == bats['roost_locs'])]

    if len(not_roosting)>0: #if not already roosting, start traveling to roost
        temp=bats['time_to_roost'][not_roosting]
        bats['dnp'][not_roosting] = temp #if heading to roost, this decreases
        bats['states'][not_roosting] = 2*np.ones(len(not_roosting))
        bats['next_state'][not_roosting] = np.zeros(len(not_roosting))
        bats['next_loc'][not_roosting] = bats['roost_locs'][not_roosting]
        travel(not_roosting, t)
    if len(already_roosting) > 0: #otherwise do roosting things
        bats['th'][already_roosting, t] = bats['roost_locs'][already_roosting]
        bats['energy'][already_roosting, t] = bats['energy'][already_roosting, t - 1] - bats['r_mr'][already_roosting,t]  # *bats['mr']* energy
        bats['food_before_roost'][already_roosting] = 0
    #add roost to roost history
    bats['roost_each_day'][:, int(t/24)] = np.copy(bats['roost_locs'])

def continue_on(t):
    # activity state hasn't changed based on time of day, so continue activities
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
    #traveling behavior of the bats
    global bats
    # since bats inn B_arr are changing patches, reset their energy gained in the current patch hto 0
    bats['energy_cp'][B_arr] = np.zeros(len(B_arr))

    # increase travel cost for pregnant bats
    tc = bats['tc'][B_arr] + bats['tc'][B_arr]*bats['preg_state'][B_arr,t]

    #update energy level of the bats based on travel cost. Use either the travel time left (if less than 1 hr) or 1 hr
    bats['energy'][B_arr,t] =bats['energy'][B_arr,t-1] - tc * np.stack((bats['dnp'][B_arr], np.ones(len(B_arr)))).min(axis=0)

    #update the total location history
    bats['th'][B_arr, t] = bats['sp']['num_p']

    #update the distance left to travel
    bats['dnp'][B_arr] = bats['dnp'][B_arr] - np.ones(len(B_arr))

    #check if done traveling and make updates if so
    arrived=B_arr[np.where(bats['dnp'][B_arr] <= 0)]
    bats['states'][arrived] = bats['next_state'][arrived]
    bats['loc'][arrived] = bats['next_loc'][arrived]
    bats['dnp'][arrived] = 0


def roost(b_arr,t):
    #roosting behavior of the bats
    global bats
    # update their energy according to the resting metabolic rate
    bats['energy'][b_arr, t] =bats['energy'][b_arr, t-1]-bats['r_mr'][b_arr,t]
    # update the total location history
    bats['th'][b_arr, t] = bats['roost_locs'][b_arr]
    # update the time spent at the current roost
    bats['tr'][b_arr] +=1

def forage(b_arr,t):
    # foraging behavior of the bats
    global bats
    # look at current foraging location of the bats
    locs = bats['loc'][b_arr]
    # update the foraging history of the bats by removing the oldest and adding on the newest
    temp = np.roll(bats['fh'][b_arr,:],-1)
    temp[:,-1]=locs
    bats['fh'][b_arr]=temp

    #update the total history
    bats['th'][b_arr, t] = locs

    rec_conv = bats['bat_resource_conversion']
    avail_rec = patches.get_patch_resources(locs)
    other_bats_in_loc = np.array([len(np.where(locs==loc)[0]) for loc in locs])-1

    # foraging rate is based on the optimal stored energy of the bat (increased to a max of 2*'fr' if no stored energy, at 'fr' when 'max_energy' stored)
    all_fr = 2*bats['fr'][b_arr,t]/(1+0.25**(-bats['energy'][b_arr, t - 1]/bats['max_energy']))
    #bats collect resources in foraging location based calculated fr above and available resources
    foraged_rec = np.array([min([all_fr[i], avail_rec[i]]) for i in range(len(b_arr))])

    #update the energy and energy related values of the bats based on collected resources
    e_temp = rec_conv * (foraged_rec* bats['fc']) * np.exp(-bats['e_discount'] * bats['tp'][b_arr] - bats['e_discount'] * other_bats_in_loc) - bats['mr'][b_arr, t]
    bats['energy_cp'][b_arr] += e_temp
    bats['energy_cr'][b_arr] += e_temp
    bats['energy'][b_arr, t] = bats['energy'][b_arr, t - 1] + e_temp
    bats['food_before_roost'][b_arr] += bats['fr'][b_arr, t]

    # update  the patch by removing resources consumed
    patches.update_used_resources(bats['loc'][b_arr], foraged_rec)

    #calculate next expected resources if staying in patch in the next step
    avail_rec_next = patches.get_patch_resources(locs)
    foraged_rec_next = np.array([min([all_fr[i], avail_rec_next[i]]) for i in range(len(b_arr))])
    next_e_temp = rec_conv * (foraged_rec_next * bats['fc']) * np.exp(
        -bats['e_discount'] * bats['tp'][b_arr] - bats['e_discount'] * other_bats_in_loc) - bats['mr'][b_arr, t]

    bats['tp'][b_arr] += 1
    bats['energy_next'][b_arr] = next_e_temp
    #decide to stay  or move in next time step
    make_decisions(b_arr, next_e_temp, t)

def make_decisions(b_arr,next_e_temp,t):
    #decide to stay or switch patches in next foraging step
    global bats
    # check if the bats will forage in the next step or if it will be time to head to the roost based on their distance from the roost
    can_forage = b_arr[np.where(bats['time_to_roost'][b_arr]<=((24-t)%24)-1)]
    go_home = b_arr[np.where(bats['time_to_roost'][b_arr] >= ((24 - t) % 24) - 1)]

    cf_ind = np.where(bats['time_to_roost'][b_arr] <= ((24 - t) % 24) - 1)

    #check if bats need to feed young
    not_max_food = b_arr[np.where(bats['food_before_roost'][b_arr]*bats['young_state'][b_arr,t] < bats['max_food_before_roost'])]
    nmf_ind = np.where(bats['food_before_roost'][b_arr] *bats['young_state'][b_arr,t] < bats['max_food_before_roost'])
    max_food = b_arr[np.where(bats['food_before_roost'][b_arr] *bats['young_state'][b_arr,t] >= bats['max_food_before_roost'])]
    keep_foraging = np.intersect1d(not_max_food, can_forage)
    kf_ind = np.intersect1d(cf_ind, nmf_ind)
    feed_young = np.intersect1d(max_food, can_forage)


    #for bats that will continue foraging, make Optimal Foraging Theory decision
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
    #update steps when the decision is made to go to a new foraging patch
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
    # update steps when it's time to head to the roost
    global bats
    bats['dnp'][b_id.astype(int)] = bats['time_to_roost'][b_id.astype(int)]
    bats['dist_traveled'][b_id, t] += bats['dnp'][b_id] * bats['avg_speed']
    bats['states'][b_id.astype(int)] = 2
    bats['next_state'][b_id.astype(int)] = 0
    bats['next_loc'][b_id.astype(int)] = bats['roost_locs'][b_id.astype(int)]
    bats['loc'][b_id.astype(int)] = bats['sp']['num_p']

def travel_to_roost_feed_young(b_id, t):
    # update steps when it's time to head to the roost to feed young
    global bats
    bats['dnp'][b_id.astype(int)] = bats['time_to_roost'][b_id.astype(int)]
    bats['dist_traveled'][b_id, t] += bats['dnp'][b_id] * bats['avg_speed']
    bats['states'][b_id.astype(int)] = 2
    bats['next_state'][b_id.astype(int)] = 3
    bats['next_loc'][b_id.astype(int)] = bats['roost_locs'][b_id.astype(int)]
    bats['loc'][b_id.astype(int)] = bats['sp']['num_p']
    bats['time_to_roost'][b_id.astype(int)] = 0

def feed_young(b_id, t):
    # update steps feeding young and deciding what to do next (roost or forage)
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
    #returns the foraging history of bat b_id
    global bats
    patch_hist=[]
    for b in b_id:
        patch_hist.append([len(bats['fh'][b,np.where(bats['fh'][b,:] == i)][0]) for i in range( bats['sp']['num_p'])])
    return np.array(patch_hist)


def get_estars(b_arr,t):
    #calculates the E^* values for making patch switching decisions based on OFT
    global bats
    locs = bats['loc'][b_arr]
    energy = bats['energy_cp'][b_arr].reshape((len(b_arr),1))* np.ones((1, bats['sp']['num_p']))
    time_in_cur_patch = bats['tp'][b_arr].reshape((len(b_arr),1))* np.ones((1, bats['sp']['num_p']))
    t_all_p=patches.patches['tbp'][locs.astype(int),:]
    tc = bats['tc'][b_arr].reshape((len(b_arr), 1)) * np.ones((1, bats['sp']['num_p']))*t_all_p
    estars = (energy - tc)/(t_all_p + time_in_cur_patch)
    estars[:, locs.astype(int)] = -np.inf
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
    b_per_p = get_bats_in_patch()
    patch_prob = patch_prob*(1-(1/bats['sp']['pop'])*b_per_p)
    patch_prob=patch_prob/np.sum(patch_prob)
    return patch_prob

def get_bats_in_patch():
    #return number of bats per patch
    global bats
    bats_per_patch = np.zeros(bats['sp']['num_p'])
    patches_w_bats = np.unique(bats['loc']).astype(int)
    for p in patches_w_bats:
        bats_per_patch[p] = len(np.where(bats['loc']==p)[0])

    return bats_per_patch

def time_to_forage(t):
    #Bats will start foraging for the day. Bats that are still living will choose a
    # patch to forage in based on available resources
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
        dists[b_id] = patches.get_time_to_next_patch(int(bats['roost_locs'][b_id]), choice)
        choices[b_id]=int(choice)
        d_r[b_id] = dists[b_id].copy()
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
    global bats, points, title, colors#, rec, n_row, n_c
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
    points = ax.scatter(rand_locs[0,:], rand_locs[1,:], c = colors[sort_index], s=1)
    #rec = ax.imshow(patches.patches['resource_history'][:,0].reshape(n_row,n_c))
    title = ax.text(0.5,1, "",transform=ax.transAxes, ha="center")

    ani = animation.FuncAnimation(fig=fig, func=lambda frame: animation_update(frame, b_arr), frames=bats['sp']['sim_len'], interval=1)
    writervideo = animation.FFMpegWriter(fps=60)
    ani.save('test_bat.mp4', writer=writervideo)
    plt.close()
    #plt.show()

def animation_update(frame, b_arr):
    global points,title, colors#, rec, n_row, n_c
    sort_index = np.argsort(bats['th'][b_arr, frame])
    sorted_locs = np.sort(bats['th'][b_arr, frame])
    rand_locs = patches.get_loc_in_patch(sorted_locs)
    #sizes = np.array([len(sorted_locs[np.where(sorted_locs == l)]) for l in unique_locs])
    #all_sizes = np.array([sizes[np.where(unique_locs == i)] for i in sorted_locs]).flatten() * 50
    data = np.stack([rand_locs[0,:], rand_locs[1,:]]).T
    points.set_offsets(data)
    #points.set_sizes(all_sizes)
    points.set_facecolor(colors[sort_index])
    #rec.set_array(patches.patches['resource_history'][:, frame].reshape(n_row, n_c))
    title.set_text(np.round(frame/24,2))
    return points

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