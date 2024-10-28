import test_bat
import patches
import numpy as np
import random
import time

def avg_bats_per_patch(simulation_parameters):
    test_time_spent_foraging = []
    test_time_spent_total = np.zeros((simulation_parameters['sim_len'], simulation_parameters['num_p']))
    for t in range(simulation_parameters['sim_len']):
        temp = test_bat.get_bat_density_map(t, [30, 30], False)
        if len(temp[np.where(temp==0)])<(simulation_parameters['num_p']-5):
            test_time_spent_foraging.append(list(temp))
        test_time_spent_total[t] = test_bat.get_bat_density_map(t, [30, 30], False)
    test_time_spent_foraging = np.array(test_time_spent_foraging)
    foraging = np.mean(test_time_spent_foraging, axis=0)
    total = np.mean(test_time_spent_total, axis=0)

    return [foraging,total]

def get_avg_roost_dur_per_bat(simulation_parameters):
    roost_switch_count = np.zeros(simulation_parameters['pop'])
    roost_duration = []

    for b in range(simulation_parameters['pop']):
        count = 1
        roost_duration_b = []
        for i in range(1,int(simulation_parameters['sim_len']/24-1)):
            if test_bat.bats['roost_each_day'][b,i]!=test_bat.bats['roost_each_day'][b,i-1]:
                roost_duration_b.append(count)
                roost_switch_count[b]+=1
                count = 1
            else:
                count +=1
        roost_duration_b.append(count)
        roost_duration.append(roost_duration_b)

    avg_r_dur = np.zeros(simulation_parameters['pop'])
    for r in range(len(roost_duration)):
        avg_r_dur[r] = np.mean(roost_duration[r])
    return avg_r_dur

def movement_probs(simulation_parameters):
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
    return movement_dicts

def run_simulation(s,patch_net,simulation_parameters):
    random.seed(s)
    #time_t1 = time.time()
    # initializing patches and bats
    patches.initialize_patches(patch_net, simulation_parameters)
    test_bat.initialize_bats(simulation_parameters)

    # update bats and grid for each time step
    for i in range(1, simulation_parameters['sim_len']):
        test_bat.update_bats(i)
        patches.update_patches(i)


    #time_t2 = time.time()
    #print("Total time: ", np.round((time_t2-time_t1)/60), " minutes")

    # shows the movement of listed bats to different patch types over the simulation
    # test_bat.view_individual_bat_behavior([0],0,simulation_parameters['sim_len'])
    # plt.show()

    # determine probabilities of going to each patch from other patch types

    m_probs = movement_probs(simulation_parameters)

    s_p_n = test_bat.get_average_foraging_changes_per_night()

    ard = get_avg_roost_dur_per_bat(simulation_parameters)

    [foraging_avg_visitors,total_avg_visitors] = avg_bats_per_patch(simulation_parameters)

    res_dict = {"PMP": m_probs,
                "PSA": s_p_n,
                "RDA": ard,
                "FAV": foraging_avg_visitors,
                "TAV": total_avg_visitors}

    return res_dict