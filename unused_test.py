def all_forage(ids,t):
    global bats
    metabolic_rate = bats['mr']
    foraging_rate = bats['fr']
    energy_conversion = bats['ev']
    rec_conv = bats['bat_resource_conversion']

    locs = bats['loc'][ids]
    energies = bats['energy'][ids][t-1]

    bats['tp'][ids] += 1
    bats['ts'][ids] += 1
    bats['th'][ids, t] = locs

    avail_rec = patches.patches['resources'][locs]
    avail_rec[np.where(avail_rec>foraging_rate)] = foraging_rate
    e_temps = rec_conv * energy_conversion * avail_rec - metabolic_rate
    bats['energy'][ids][t] = energies+e_temps
    patches.patches['resources'][locs] = patches.patches['resources'][locs]-avail_rec

    next_e_temps = rec_conv * energy_conversion * foraging_rate - metabolic_rate

    if avail_rec>foraging_rate:
        e_temp =rec_conv*energy_conversion*foraging_rate-metabolic_rate#* energy#rec_conv * foraging_rate*(1/time_in_cur_patch) - metabolic_rate * energy
        bats['daily_diet_hist'][b_id] += patches.get_patch_resource_types(loc) * foraging_rate
        patches.update_used_resources(loc, foraging_rate)

        bats['energy'][b_id][t] = energy + e_temp
        next_e_temp = rec_conv * energy_conversion*foraging_rate - metabolic_rate#* energy#rec_conv * foraging_rate*(1/(time_in_cur_patch+1)) - metabolic_rate * energy
    else:
        e_temp = rec_conv*energy_conversion*avail_rec-metabolic_rate #* energy
        bats['daily_diet_hist'][b_id] += patches.get_patch_resource_types(loc) * avail_rec
        patches.update_used_resources(loc, avail_rec)

        bats['energy'][b_id][t] = energy + e_temp
        next_e_temp = 0


#old way to check resource type use
#if np.any(prop_necessary_res_consumed>=1) and sum(bats['daily_diet_hist'][b_id])<1: #take diet variability into account
            #    enough_res = np.where(prop_necessary_res_consumed>=1)
                # increase probability of needed resource being consumed
            #    for i in range(len(p_options)):
            #        res = patches.get_patch_resource_types(p_options[i])
            #        if res[enough_res]>0 and res[np.where(prop_necessary_res_consumed<1)]==0:
            #            patch_probs[i] = 0
            #    if sum(patch_probs) <= 0:
            #        patch_probs = np.ones(len(p_options)) / len(p_options)
            #    else:
            #        patch_probs = patch_probs / sum(patch_probs)