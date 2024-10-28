import numpy as np
import os
out_id =int(os.environ['SLURM_ARRAY_TASK_ID'])
test = np.arange(out_id)
np.savetxt("/proj/nobackup/hpc2n2023-026/bat_distribution/Bat_model/results/test"+str(out_id)+".txt",test)