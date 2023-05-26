import os
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K

import sys

data_path = "/clusterfs/ml4hep/yxu2/unfolding_mc_inputs"
systematics = ["sys_0","sys_1","sys_5","sys_7","sys_11"]
# systematics = ["nominal"]
for sys in systematics:

    #argv[3] = 0 means no bootstrapping
    print(f"python unfold_fullstats_boot.py Rapgap {sys} 0 ")
    os.system(f"python unfold_fullstats_boot.py Rapgap {sys} 0")

    # print(f"python unfold_fullstats_boot.py Django {sys} 0 ")
    # os.system(f"python unfold_fullstats_boot.py Django {sys} 0")

    K.clear_session()
print("DONE")
