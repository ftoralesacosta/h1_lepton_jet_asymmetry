import os
import numpy as np

# import tensorflow as tf
import tensorflow.keras.backend as K

n_bootstraps = 101
# n_bootstraps = 3
for i in range(1, n_bootstraps):
    myrand = np.random.randint(0,10000)  # sets random seed in strap.py
    # print("python strap.py "+str(i)+" "+str(myrand)+" &> mylog"+str(i)+".txt &")
    # os.system("python strap.py "+str(i)+" "+str(myrand)+" &> mylog"+str(i)+".txt &")
    # os.system("python strap.py "+str(i)+" "+str(myrand))

    print("python unfold_fullstats_boot.py Rapgap bootstrap_%i %i"%(i,myrand))
    # os.system("python unfold_fullstats_boot.py Rapgap bootstrap_%i %i"%(i,myrand))
    os.system("python unfold_fullstats_boot.py Rapgap bootstrap_%i %i"%(i,myrand)+" &> logs/mylog%i.txt"%(i))
    print(i)

    K.clear_session()
print("DONE")
