import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from unfold import weighted_binary_crossentropy, multifold

import os
os.environ['CUDA_VISIBLE_DEVICES']="2"

import tensorflow as tf
import tensorflow.keras.backend as K

import sys
import json

# When run from 'run_many.py':
# os.system("python strap.py "+str(i)+" "+str(myrand)+" &> mylog"+str(i)+".txt &")

seed = int(sys.argv[2])
mc_i = int(sys.argv[1])
print("Running Strap.py i=%i, seed=%i"%(mc_i,seed))

tf.random.set_seed(seed)#setting a seed with random num from args
np.random.seed(seed)

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

data = pd.read_pickle("/clusterfs/ml4hep/yxu2/unfolding_mc_inputs/Data_nominal.pkl")
mc = pd.read_pickle("/clusterfs/ml4hep/yxu2/unfolding_mc_inputs/Rapgap_nominal.pkl")

#Preprocessing
theta_unknown_S = data[['e_px','e_py','e_pz','jet_pt','jet_eta','jet_phi','jet_dphi','jet_qtnorm']].to_numpy()
theta0_S = mc[['e_px','e_py','e_pz','jet_pt','jet_eta','jet_phi','jet_dphi','jet_qtnorm']].to_numpy()
theta0_G = mc[['gene_px','gene_py','gene_pz','genjet_pt','genjet_eta','genjet_phi','genjet_dphi','genjet_qtnorm']].to_numpy()
weights_MC_sim = mc['wgt']
pass_reco = np.array(mc['pass_reco'])
pass_truth = np.array(mc['pass_truth'])

#Standardize
scaler_data = StandardScaler()
scaler_data.fit(theta_unknown_S)

scaler_mc_truth = StandardScaler()
scaler_mc_truth.fit(theta0_G[pass_truth==1])
print("Done Standardization")

print(np.sum(np.isnan(theta_unknown_S)))
theta_unknown_S_scaled = scaler_data.transform(theta_unknown_S)
theta0_S_scaled = scaler_data.transform(theta0_S)
theta0_G_scaled = scaler_mc_truth.transform(theta0_G)

weights_MC_sim_scaled = weights_MC_sim/np.average(weights_MC_sim)

#Acceptance effects
theta0_S_scaled[:,0][pass_reco==0] = -10
theta0_G_scaled[:,0][pass_truth==0] = -10

weights_MC_sim_scaled *= len(theta_unknown_S_scaled)/len(theta0_S_scaled)

# iterations = 5
# iterations = 10
# iterations = 1
num_observables = 8

weights = {}
models = {}
history = {}

print("before bootstrap loop")
K.clear_session()
weights, models, history = multifold(num_observables=num_observables,
                                                             iterations=iterations,
                                                             theta0_G=theta0_G_scaled,
                                                             theta0_S=theta0_S_scaled,
                                                             theta_unknown_S= theta_unknown_S_scaled,
                                                             verbose=0,weights_MC_sim=weights_MC_sim_scaled,
                                                             weights_MC_data=np.random.poisson(1,len(theta_unknown_S_scaled)))
        
for iiter in range(iterations):
    iboot=mc_i
    for step in range(2):
        # np.save("./stat"+str(mc_i)+"_Rapgap_nominal_b"+str(iboot)+"_i"+str(iiter)+"_s"+str(step),weights[mc_i,iboot][iiter,step,:])
        # np.save("./stat"+str(mc_i)+"_model_Rapgap_nominal_b"+str(iboot)+"_i"+str(iiter)+"_s"+str(step),models[mc_i,iboot][iiter,step+1])
        # myjson = json.dumps(history['step'+str(step+1)][iiter].history)

        np.save("./bootstraps/weights_BootStrap_%i_Iteration_%i_Rapgap_Step_%i.npy"%(iboot,iiter,step),weights[iiter,step,:])
        np.save("./bootstraps/models_BootStrap_%i_Iteration_%i_Rapgap_Step_%i.npy"%(iboot,iiter,step),models[iiter,step+1])
        f = open("./stat"+str(mc_i)+"_losshistory_Rapgap_nominal_b"+str(iboot)+"_i"+str(iiter)+"_s"+str(step)+".json","w")
        f.write(myjson)
        f.close()
