import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
from matplotlib import gridspec
import time
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import h5py

from unfold import weighted_binary_crossentropy

import os
os.environ['CUDA_VISIBLE_DEVICES']="2"

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)

tf.config.experimental.set_memory_growth(physical_devices[0], True)

from matplotlib import style
style.use('/global/home/users/ftoralesacosta/dotfiles/scientific.mplstyle')

label = {}
label['sys0'] = 'HFS scale (in jet)'
label['sys1'] = 'HFS scale (remainder)'
label['sys4'] = 'HFS polar angle'
label['sys5'] = 'HFS $\phi$ angle' 
label['sys7'] = 'Lepton energy scale'
label['sys10'] = 'Lepton polar angle'
label['sys11'] = 'Lepton $\phi$ angle'
label['QED'] = 'QED rad corr.'


#define mc according to the label keys
mc = label stuff idk

theta0_G = mc[['gene_px','gene_py','gene_pz','genjet_pt','genjet_eta','genjet_phi','genjet_dphi','genjet_qtnorm']].to_numpy()
weights_MC_sim = mc['wgt']
pass_reco = np.array(mc['pass_reco'])
pass_truth = np.array(mc['pass_truth'])
pass_fiducial = np.array(mc['pass_fiducial'])
del mc
_ = gc.collect()


#WEIGHTS
test_model = tf.keras.models.load_model("/clusterfs/ml4hep_nvme2/ftoralesacosta/disjets/FinalReading/inputfiles/fullscan/models/Rapgap_sys_0/", custom_objects={"weighted_binary_crossentropy": weighted_binary_crossentropy })

NNweights_step2 = np.ones(len(theta0_G))
for i in range(5):
    mymodel = tf.keras.models.load_model("../../disjets/inputfiles/fullscan3/models/Rapgap_sys_11_iteration"+str(i)+"_step2", compile=False)
    NNweights_step2_hold = mymodel.predict(scaler_data.transform(theta0_G),batch_size=10000)
    NNweights_step2_hold = NNweights_step2_hold/(1.-NNweights_step2_hold)
    NNweights_step2_hold = NNweights_step2_hold[:,0]
    NNweights_step2_hold = np.squeeze(np.nan_to_num(NNweights_step2_hold,posinf=1))
    NNweights_step2_hold[pass_truth==0] = 1.
    NNweights_step2 = NNweights_step2_hold*NNweights_step2_hold


weights = weights_MC_sim*NNweights_step2_hold

# KINEMATICS
e_px = theta0_G[:,0]
e_py = theta0_G[:,1]

jet_pT_mag = theta0_G[:,3]
jet_phi = theta0_G[:,5]
print(min(jet_pT_mag))

jet_qT = theta0_G[:,7] #[not to be confused with q_Perp!]

print(min(jet_phi),max(jet_phi))
print(min(jet_pT_mag))

jet_px = np.multiply(jet_pT_mag, np.cos(jet_phi))
jet_py = np.multiply(jet_pT_mag, np.sin(jet_phi))

jet_pT = np.array([jet_px,jet_py])
e_pT = np.array([e_px,e_py])

q_perp_vec = jet_pT + e_pT
P_perp_vec = (e_pT-jet_pT)/2

q_perp_mag = np.linalg.norm(q_perp_vec,axis=0)
P_perp_mag = np.linalg.norm(P_perp_vec,axis=0)



q_dot_P = q_perp_vec[0,:]*P_perp_vec[0,:] + q_perp_vec[1,:]*P_perp_vec[1,:]

cosphi = (q_dot_P)/(q_perp_mag*P_perp_mag)
asymm_phi = np.arccos(cosphi)



# CUTS
pT_cut = jet_pT_mag > 10.
q_over_pT_cut = q_perp_mag/jet_pT_mag < 0.3 #Kyle guessed ~0.3, needs variation
qT_cut = np.where((jet_qT<0.25), True, False)
phi_nan_cut = ~np.isnan(asymm_phi)
#q_perp_cut = q_perp_mag < 10.0 #q_perp_max

print(np.shape(pT_cut))
#print(np.shape(q_perp_cut))


cut_arrays = [pass_fiducial,
              pT_cut,
              q_over_pT_cut,
              qT_cut,
              phi_nan_cut]
              #q_perp_cut]

cuts = np.ones(len(pT_cut))

for cut in cut_arrays:
    print(len(cut))
    cuts = np.logical_and(cuts,cut)



