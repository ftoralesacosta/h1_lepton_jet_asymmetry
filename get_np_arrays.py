import sys

import numpy as np
import pandas as pd
import gc
import time
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import h5py
from unfold import weighted_binary_crossentropy


def npy_from_pkl(label):

    #DATA LOADING
    print("Loading PKL /clusterfs/ml4hep/yxu2/unfolding_mc_inputs/"+label+".pkl")
    mc = pd.read_pickle("/clusterfs/ml4hep/yxu2/unfolding_mc_inputs/"+label+".pkl") #FIXME: Do these fit on git?

    theta0_G = mc[['gene_px','gene_py','gene_pz','genjet_pt','genjet_eta','genjet_phi','genjet_dphi','genjet_qtnorm']].to_numpy()
    weights_MC_sim = mc['wgt']
    pass_reco = np.array(mc['pass_reco'])
    pass_truth = np.array(mc['pass_truth'])
    pass_fiducial = np.array(mc['pass_fiducial'])
    del mc
    _ = gc.collect()

    data = pd.read_pickle("/clusterfs/ml4hep/yxu2/unfolding_mc_inputs/Data_nominal.pkl")
    theta_unknown_S = data[['e_px','e_py','e_pz','jet_pt','jet_eta','jet_phi','jet_dphi','jet_qtnorm']].to_numpy()
    scaler_data = StandardScaler()
    scaler_data.fit(theta_unknown_S)
    del data
    _ = gc.collect()

    #WEIGHTS
    NNweights_step2 = np.ones(len(theta0_G))
    model_path = "/clusterfs/ml4hep_nvme2/ftoralesacosta/disjets/inputfiles/fullscan3/models/"
    print("Loading Model "+model_path+label)

    for i in range(5):

        print("Iteration"+str(i)+" Step 2")

        mymodel = tf.keras.models.load_model(model_path+label+"_iteration"+str(i)+"_step2",compile=False)
        NNweights_step2_hold = mymodel.predict(scaler_data.transform(theta0_G),batch_size=10000)
        NNweights_step2_hold = NNweights_step2_hold/(1.-NNweights_step2_hold)
        NNweights_step2_hold = NNweights_step2_hold[:,0]
        NNweights_step2_hold = np.squeeze(np.nan_to_num(NNweights_step2_hold,posinf=1))
        NNweights_step2_hold[pass_truth==0] = 1.
        NNweights_step2 = NNweights_step2_hold*NNweights_step2

    weights = weights_MC_sim*NNweights_step2

    # KINEMATICS
    q_perp_mag, jet_pT_mag, asymm_phi, jet_qT = get_kinematics(theta0_G)
    
    # CUTS
    cuts = get_cuts(pass_fiducial, q_perp_mag, jet_pT_mag, asymm_phi, jet_qT)


    np.save('npy_files/'+label+'_cuts.npy',cuts)
    np.save('npy_files/'+label+'_jet_pT.npy',jet_pT_mag)
    np.save('npy_files/'+label+'_q_perp.npy',q_perp_mag)
    np.save('npy_files/'+label+'_asymm_angle.npy',asymm_phi)
    np.save('npy_files/'+label+'_weights.npy',weights)
    np.save('npy_files/'+label+'_nn_weights.npy',NNweights_step2)
    np.save('npy_files/'+label+'_mc_weights.npy',weights_MC_sim)


def get_kinematics(theta0_G):
    print("Calculating q_perp, asymm_phi, and jet_pT")

    e_px = theta0_G[:,0]
    e_py = theta0_G[:,1]
    e_pT = np.array([e_px,e_py])

    jet_pT_mag = theta0_G[:,3]
    jet_phi = theta0_G[:,5]
    jet_qT = theta0_G[:,7] #[not to be confused with q_Perp!]
    
    jet_px = np.multiply(jet_pT_mag, np.cos(jet_phi))
    jet_py = np.multiply(jet_pT_mag, np.sin(jet_phi))
    jet_pT = np.array([jet_px,jet_py])

    
    q_perp_vec = jet_pT + e_pT
    q_perp_mag = np.linalg.norm(q_perp_vec,axis=0)
    P_perp_vec = (e_pT-jet_pT)/2
    P_perp_mag = np.linalg.norm(P_perp_vec,axis=0)

    q_dot_P = q_perp_vec[0,:]*P_perp_vec[0,:] + q_perp_vec[1,:]*P_perp_vec[1,:]
    
    cosphi = (q_dot_P)/(q_perp_mag*P_perp_mag)
    asymm_phi = np.arccos(cosphi)
    
    return q_perp_mag, jet_pT_mag, asymm_phi, jet_qT


def get_cuts(pass_fiducial, q_perp_mag, jet_pT_mag, asymm_phi, jet_qT):
    print("Getting Cut Mask")

    pT_cut = jet_pT_mag > 10.
    q_over_pT_cut = q_perp_mag/jet_pT_mag < 0.3 #Kyle guessed ~0.3, may need variation
    qT_cut = np.where((jet_qT<0.25), True, False)
    phi_nan_cut = ~np.isnan(asymm_phi)
    #q_perp_cut = q_perp_mag < 10.0 #q_perp_max
    
    cut_arrays = [pass_fiducial,
        pT_cut,
        q_over_pT_cut,
        qT_cut,
        phi_nan_cut]
    
    cuts = np.ones(len(pT_cut))
    
    for cut in cut_arrays:
        cuts = np.logical_and(cuts,cut)

    print("Cut Length OK = ",len(q_perp_mag)==len(cuts))

    return cuts
