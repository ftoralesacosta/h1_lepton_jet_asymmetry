import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import os
os.environ['CUDA_VISIBLE_DEVICES']="3"
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

mc = pd.read_pickle("/clusterfs/ml4hep/yxu2/unfolding_mc_inputs/Rapgap_nominal.pkl")
data = pd.read_pickle("/clusterfs/ml4hep/yxu2/unfolding_mc_inputs/Data_nominal.pkl")

# Apply Cuts
cut_subleading_jet = True
if cut_subleading_jet:
    mc = mc.loc[(slice(None), 0), :]
    data = data.loc[(slice(None), 0), :]

pass_fiducial = np.array(mc['pass_fiducial'])
pass_truth = np.array(mc['pass_truth'])

theta0_G = mc[['gene_px','gene_py','gene_pz','genjet_pt','genjet_eta','genjet_phi','genjet_dphi','genjet_qtnorm']].to_numpy()
# Q2 = theta0_G[:,8]
# Q2 = Q2[pass_fiducial==1]
# theta0_G = theta0_G[:,0:8]

theta_unknown_S = data[['e_px','e_py','e_pz','jet_pt','jet_eta','jet_phi','jet_dphi','jet_qtnorm']].to_numpy()

# Get StandardScalar from raw data, apply to Rapgap in each iteration later
scaler_data = StandardScaler()
scaler_data.fit(theta_unknown_S)
print("\n Length of theta0G =", len(theta0_G), "\n")

model_dir="/clusterfs/ml4hep_nvme2/ftoralesacosta/h1_check/h1_lepton_jet_asymmetry"
base_string = f"{model_dir}/new_models/Rapgap_nominal_rerunning_systematics_nominalIteration"
n_iterations = 20
n_passes = 5
# run_iter = 4
pass_weights = []
# for pass_iter in tqdm(np.concatenate([range(1,30),range(34,45),[46,47,48,49],range(54,66),range(80,86),range(100,106),range(120,126)])):

for pass_iter in range(n_passes):

    iteration_weights = []
    # Make sure to reset weights
    NNweights_step2 = np.ones(len(theta0_G))
    NNweights_step2_hold = np.ones(len(theta0_G))

    for run_iter in range(n_iterations):
        # Rapgap_nominal_rerunning_systematics_nominalIteration_19Pass4model
        model_string = f"{base_string}_{run_iter}Pass{pass_iter}model"
        print(f"Loading {model_string}")

        mymodel = tf.keras.models.load_model(model_string, compile=False)
        # mymodel = tf.keras.models.load_model(
        #     "/clusterfs/ml4hep/yxu2/inputfiles/fullscan_stat/models/Rapgap_nominal_iteration"+str(run_iter)+"_"+str(pass_iter)+"_step2", compile=False)

        NNweights_step2_hold = mymodel.predict(scaler_data.transform(theta0_G), batch_size=10000)
        NNweights_step2_hold = NNweights_step2_hold/(1.-NNweights_step2_hold)
        NNweights_step2_hold = NNweights_step2_hold[:, 0]
        NNweights_step2_hold = np.squeeze(np.nan_to_num(NNweights_step2_hold, posinf=1))
        NNweights_step2_hold[pass_truth == 0] = 1.
        NNweights_step2 = NNweights_step2_hold*NNweights_step2

        tf.keras.backend.clear_session()

        iteration_weights.append(NNweights_step2)
        print(f"Shape of Iteration Array = {np.shape(iteration_weights)}")

    pass_weights.append(iteration_weights)
    print(f"Shape of Pass Array = {np.shape(pass_weights)}")

pass_weights = np.asarray(pass_weights)
np.save("/global/ml4hep/spss/ftoralesacosta/passes_rapgap_weights.npy", pass_weights)
