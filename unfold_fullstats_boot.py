import sys

'''
run like python unfold_fullstats.py Rapgap nominal  0
[command] [Rapgap or Django] [label: nominal, sys_0...] [BOOTSTRAPPING Seed]
'''

print("Running on MC sample",sys.argv[1],"with setting",sys.argv[2])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
from matplotlib import gridspec
import time

from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

import os
# os.environ['CUDA_VISIBLE_DEVICES']=sys.argv[3] #"1"
os.environ['CUDA_VISIBLE_DEVICES']= "0"

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# LABEL = "Feb20_NewNominal"
# LABEL = "Feb20_sys"+sys.argv[2]
LABEL = "rerunning_systematics_"+sys.argv[2]

tf.random.set_seed(int(sys.argv[3]))
np.random.seed(int(sys.argv[3]))

def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss

    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    t_loss = -weights * ((y_true) * K.log(y_pred) +
                         (1 - y_true) * K.log(1 - y_pred))
    return K.mean(t_loss)

def reweight(events):
    f = model.predict(events, batch_size=5000)
    weights = f / (1. - f)
    return np.squeeze(np.nan_to_num(weights,posinf=1))




#Read in the data
# data = pd.read_pickle("datafiles/data.pkl")
mc = pd.read_pickle("/clusterfs/ml4hep/yxu2/unfolding_mc_inputs/"+sys.argv[1]+"_"+sys.argv[2]+".pkl")
data = pd.read_pickle("/clusterfs/ml4hep/yxu2/unfolding_mc_inputs/Data_nominal.pkl")

#Read in the MC
# mc = pd.read_pickle("/data1/bpnachman/july16/datasets/"+sys.argv[1]+"_"+sys.argv[2]+".pkl")
# mc = pd.read_pickle("/clusterfs/ml4hep/yxu2/unfolding_mc_inputs/"+sys.argv[1]+"_nominal.pkl")
# mc = pd.read_pickle("/clusterfs/ml4hep/yxu2/unfolding_mc_inputs/Django_nominal.pkl")

leading_jets_only = True
if (leading_jets_only):
    njets_tot = len(data["e_px"])
    data = data.loc[(slice(None),0), :]
    mc = mc.loc[(slice(None),0), :]
    print("Number of subjets cut = ",njets_tot-len(data["e_px"])," / ",len(data["jet_pt"]))

theta_unknown_S = data[['e_px','e_py','e_pz','jet_pt','jet_eta','jet_phi','jet_dphi','jet_qtnorm']].to_numpy()
theta0_S = mc[['e_px','e_py','e_pz','jet_pt','jet_eta','jet_phi','jet_dphi','jet_qtnorm']].to_numpy()
theta0_G = mc[['gene_px','gene_py','gene_pz','genjet_pt','genjet_eta','genjet_phi','genjet_dphi','genjet_qtnorm']].to_numpy()
weights_MC_sim = mc['wgt']
pass_reco = np.array(mc['pass_reco'])
pass_truth = np.array(mc['pass_truth'])
pass_fiducial = np.array(mc['pass_fiducial'])
print("THE LENGTH OF THE ARRAYS IS =",len(pass_fiducial))

del mc
gc.collect()

#Early stopping
# earlystopping = EarlyStopping(patience=20,
earlystopping = EarlyStopping(patience=10,
                              verbose=True,
                              restore_best_weights=True)

#Now, for the unfolding!

nepochs = 10000
NIter = 20
# nepochs = 2
# NIter = 2

bins = np.logspace(np.log10(0.03),np.log10(3.03),9) - 0.03
bins = bins[1:]
bins[0] = 0.0
starttime = time.time()

NNweights_step2 = np.ones(len(theta0_S))
np.save("./new_models/test_weights_ones.npy",NNweights_step2)

#Set up the model
inputs = Input((8, ))
hidden_layer_1 = Dense(50, activation='relu')(inputs)
hidden_layer_2 = Dense(100, activation='relu')(hidden_layer_1)
hidden_layer_3 = Dense(50, activation='relu')(hidden_layer_2)
outputs = Dense(1, activation='sigmoid')(hidden_layer_3)
mymodel = Model(inputs=inputs, outputs=outputs)

dataw = np.ones(len(theta_unknown_S))
if (int(sys.argv[3])!=0):
    dataw = np.random.poisson(1,len(theta_unknown_S))
    print("Doing Bootstrapping")
else:
    print("Not doing bootstrapping")

N_passes = 5 #average models trained on same exact input
for pass_i in range(N_passes):
    #logfile
    logfile = open("log_files/"+sys.argv[1]+"_"+sys.argv[2]+"_"+sys.argv[3]+"Pass%i"%(pass_i)+".txt","w")
    logfile.write("Running on MC sample "+sys.argv[1]+" with setting "+sys.argv[2]+" seed "+sys.argv[3]+"Pass%i"%(pass_i)+"\n")
    for iteration in range(NIter):


        #Process the data
        print("on iteration=",iteration," processing data for step 1, time elapsed=",time.time()-starttime)
        logfile.write("on iteration="+str(iteration)+" processing data for step 1, time elapsed="+str(time.time()-starttime)+"\n")

        xvals_1 = np.concatenate([theta0_S[pass_reco==1],theta_unknown_S])
        yvals_1 = np.concatenate([np.zeros(len(theta0_S[pass_reco==1])),np.ones(len(theta_unknown_S))])
        weights_1 = np.concatenate([NNweights_step2[pass_reco==1]*weights_MC_sim[pass_reco==1],dataw])

        scaler_data = StandardScaler()
        scaler_data.fit(theta_unknown_S)
        xvals_1 = scaler_data.transform(xvals_1)

        X_train_1, X_test_1, Y_train_1, Y_test_1, w_train_1, w_test_1 = train_test_split(xvals_1, yvals_1, weights_1,test_size=0.5)
        del xvals_1,yvals_1,weights_1
        gc.collect()

        Y_train_1 = np.stack((Y_train_1, w_train_1), axis=1)
        Y_test_1 = np.stack((Y_test_1, w_test_1), axis=1)
        del w_train_1,w_test_1
        gc.collect()

        print("on iteration=",iteration," done processing data for step 1, time elapsed=",time.time()-starttime)
        print("data events = ",len(X_train_1[Y_train_1[:,0]==1]))
        print("MC events = ",len(X_train_1[Y_train_1[:,0]==0]))

        logfile.write("on iteration="+str(iteration)+" done processing data for step 1, time elapsed="+str(time.time()-starttime)+"\n")
        logfile.write("data events = "+str(len(X_train_1[Y_train_1[:,0]==1]))+"\n")
        logfile.write("MC events = "+str(len(X_train_1[Y_train_1[:,0]==0]))+"\n")
        
        #Step 1
        print("on step 1, time elapsed=",time.time()-starttime)
        logfile.write("on step 1, time elapsed="+str(time.time()-starttime)+"\n")
        
        # opt = tf.keras.optimizers.Adam(learning_rate=2e-6)
        opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        mymodel.compile(loss=weighted_binary_crossentropy,
                          optimizer=opt,
                          metrics=['accuracy'])

        hist_s1 =  mymodel.fit(X_train_1,Y_train_1,
                  epochs=nepochs,
                  batch_size=50000,
                  validation_data=(X_test_1,Y_test_1),
                  callbacks=[earlystopping],
                  verbose=1)

        print("done with step 1, time elapsed=",time.time()-starttime)
        logfile.write("done with step 1, time elapsed="+str(time.time()-starttime)+"\n")
        
        #Now, let's do some checking.

        ###
        # Loss
        ###

        fig = plt.figure(figsize=(7, 5)) 
        gs = gridspec.GridSpec(1, 1, height_ratios=[1]) 
        ax0 = plt.subplot(gs[0])
        ax0.yaxis.set_ticks_position('both')
        ax0.xaxis.set_ticks_position('both')
        ax0.tick_params(direction="in",which="both")
        ax0.minorticks_on()
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.plot(np.array(hist_s1.history['loss']),label="loss")
        plt.plot(np.array(hist_s1.history['val_loss']),label="val. loss",ls=":")
        plt.xlabel("Epoch",fontsize=20)
        plt.ylabel("Loss",fontsize=20)
        plt.title("OmniFold iteration "+str(iteration)+", step 1",loc="left",fontsize=20)
        plt.text(0.05, 1.15,'H1', horizontalalignment='center', verticalalignment='center', transform = ax0.transAxes, fontsize=25, fontweight='bold')
        plt.legend(frameon=False,fontsize=20)
        plt.locator_params(axis='x', nbins=5)
        plt.xscale("log")
        fig.savefig("./storage_plots/"+sys.argv[1]+"_"+sys.argv[2]+"Iteration"+str(iteration)+"_"+sys.argv[3]+"_Step1_loss.pdf",bbox_inches='tight')
        
        mypred = mymodel.predict(scaler_data.transform(np.nan_to_num(theta0_S,posinf=0,neginf=0)),batch_size=10000)
        mypred = mypred/(1.-mypred)
        mypred = mypred[:,0]
        mypred = np.squeeze(np.nan_to_num(mypred,posinf=1))


##############
### STEP 2 ###
##############

        print("time for step 2, time elapsed=",time.time()-starttime)
        logfile.write("time for step 2, time elapsed="+str(time.time()-starttime)+"\n")
        
        xvals_2 = np.concatenate([theta0_G[pass_truth==1],theta0_G[pass_truth==1]])
        yvals_2 = np.concatenate([np.zeros(len(theta0_G[pass_truth==1])),np.ones(len(theta0_G[pass_truth==1]))])

        xvals_2 = scaler_data.transform(xvals_2)

        NNweights = mymodel.predict(scaler_data.transform(np.nan_to_num(theta0_S[pass_truth==1],posinf=0,neginf=0)),batch_size=10000)
        NNweights = NNweights/(1.-NNweights)
        NNweights = NNweights[:,0]
        NNweights = np.squeeze(np.nan_to_num(NNweights,posinf=1))
        NNweights[pass_reco[pass_truth==1]==0] = 1.
        weights_2 = np.concatenate([NNweights_step2[pass_truth==1]*weights_MC_sim[pass_truth==1],NNweights*NNweights_step2[pass_truth==1]*weights_MC_sim[pass_truth==1]])

        X_train_2, X_test_2, Y_train_2, Y_test_2, w_train_2, w_test_2 = train_test_split(xvals_2, yvals_2, weights_2,test_size=0.5)
        del xvals_2,yvals_2,weights_2
        gc.collect()

        Y_train_2 = np.stack((Y_train_2, w_train_2), axis=1)
        Y_test_2 = np.stack((Y_test_2, w_test_2), axis=1)
        del w_train_2,w_test_2
        gc.collect()

        print("on iteration=",iteration," done processing data for step 2, time elapsed=",time.time()-starttime)
        print("MC events = ",len(X_train_1[Y_train_1[:,0]==1]))
        print("MC events = ",len(X_train_1[Y_train_1[:,0]==0]))

        logfile.write("on iteration="+str(iteration)+" done processing data for step 2, time elapsed="+str(time.time()-starttime)+"\n")
        logfile.write("MC events = "+str(len(X_train_1[Y_train_1[:,0]==1]))+"\n")
        logfile.write("MC events = "+str(len(X_train_1[Y_train_1[:,0]==0]))+"\n")
        
        #step 2
        # opt = tf.keras.optimizers.Adam(learning_rate=5e-6)
        opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        mymodel.compile(loss=weighted_binary_crossentropy,
                          optimizer=opt,
                          metrics=['accuracy'])

        hist_s2 =  mymodel.fit(X_train_2,Y_train_2,
                  epochs=nepochs,
                  batch_size=100000,
                  validation_data=(X_test_2,Y_test_2),
                  callbacks=[earlystopping],
                  verbose=1)

        print("on iteration=",iteration," finished step 2; time elapsed=",time.time()-starttime)
        logfile.write("on iteration="+str(iteration)+" finished step 2; time elapsed="+str(time.time()-starttime)+"\n")
        
        NNweights_step2_hold = mymodel.predict(scaler_data.transform(theta0_G),batch_size=10000)
        NNweights_step2_hold = NNweights_step2_hold/(1.-NNweights_step2_hold)
        NNweights_step2_hold = NNweights_step2_hold[:,0]
        NNweights_step2_hold = np.squeeze(np.nan_to_num(NNweights_step2_hold,posinf=1))
        NNweights_step2_hold[pass_truth==0] = 1.
        NNweights_step2 = NNweights_step2_hold*NNweights_step2
        tf.keras.models.save_model(mymodel,"./new_models/"+sys.argv[1]+"_"+sys.argv[2]+"_"+LABEL+"Iteration_%i"%(iteration)+f"Pass{pass_i}"+"model")

        #Now, let's do some checking.

        ###
        # Loss
        ###

        fig = plt.figure(figsize=(7, 5)) 
        gs = gridspec.GridSpec(1, 1, height_ratios=[1]) 
        ax0 = plt.subplot(gs[0])
        ax0.yaxis.set_ticks_position('both')
        ax0.xaxis.set_ticks_position('both')
        ax0.tick_params(direction="in",which="both")
        ax0.minorticks_on()
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.plot(np.array(hist_s2.history['loss']),label="loss")
        plt.plot(np.array(hist_s2.history['val_loss']),label="val. loss",ls=":")
        plt.xlabel("Epoch",fontsize=20)
        plt.ylabel("Loss",fontsize=20)
        plt.title("OmniFold iteration "+str(iteration)+", step 2",loc="left",fontsize=20)
        plt.text(0.05, 1.15,'H1', horizontalalignment='center', verticalalignment='center', transform = ax0.transAxes, fontsize=25, fontweight='bold')
        plt.legend(frameon=False,fontsize=20)
        plt.locator_params(axis='x', nbins=5)
        plt.xscale("log")
        fig.savefig("./storage_plots/"+sys.argv[1]+"_"+sys.argv[2]+"Iteration"+str(iteration)+"_"+sys.argv[3]+"_Step2_loss.pdf",bbox_inches='tight')

        ###
        # qT
        ###

        fig = plt.figure(figsize=(7, 5)) 
        gs = gridspec.GridSpec(2, 1, height_ratios=[2,1]) 
        ax0 = plt.subplot(gs[0])
        ax0.yaxis.set_ticks_position('both')
        ax0.xaxis.set_ticks_position('both')
        ax0.tick_params(direction="in",which="both")
        ax0.minorticks_on()
        plt.xticks(fontsize=0)
        plt.yticks(fontsize=20)

        n_data,b,_=plt.hist(theta0_G[pass_fiducial==1][:,7],bins=bins,weights=weights_MC_sim[pass_fiducial==1]*NNweights[pass_fiducial==1]*NNweights_step2[pass_fiducial==1]/NNweights_step2_hold[pass_fiducial==1],density=True,alpha=0.5,label="MC + step 1")
        n_MC,_,_=plt.hist(theta0_G[pass_fiducial==1][:,7],bins=bins,weights=weights_MC_sim[pass_fiducial==1],density=True,histtype="step",color="black",label="MC")
        n_Omni_step2_qT,_,_=plt.hist(theta0_G[pass_fiducial==1][:,7],bins=bins,weights=weights_MC_sim[pass_fiducial==1]*NNweights_step2[pass_fiducial==1],density=True,histtype="step",color="black",ls=":",label="MC + step 2")

        plt.ylabel("Normalized to unity",fontsize=20)
        plt.title("OmniFold iteration "+str(iteration)+", step 2",loc="left",fontsize=20)
        plt.text(0.05, 1.25,'H1', horizontalalignment='center', verticalalignment='center', transform = ax0.transAxes, fontsize=25, fontweight='bold')
        plt.legend(frameon=False,fontsize=15)
        plt.locator_params(axis='x', nbins=5)
        plt.yscale("log")

        ax1 = plt.subplot(gs[1])
        ax1.yaxis.set_ticks_position('both')
        ax1.xaxis.set_ticks_position('both')
        ax1.tick_params(direction="in",which="both")
        ax1.minorticks_on()
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)

        plt.xlabel("Particle-level $q_T/Q$",fontsize=15)
        plt.ylabel("step 2/step 1",fontsize=15)

        plt.plot(0.5*(b[0:-1]+b[1:]),n_MC/n_data,ls="--",color="black")
        plt.plot(0.5*(b[0:-1]+b[1:]),n_Omni_step2_qT/n_data,ls=":",color="black")

        fig.savefig("./storage_plots/"+sys.argv[1]+"_"+sys.argv[2]+"_Iteration"+str(iteration)+"_"+sys.argv[3]+"_Step2_qT.pdf",bbox_inches='tight')
        # np.save("./storage_files/"+sys.argv[1]+"_"+sys.argv[2]+"_Omni_step2_qT_iteration"+str(iteration)+"_"+sys.argv[3],n_Omni_step2_qT)
        
        print("done with the "+str(iteration)+"iteration, time elapsed=",time.time()-starttime)
        logfile.write("done with the "+str(iteration)+"iteration, time elapsed="+str(time.time()-starttime)+"\n")

        tensorflow.keras.backend.clear_session()
        
        pass
        #end OF Loop

    np.save("./new_models/"+LABEL+sys.argv[1]+"_"+sys.argv[2]+"_NNweights_step2_Pass%i.npy"%(pass_i),NNweights_step2)
    #end pass loop

    # tf.keras.models.save_model(mymodel,"./models/Jan20_"+sys.argv[1]+"_"+sys.argv[2]+"_"+"Pass%i"%(pass_i)+"model")
logfile.close()
print("FINISHED")
