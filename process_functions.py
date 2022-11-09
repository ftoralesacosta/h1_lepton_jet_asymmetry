import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

    
def averages_in_qperp_bins(dic, q_perp_bins,q_perp,asymm_phi,weights):

    digits = np.digitize(q_perp,q_perp_bins)-1
    N_Bins = len(q_perp_bins)-1
    
    q_w = q_perp*weights
    phi_w  = asymm_phi*weights
    cos1_w = np.cos(1*asymm_phi)*weights
    sin1_w = np.sin(1*asymm_phi)*weights
    cos2_w = np.cos(2*asymm_phi)*weights
    cos3_w = np.cos(3*asymm_phi)*weights

    q_avg = np.zeros(N_Bins)
    phi_avg = np.zeros(N_Bins)
    cos1_avg = np.zeros(N_Bins)
    sin1_avg = np.zeros(N_Bins)
    cos2_avg = np.zeros(N_Bins)
    cos3_avg = np.zeros(N_Bins)
    
    for i in range(N_Bins):
        bin_mask = digits==i
        bin_wsum = np.nansum(weights[bin_mask])
        
        q_avg[i] = np.nansum(q_w[bin_mask])/bin_wsum
        phi_avg[i] = np.nansum(phi_w[bin_mask])/bin_wsum
        cos1_avg[i] = np.nansum(cos1_w[bin_mask])/bin_wsum
        sin1_avg[i] = np.nansum(sin1_w[bin_mask])/bin_wsum
        cos2_avg[i] = np.nansum(cos2_w[bin_mask])/bin_wsum
        cos3_avg[i] = np.nansum(cos3_w[bin_mask])/bin_wsum

    dic["q_perp"] = q_avg
    dic["phi"] = phi_avg
    dic["cos1"] = cos1_avg
    dic["cos2"] = cos2_avg
    dic["sin1"] = sin1_avg
    dic["cos3"] = cos3_avg

    print("Keys =", dic.keys(), "N_Bins = ",N_Bins)
    return

def get_bootstrap_errors(boot_ensemble,q_perp,q_perp_bins,asymm_phi,cuts,title=""):
    #takes q_perp and and phi, and calcs weighted <cos(n*phi)> for all ensemble iterations

    N_Bootstraps = np.shape(boot_ensemble)[0]
    N_Bins = len(q_perp_bins)-1
    digits = np.digitize(q_perp,q_perp_bins)-1

    q_avg = np.zeros((N_Bootstraps,N_Bins))
    phi_avg = np.zeros((N_Bootstraps,N_Bins))
    cos1_avg = np.zeros((N_Bootstraps,N_Bins))
    cos2_avg = np.zeros((N_Bootstraps,N_Bins))
    cos3_avg = np.zeros((N_Bootstraps,N_Bins))

    #Just Used for Plotting:
    fig,axes = plt.subplots(2,2,figsize=(12,9))
    q_centers = (q_perp_bins[:-1]+q_perp_bins[1:])/2.0
    q_width = (q_perp_bins[1]+q_perp_bins[0])/2
    axes = np.ravel(axes)

    for istrap in tqdm(range(N_Bootstraps)):

        weights=boot_ensemble[istrap][cuts] #This is what fundamentally changes per iteration

        q_w = q_perp*weights
        phi_w = np.cos(asymm_phi)*weights
        cos1_w = np.cos(1*asymm_phi)*weights
        cos2_w = np.cos(2*asymm_phi)*weights
        cos3_w = np.cos(3*asymm_phi)*weights


        for i in range(N_Bins):
            bin_mask = digits==i
            bin_wsum = np.sum(weights[bin_mask])
            
            q_avg[istrap,i]    = np.nansum(q_w[bin_mask])/bin_wsum
            phi_avg[istrap,i]  = np.nansum(asymm_phi[bin_mask])/bin_wsum
            cos1_avg[istrap,i] = np.nansum(cos1_w[bin_mask])/bin_wsum
            cos2_avg[istrap,i] = np.nansum(cos2_w[bin_mask])/bin_wsum
            cos3_avg[istrap,i] = np.nansum(cos3_w[bin_mask])/bin_wsum

        axes[0].errorbar(q_centers,phi_avg[istrap],xerr=q_width,alpha=0.2)
        axes[1].errorbar(q_centers,cos1_avg[istrap],xerr=q_width,alpha=0.2)
        axes[2].errorbar(q_centers,cos2_avg[istrap],xerr=q_width,alpha=0.2)
        axes[3].errorbar(q_centers,cos3_avg[istrap],xerr=q_width,alpha=0.2)

        axes[0].set_ylabel("$\phi$")
        axes[1].set_ylabel("$\cos(\phi)$")
        axes[2].set_ylabel("$\cos(2\phi)$")
        axes[3].set_ylabel("$\cos(3\phi)$")

        axes[0].set_xlabel("$q_\perp$")
        axes[1].set_xlabel("$q_\perp$")
        axes[2].set_xlabel("$q_\perp$")
        axes[3].set_xlabel("$q_\perp$")

    plt.tight_layout()

    q_errors = np.zeros(N_Bins)
    phi_errors = np.zeros(N_Bins)
    cos1_errors = np.zeros(N_Bins)
    cos2_errors = np.zeros(N_Bins)
    cos3_errors = np.zeros(N_Bins)

    for ibin in range(N_Bins):
        q_errors[ibin] = np.nanstd(q_avg[:,ibin])/np.nanmean(q_avg[:,ibin])
        phi_errors[ibin] = np.nanstd(phi_avg[:,ibin])/np.nanmean(phi_avg[:,ibin])
        cos1_errors[ibin] = np.nanstd(cos1_avg[:,ibin])/np.nanmean(cos1_avg[:,ibin])
        cos2_errors[ibin] = np.nanstd(cos2_avg[:,ibin])/np.nanmean(cos2_avg[:,ibin])
        cos3_errors[ibin] = np.nanstd(cos3_avg[:,ibin])/np.nanmean(cos3_avg[:,ibin])

    bootstrap_errors = {}
    bootstrap_errors["q_perp"] = q_errors
    bootstrap_errors["phi"] = phi_errors
    bootstrap_errors["cos1"] = cos1_errors
    bootstrap_errors["cos2"] = cos2_errors
    bootstrap_errors["cos3"] = cos3_errors

    return bootstrap_errors
