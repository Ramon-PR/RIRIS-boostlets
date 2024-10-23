import os
import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
from boostlets_mod import Boostlet_syst, rm_sk_index_in_horiz_cone
from mod_plotting_utilities import plot_array_images
from mod_RIRIS_func import (
    load_DB_ZEA, jitter_downsamp_RIR, load_sk,
    computePareto, ista, iffst, linear_interpolation_fft, perforMetrics, ImageOps, ffst,
)
from scipy.io import savemat

# python ./main.py folder_dict=saved_dicts/tan_dicts file_dict=BS_m_128_n_128_vsc_2_hsc_2_bases_0.5_0.5_thV_15_thH_15.mat
# Check also run_main_multiple_dicts.py

@hydra.main(version_base=None, config_path="./configs", config_name="main")
def main(cfg: DictConfig):
    # Inputs
    folder_dict = cfg.folder_dict
    file_dict = cfg.file_dict
    
    # Image parameters
    room = cfg.database.room
    folder = cfg.database.folder
    file_path = os.path.join(folder, cfg.database.get("file", f"{room}RIR.mat"))
    dx = cfg.database.sampling.dx
    fs = cfg.database.sampling.fs
    cs = cfg.database.sampling.cs
    dt = 1/fs
    
    M0, N0 = cfg.database.M0, cfg.database.N0
    Tstart, Tend = cfg.database.Tstart, cfg.database.Tstart + M0
    ratio_mics, extrap_mode = cfg.subsampling.ratio_mics, cfg.subsampling.extrap_mode
    u = round(1 / ratio_mics)

    # Pareto & ISTA parameters
    beta_set = np.logspace(cfg.ista.beta_set.exp_inf, cfg.ista.beta_set.exp_sup, cfg.ista.beta_set.n_points)
    epsilon = cfg.ista.ISTAepsilon

    # Load the dictionary and image
    print(folder_dict)
    print(file_dict)
    Sk = load_sk(folder=folder_dict, file=file_dict, build_dict=None)
    
    if not cfg.get("file_im"): # If not given "file_im" load and select image from full RIR
        full_image = load_DB_ZEA(file_path)[0]
        orig_image = full_image[Tstart:Tend, :N0]
    else: # If we give "file_im" then load the image and its parameters
        orig_image = None

    mask0, _ = jitter_downsamp_RIR(orig_image.shape, ratio_t=1, ratio_x=ratio_mics)

    # Extrapolation
    extr_size = Sk.shape[:2]
    imOps = ImageOps(orig_image.shape, mask=mask0, extrap_shape=extr_size, mode=extrap_mode)
    image = imOps.expand_image(orig_image)
    mask = imOps.get_mask(image)

    # Remove selected elements from dictionary
    rm_sk_ids = rm_sk_index_in_horiz_cone(dx=dx, dt=dt, cs=cs, Sk=Sk)
    print(f"Removed IDs: {rm_sk_ids}")
    Sk = np.delete(Sk, rm_sk_ids, axis=2)

    # Pareto optimization
    beta_star, Jcurve, rho, eta = computePareto(image, mask, Sk, beta_set, f_Lcurve=True)

    # ISTA recovery
    alpha = ista(image, mask, Sk, beta=beta_star, epsilon=epsilon, max_iterations=15)
    image_recov = iffst(alpha, Sk)
    final_image = imOps.recover_image(image_recov)

    # Linear interpolation
    image_linear = linear_interpolation_fft(image * mask, dx=dx, fs=fs, cs=cs)
    image_lin = imOps.recover_image(image_linear)

    # Performance metrics
    NMSE_nlin, MAC, frqMAC = perforMetrics(
        image=image, image_recov=image_recov, image_under=image * mask,
        fs=fs, u=u, dx=dx, room=room
    )
    print(f"NMSE: lin = {NMSE_nlin[0]} / boostlet = {NMSE_nlin[1]}")

    # Sparsity
    alpha0 = ffst(image, Sk)
    non_zero_FIm = np.sum(np.abs(np.fft.fft2(image).flatten())>0)
    non_zero_alpha0 = np.sum(np.abs(alpha0.flatten())>0)
    non_zero_alpha = np.sum(np.abs(alpha.flatten())>0)

    # Save performance outputs
    if cfg.outputs.performance.f_write_dict:
        perf_outputs = {
            "dic_name": file_dict,
            "rm_sk_ids": rm_sk_ids,
            "beta_star": beta_star,
            "NMSE_lin": NMSE_nlin[0],
            "NMSE": NMSE_nlin[1],
            "frqMAC": frqMAC,
            "MAC_lin": MAC[0],
            "MAC": MAC[1],
            "beta_set": beta_set,
            "Jcurve": Jcurve,
            "rho": rho, # ||(im - im*)*mask||_2 (beta_i) Pareto
            "eta": eta, # ||alpha||_1 (beta_i)  Pareto
            "non_zero_FIm": non_zero_FIm, # Sparsity factors
            "non_zero_alpha0": non_zero_alpha0, # Sparsity factors
            "non_zero_alpha": non_zero_alpha, # Sparsity factors
        }
        os.makedirs(cfg.outputs.performance.folder, exist_ok=True)
        savemat(os.path.join(cfg.outputs.performance.folder, cfg.outputs.performance.file), perf_outputs)

    # Save image outputs
    if cfg.outputs.images.f_write_dict:
        image_outputs = {
            "orig_image": orig_image,
            "masked_image": orig_image * mask0,
            "final_image": final_image,
            "image_lin": image_lin,
        }
        os.makedirs(cfg.outputs.images.folder, exist_ok=True)
        savemat(os.path.join(cfg.outputs.images.folder, cfg.outputs.images.file_mat), image_outputs)

    # Visualize results
    images = [(orig_image*mask0)[:100, :], image_lin[:100, :], final_image[:100, :], orig_image[:100, :]]
    titles = ['Masked Image', "Linear reconst", 'Final reconst image', 'Original Image']

    fig, ax = plt.subplots(1, len(images), figsize=(18, 6))
    for i, img in enumerate(images):
        ax[i].imshow(img)
        ax[i].set_title(titles[i])
        ax[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.outputs.images.folder, cfg.outputs.images.file_im))


if __name__ == "__main__":
    main()
