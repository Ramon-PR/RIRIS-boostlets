# %% [markdown]
# # Imports 

import os
import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig

from boostlets_mod import Boostlet_syst
from mod_plotting_utilities import plot_array_images
from mod_RIRIS_func import load_DB_ZEA, rand_downsamp_RIR, ImageOps, jitter_downsamp_RIR, load_sk
from mod_RIRIS_func import computePareto, ista, iffst, linear_interpolation_fft, perforMetrics


# %% [markdown]
# # Hydra: load configuration from YAML

@hydra.main(version_base=None, config_path="./configs", config_name="main")
def main(cfg: DictConfig):
    
    # ------------- Inputs dictionary -----------------------------
    folder_dict = cfg.saving_param.folder
    file_dict = cfg.saving_param.file
    # -------------------------------------------------------------
    
    # ----------------- Image ------------------------------------- 
    room = cfg.ImOps.room
    folder = cfg.ImOps.folder

    # sampling
    dx = cfg.sampling.dx
    fs = cfg.sampling.fs
    cs = cfg.sampling.cs

    if "file" in cfg.ImOps :
        file = cfg.ImOps.file
    else:
        file = room + "RIR.mat"
    file_path = os.path.join(folder, file)

    M0 = cfg.ImOps.M0 # number of rows
    N0 = cfg.ImOps.N0 # number of cols
    Tstart = cfg.ImOps.Tstart
    Tend = Tstart + M0
    
    ratio_mics = cfg.ImOps.ratio_mics
    u = round(1/ratio_mics)
    extrap_mode = cfg.ImOps.extrap_mode    
    # -------------------------------------------------------------

    # ----------  Pareto & Ista -----------------------------------
    beta_set = np.logspace(cfg.ista.beta_set.exp_inf,
                           cfg.ista.beta_set.exp_sup,
                           cfg.ista.beta_set.n_points)
    # ISTA
    epsilon = cfg.ista.ISTAepsilon
    # -------------------------------------------------------------

    # ----------------- Outputs -----------------------------------
    output_folder = cfg.outputs.images.folder
    fname_images_reconst = cfg.outputs.images.file

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    # -------------------------------------------------------------


    # %% [markdown]
    # ## Load dict
    Sk = load_sk(folder=folder_dict , file=file_dict , build_dict=None)

    # ## Load Image

    if "file" in cfg.ImOps :
        pass
        # print("Image loaded:")
        # print(file_path)
    else:
        # Load full image and select a subimage to apply decomposition
        full_image = load_DB_ZEA(file_path)[0]
        orig_image = full_image[Tstart:Tend, :N0]
        print("Image loaded:")
        print(file_path)

    # mask0, _ = rand_downsamp_RIR(orig_image.shape, ratio_t=1, ratio_x=ratio_mics)
    mask0, _ = jitter_downsamp_RIR(orig_image.shape, ratio_t=1, ratio_x=ratio_mics)


    # %% [markdown]
    # ## Extrapolation

    extr_size = Sk.shape[:2]
    imOps = ImageOps(orig_image.shape, mask=mask0, extrap_shape=extr_size, mode=extrap_mode) 
    image = imOps.expand_image(orig_image)
    mask = imOps.get_mask(image)

    # %% [markdown]
    # ## Remove elements from dict

    # ----------------------------------------------------
    # Pareto
    # ----------------------------------------------------
    beta_star, Jcurve = computePareto(image, mask, Sk, beta_set)

    # ----------------------------------------------------
    # ISTA recovery
    # ----------------------------------------------------
    alpha = ista(image, mask, Sk, beta=beta_star, epsilon=epsilon, max_iterations=15 )

    # recover inpainted image from sparse coefficients (Eq. 19)
    image_recov = iffst(alpha, Sk)
    final_image = imOps.recover_image(image_recov)

    # ## Linear interpolation
    image_linear = linear_interpolation_fft(image*mask, dx=dx, fs=fs, cs=cs)
    image_lin = imOps.recover_image(image_linear)

    NMSE_nlin, MAC, frqMAC = perforMetrics(image=image, image_recov=image_recov, 
                                        image_under=image*mask, 
                                        fs=fs, u=u, dx=dx, room=room)

    # %% [markdown]
    # ## Visual results

    images = [orig_image[:100,:], (orig_image*mask0)[:100,:], final_image[:100,:], image_lin[:100,:]]
    titles = ['Original Image', 'Masked Image', 'Final reconst image', "Linear reconst"]

    fig, ax = plt.subplots(1, len(images), figsize=(18, 6))
    for i in range(len(images)):
        ax[i].imshow(images[i])
        ax[i].set_title(titles[i])
        ax[i].axis('off')
    # Save the figure
    output_image_path = os.path.join(output_folder, f"{fname_images_reconst}.png")
    plt.tight_layout()
    plt.savefig(output_image_path)
    print(f"Image saved to: {output_image_path}")

if __name__ == "__main__":
    main()
