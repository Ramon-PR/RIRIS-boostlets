import os
import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
from boostlets_mod import Boostlet_syst
from mod_plotting_utilities import plot_array_images
from mod_RIRIS_func import (
    load_DB_ZEA, jitter_downsamp_RIR, load_sk,
    computePareto, ista, iffst, linear_interpolation_fft, perforMetrics, ImageOps
)
from scipy.io import savemat


@hydra.main(version_base=None, config_path="./configs", config_name="main")
def main(cfg: DictConfig):
    # Inputs
    folder_dict = cfg.saving_param.folder
    file_dict = cfg.saving_param.file
    rm_sk_ids = cfg.rm_sk_ids  # List of removed dictionary elements
    
    # Image parameters
    room = cfg.ImOps.room
    folder = cfg.ImOps.folder
    file_path = os.path.join(folder, cfg.ImOps.get("file", f"{room}RIR.mat"))
    
    M0, N0 = cfg.ImOps.M0, cfg.ImOps.N0
    Tstart, Tend = cfg.ImOps.Tstart, cfg.ImOps.Tstart + M0
    ratio_mics, extrap_mode = cfg.ImOps.ratio_mics, cfg.ImOps.extrap_mode
    u = round(1 / ratio_mics)

    # Pareto & ISTA parameters
    beta_set = np.logspace(cfg.ista.beta_set.exp_inf, cfg.ista.beta_set.exp_sup, cfg.ista.beta_set.n_points)
    epsilon = cfg.ista.ISTAepsilon

    # Load the dictionary and image
    Sk = load_sk(folder=folder_dict, file=file_dict, build_dict=None)
    
    if not cfg.ImOps.get("file"):
        full_image = load_DB_ZEA(file_path)[0]
        orig_image = full_image[Tstart:Tend, :N0]
    else:
        orig_image = None

    mask0, _ = jitter_downsamp_RIR(orig_image.shape, ratio_t=1, ratio_x=ratio_mics)

    # Extrapolation
    extr_size = Sk.shape[:2]
    imOps = ImageOps(orig_image.shape, mask=mask0, extrap_shape=extr_size, mode=extrap_mode)
    image = imOps.expand_image(orig_image)
    mask = imOps.get_mask(image)

    # Remove selected elements from dictionary
    print(f"Removed IDs: {rm_sk_ids}")
    Sk = np.delete(Sk, rm_sk_ids, axis=2)

    # Pareto optimization
    beta_star, Jcurve = computePareto(image, mask, Sk, beta_set)

    # ISTA recovery
    alpha = ista(image, mask, Sk, beta=beta_star, epsilon=epsilon, max_iterations=15)
    image_recov = iffst(alpha, Sk)
    final_image = imOps.recover_image(image_recov)

    # Linear interpolation
    image_linear = linear_interpolation_fft(image * mask, dx=cfg.sampling.dx, fs=cfg.sampling.fs, cs=cfg.sampling.cs)
    image_lin = imOps.recover_image(image_linear)

    # Performance metrics
    NMSE_nlin, MAC, frqMAC = perforMetrics(
        image=image, image_recov=image_recov, image_under=image * mask,
        fs=cfg.sampling.fs, u=u, dx=cfg.sampling.dx, room=room
    )

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
            "Jcurve": Jcurve,
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
    images = [orig_image[:100, :], (orig_image * mask0)[:100, :], final_image[:100, :], image_lin[:100, :]]
    titles = ['Original Image', 'Masked Image', 'Reconstructed Image', 'Linear Reconstruction']
    fig, ax = plt.subplots(1, len(images), figsize=(18, 6))
    
    for i, img in enumerate(images):
        ax[i].imshow(img)
        ax[i].set_title(titles[i])
        ax[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.outputs.images.folder, cfg.outputs.images.file_im))


if __name__ == "__main__":
    main()
