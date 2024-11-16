import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import hydra
from omegaconf import DictConfig
from boostlets_mod import Boostlet_syst, rm_sk_index_in_horiz_cone
from mod_plotting_utilities import plot_array_images
from mod_RIRIS_func import (
    load_DB_ZEA, jitter_downsamp_RIR, load_sk,
    computePareto, ista, iffst, linear_interpolation_fft, perforMetrics, ImageOps, ffst, calculate_NMSE, calculate_MAC
)

np.random.seed(42)

def vars_from_dictname(fname):
    # Convertir a string
    cadena = str(fname)
    # Encontrar posiciones
    inicio = cadena.find('S_')
    fin = cadena.rfind('.mat')
    # Extraer el substring
    fname = cadena[inicio:fin]

    # Dividir el nombre del archivo por los guiones bajos
    partes = fname.split('_')

    # Inicializar un diccionario para almacenar los valores
    sk_params = {}

    # Extraer las sk_params
    for i in range(len(partes)):
        if partes[i] == 'm':
            sk_params['M'] = int(partes[i + 1])
        elif partes[i] == 'n':
            sk_params['N'] = int(partes[i + 1])
        elif partes[i] == 'vsc':
            sk_params['n_v_scales'] = int(partes[i + 1])
        elif partes[i] == 'hsc':
            sk_params['n_h_scales'] = int(partes[i + 1])
        elif partes[i] == 'bases':
            bases = [float(partes[i + 1]), float(partes[i + 2])]
            sk_params['bases'] = bases
        elif partes[i] == 'thV':
            sk_params['n_v_thetas'] = int(partes[i + 1])
        elif partes[i] == 'thH':
            sk_params['n_h_thetas'] = int(partes[i + 1])

    return sk_params

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
    dt = 1 / fs

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

    if not cfg.get("file_im"):  # If not given "file_im" load and select image from full RIR
        full_image = load_DB_ZEA(file_path)[0]
        orig_image = full_image[Tstart:Tend, :N0]
    else:  # If we give "file_im" then load the image and its parameters
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
    # image_linear = linear_interpolation_fft(image * mask, dx=dx, fs=fs, cs=cs)
    # image_lin = imOps.recover_image(image_linear)

    # Performance metrics
    # NMSE_nlin, MAC, frqMAC = perforMetrics(
    #     image=image, image_recov=image_recov, image_under=image * mask,
    #     fs=fs, u=u, dx=dx, room=room
    # )
    # print(f"NMSE: lin = {NMSE_nlin[0]} / boostlet = {NMSE_nlin[1]}")
    NMSE = calculate_NMSE(orig_image, final_image)
    NMSE = calculate_NMSE(orig_image, final_image)
    frqMAC, MAC = calculate_MAC(orig_image, final_image, fs)

    # Sparsity
    # alpha0 = ffst(image, Sk)
    # non_zero_FIm = np.sum(np.abs(np.fft.fft2(image).flatten()) > 0)
    # non_zero_alpha0 = np.sum(np.abs(alpha0.flatten()) > 0)
    # non_zero_alpha = np.sum(np.abs(alpha.flatten()) > 0)

    # Convert results to DataFrame
    results = {
        "dic_name": [file_dict],
        "rm_sk_ids": [list(rm_sk_ids)],
        "beta_star": [beta_star],
        # "NMSE_lin": [NMSE_nlin[0]],
        "NMSE": [NMSE],
        "frqMAC": [frqMAC],
        # "MAC_lin": [MAC[0]],
        "MAC": [MAC[1]],
        "beta_set": [list(beta_set)],
        "Jcurve": [list(Jcurve)],
        "rho": [list(rho)],
        "eta": [list(eta)],
        # "non_zero_FIm": [non_zero_FIm],
        # "non_zero_alpha0": [non_zero_alpha0],
        # "non_zero_alpha": [non_zero_alpha],
        "room": [room],
        "Tstart": [Tstart],
        "ratio_mics": [ratio_mics],
    }
    df = pd.DataFrame(results)

    # Save DataFrame to CSV
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "results.csv")
    df.to_csv(output_file, index=False)

    print(f"Results saved to {output_file}")

    # Save images (optional visualization)
    # image_folder = os.path.join(output_dir, "images")
    # os.makedirs(image_folder, exist_ok=True)
    # images = [(orig_image * mask0)[:100, :], image_lin[:100, :], final_image[:100, :], orig_image[:100, :]]
    # titles = ['Masked Image', "Linear reconst", 'Final reconst image', 'Original Image']

    # fig, ax = plt.subplots(1, len(images), figsize=(18, 6))
    # for i, img in enumerate(images):
    #     ax[i].imshow(img)
    #     ax[i].set_title(titles[i])
    #     ax[i].axis('off')
    # plt.tight_layout()
    # plt.savefig(os.path.join(image_folder, "reconstruction_results.png"))

if __name__ == "__main__":
    main()
