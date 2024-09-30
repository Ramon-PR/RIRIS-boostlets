import os
import numpy as np
import hydra
from omegaconf import DictConfig
from scipy.io import savemat
from boostlets_mod import Boostlet_syst

# Hydra --multirun (-m)
# Reads the file ./configs/boost_sys.yaml but run this script with BS.M=120, then 140 and finally 160
# python save_boostlets_dict.py -m BS.M=120,140,160

@hydra.main(version_base=None, config_path="./configs", config_name="boost_sys")
def main(cfg: DictConfig):
    # Extract parameters from the config
    BS_params = cfg.BS
    M, N = BS_params.M, BS_params.N
    dx, fs, cs = BS_params.dx, BS_params.fs, BS_params.cs
    n_v_scales, n_h_scales = BS_params.n_v_scales, BS_params.n_h_scales
    base_v, base_h = BS_params.base_v, BS_params.base_h
    n_v_thetas, n_h_thetas = BS_params.n_v_thetas, BS_params.n_h_thetas

    # Initialize Boostlet system
    BS = Boostlet_syst(
        dx=dx, dt=1/fs, cs=cs,
        M=M, N=N, 
        n_v_scales=n_v_scales, n_h_scales=n_h_scales,
        n_v_thetas=n_v_thetas, n_h_thetas=n_h_thetas
    )

    # Retrieve boostlet dictionary
    Psi = BS.get_boostlet_dict()

    # Check if saving parameters are provided
    if "folder" in cfg.saving_param:
        folder = cfg.saving_param.folder
        BS.save_dict(folder=folder)
    else:
        BS.save_dict()

if __name__ == "__main__":
    main()

