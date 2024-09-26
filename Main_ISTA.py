import os
import numpy as np
import matplotlib.pyplot as plt

from boostlets_mod import Boostlet_syst
from mod_plotting_utilities import plot_array_images
from mod_RIRIS_func import load_DB_ZEA, rand_downsamp_RIR, ImageOps, jitter_downsamp_RIR
from mod_RIRIS_func import computePareto, ista, iffst, linear_interpolation_fft, perforMetrics


# %% [markdown]
# # Inputs 
# Dictionary and Image

# %%
# Tamaño diccionario // Tamaño imagen interpolada
M, N  = 128, 128
# Imagen tamaño:
M0, N0 = 100, 100

# Dictionary
n_v_scales, n_h_scales = 10, 10  
base_v, base_h = 1/2, 1/2
n_v_thetas, n_h_thetas = 3, 3 

# Image
room = "Balder"
# //////////////////////////////////////
ratio_undersampling = 0.3
u = round(1/ratio_undersampling)
ratio_mics = 0.5
# //////////////////////////////////////

extrap_mode = "extrapolate" # or pad

# Seleccionar subimage 
Tstart = 0
Tend = Tstart+M0

# sampling 
dx=3e-2
fs=11250
cs=340



# %% [markdown]
# ## Create dict

# %%
BS = Boostlet_syst(dx=dx, dt=1/fs, cs=cs,
                 M=M, N=N, 
                 n_v_scales=n_v_scales, n_h_scales=n_h_scales, 
                 base_v=base_v, base_h=base_h, 
                 )

BS.print_max_scales()
Sk = BS.get_boostlet_dict()

plot_array_images(Sk)

# %% [markdown]
# ## Load Image

# %%
# ---------- LOAD Image ---------------------------
folder = "./dependencies/measurementData"
file = room+"RIR.mat"
file_path = os.path.join(folder, file)
print("Image loaded:")
print(file_path)

# Load full image and select a subimage to apply decomposition
full_image = load_DB_ZEA(file_path)[0]
orig_image = full_image[Tstart:Tend, :N0]

# mask0, _ = rand_downsamp_RIR(orig_image.shape, ratio_t=1, ratio_x=ratio_undersampling)
mask0, _ = jitter_downsamp_RIR(orig_image.shape, ratio_t=1, ratio_x=ratio_mics)

# fig, ax = plt.subplots(1,2)
# ax[0].imshow(orig_image[Tstart:Tstart+100, :])
# ax[1].imshow((orig_image*mask0)[Tstart:Tstart+100, :])

# ax[0].axis('off')
# ax[1].axis('off')
# plt.show()

# %% [markdown]
# ## Remove elements from dict

# %%




# %% [markdown]
# ## Extrapolation

# %%

# ----------------------------------------------------
# Extrapolation
# ----------------------------------------------------

extr_size = Sk.shape[:2]
imOps = ImageOps(orig_image.shape, mask=mask0, extrap_shape=extr_size, mode=extrap_mode) 
image = imOps.expand_image(orig_image)
mask = imOps.get_mask(image)

images = [orig_image, image, mask*image]
titles = ['Original Image', 'Expanded Image', 'Masked image']
fig, ax = plt.subplots(1, len(images), figsize=(18, 6))
for i in range(len(images)):
    ax[i].imshow(images[i][:128,:])
    ax[i].set_title(titles[i])
    ax[i].axis('off')
plt.tight_layout()
plt.show()

# %% [markdown]
# # ISTA
# 

# %%
from mod_RIRIS_func import computePareto
# ----------------------------------------------------
# Pareto
# ----------------------------------------------------

beta_set = np.logspace(-2.5, -1, 50)
beta_star, Jcurve = computePareto(image, mask, Sk, beta_set)

# %%

# ----------------------------------------------------
# ISTA recovery
# ----------------------------------------------------
epsilon = 9.4e-6
alpha = ista(image, mask, Sk, beta=beta_star, epsilon=epsilon, max_iterations=15 )


# %% [markdown]
# ## Recover image

# %%
# recover inpainted image from sparse coefficients (Eq. 19)
image_recov = iffst(alpha, Sk)
final_image = imOps.recover_image(image_recov)


# %% [markdown]
# ## Linear interpolation

# %%
image_linear = linear_interpolation_fft(image*mask, dx=dx, fs=fs, cs=cs)
image_lin = imOps.recover_image(image_linear)


# %% Performance Metrics
image_under = image*mask
u = 3 #undersampling value
NMSE_nlin, MAC, frqMAC = perforMetrics(image=image, image_recov=image_recov, 
                                    image_under=image*mask, 
                                    fs=fs, u=u, dx=dx, room=room)


# %% [markdown]
# ## Visual results

# %%

images = [orig_image[:100,:], (orig_image*mask0)[:100,:], final_image[:100,:], image_lin[:100,:]]
titles = ['Original Image', 'Masked Image', 'Final reconst image', "Linear reconst"]

fig, ax = plt.subplots(1, len(images), figsize=(18, 6))
for i in range(len(images)):
    ax[i].imshow(images[i])
    ax[i].set_title(titles[i])
    ax[i].axis('off')
plt.tight_layout()
plt.show()


