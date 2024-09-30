# %% [markdown]
# # Linear predictive border padding

# %%
import numpy as np
from scipy.signal import lfilter, lfiltic
from spectrum import arburg
import os
import scipy.io as sio
import matplotlib.pyplot as plt
from boostlets_mod import get_boostlets_dict



def arburg_extrap(signal, extr_length, order=2):
    """ 
    Given a "signal", it concatenates the extrapolation at the end of the signal of length "extr_length"
    _______signal________ + ___extrap___

    The extrapolation is calculated using a 1-D digital filter (filter in MATLAB or scipy.signal.lfilter in Python)
    The parameters for the filter comes from an autoregressive Burg's method (arburg) with order "order"    

    Example:

    signal = np.arange(10)
    print(f" signal: {signal}")
    extrap_forward = arburg_extrap(signal, 3, 2)
    print(f" extrap_forward: {extrap_forward}")
    extrap_backward = arburg_extrap(extrap_forward[::-1], 3, 2)[::-1]
    print(f" extrap_backward: {extrap_backward}")
    """
    a0, _, _ = arburg(signal, order)
    a = np.concatenate(([1], a0)) # add a 1 at the begining of a0 to look like on Matlab

    last_points = signal[-order:] #last points, number of points == order
    last_points = last_points[::-1] # give the "last points" in reverse order, the oldest the first (as lfilter wants the x and y)

    Z = lfiltic(b=1, a=a, y=last_points, x=None) # Initial conditions for the lfilter
    yzi, _ = lfilter(b=np.array([1]), a=a, x=np.zeros(extr_length), zi=Z)

    extrap = np.concatenate((signal, yzi.real))

    return extrap


def lpbp1D(signalIn, dL, dR, order=2):
    """
    LPBP1D 1D Linear Predictive Border Padding
    
    This function extrapolates linear microphone array data by means of
    designing and applying filters with AR prediction coefficients.
    
    Parameters:
    signalIn : array-like (size N)
        1D input signal
    dL, dR,: int
        Number of samples added before and after the signal
    order : int
        Order of the filter ("number of Fourier peaks")
        
    Returns:
    signalOut : array-like
        Extrapolated signal. 
        1D array resulting of a concatenation of [dL, N, dR] samples
        dN being extrapolated parts.
    """
    
    signal_extr_forw = arburg_extrap(signalIn, extr_length=dR, order=order) # extrapolate forward in time
    signalOut = arburg_extrap(signal_extr_forw[::-1], extr_length=dL, order=order)[::-1] # flip the signal and extrapolate to extrapolate backwards in time
    
    return signalOut


# %% [markdown]
# # Operaciones con imagenes

# %%

def next_power_of_2(n):
    """Return the next power of 2 greater than or equal to n."""
    return 1 if n == 0 else 2**int(np.ceil(np.log2(n)))


class ImageOps:
    """ 
    Lo uso para crear un objeto imagen con un método:
        expand_image() 
            que pueda expandirse en la imagen extrapolada
        recover_image()
            que devuelva la imagen original
        mask_image()
            que devuelva la imagen enmascarada     
    """
    def __init__(self, image_shape, mask, mode="pad", extrap_shape=None) -> None:
        # self.image = image
        self.orig_shape = image_shape
        self.mask = mask
        self.state = "original"
        self.mode = mode

        m0, n0 = image_shape
        # Si extrap_shape es None, usar next_power_of_2 para calcular m y n
        if extrap_shape is None:
            m = next_power_of_2(m0)
            n = next_power_of_2(n0)
        else:
            m, n = extrap_shape

        # Asignar extrap_shape basado en los valores calculados o proporcionados
        self.extrap_shape = (m, n)

        self.up_pad = max((m - m0) // 2, 0)
        self.down_pad = max(m - self.up_pad - m0, 0)
        self.left_pad = max((n - n0) // 2, 0)
        self.right_pad = max(n - self.left_pad - n0, 0)

    def expand_image(self, image):
        """
        Extrapolate an image to the next power of 2 dimensions or specified padding.
        Parameters:
        image (np.array): 2D array to be extrapolated.
        flag_extrap (int): Currently unused. Reserved for future use.

        Returns:
        np.array: Padded 2D array.
        """

        if image.shape == self.extrap_shape:
            return image
        
        elif self.mode == "extrapolate":

            T = image.shape[0]
            M = self.extrap_shape[1]            

            image_ext = np.zeros((T, M))
            for tt in range(T):  # execute linear predictive border padding
                image_ext[tt, :] = lpbp1D(image[tt, :], dL=self.left_pad, dR=self.right_pad, order=2)

            # Pad vertically with 0s
            pad_width = ((self.up_pad, self.down_pad), (0, 0))
            padded_image = np.pad(image_ext, pad_width, mode='constant', constant_values=0)

            self.state = "expanded"

            return padded_image
        
        else:   
            # Padding the 2D array with zeros
            pad_width = ((self.up_pad, self.down_pad), (self.left_pad, self.right_pad))
            padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)

            self.state = "expanded"

            return padded_image
    
        
    def recover_image(self, image):
        """ Take the original image from the extrapolated image """
        if image.shape == self.extrap_shape:
            m, n = self.extrap_shape
            rec_image = image[self.up_pad: m-self.down_pad, self.left_pad:n-self.right_pad]
            self.state = "original"
            return rec_image
        else:
            return image
    

    def get_mask(self, image):
        if image.shape == self.orig_shape:
            return self.mask
        
        # Pad the mask to match the extrapolated image dimensions
        if self.mask.ndim == 1:
            pad_width = (self.left_pad, self.right_pad)
        elif self.mask.ndim == 2:
            pad_width = ((0, 0), (self.left_pad, self.right_pad))
        else:
            raise ValueError("Unsupported mask dimension: {}".format(self.mask.ndim))

        if self.mode == "extrapolate":
            pad_values = 1
        else:
            pad_values = 0
        
        padded_mask = np.pad(self.mask, pad_width, mode='constant', constant_values=pad_values)
        return padded_mask

        
    def mask_image(self, image):
        return self.get_mask(image) * image



# %% [markdown]
# # Transformaciones

# %%

def ffst(image, Sk_func_dict):
    """
    image (M,N)
    Sk_func_dict (M,N,d) (Fourier Space)
    alpha (coefs in physical space of Sk functions)
    """

    F2d_im = np.fft.fft2(image) 
    F2d_im = np.fft.fftshift(F2d_im) # order coefs from negative to positive in axes=(0,1)

    # Proyection in Fourier space for all functions in dictionary
    Fproyection = Sk_func_dict * F2d_im[:, :, np.newaxis]
    Fproyection = np.fft.ifftshift(Fproyection, axes=(0,1))  # no ifftshift in 3rd axis
    
    # alpha contains the projection of the image in each decomposition/scale in physical space
    alpha = np.fft.ifft2(Fproyection, axes=(0, 1)).real  # by default, fft2 operates in the last 2 axes

    return alpha


def iffst(alpha, Sk_func_dict):
    """
    alpha (M,N, d): coefs in phys space of the Sk functions
    Sk_func_dict (M,N,d): Functions/kernels in dictionary
    rec_image: reconstruction of the image in physical space
    """

    # Change alpha to Fourier space
    F_alpha = np.fft.fft2(alpha, axes=(0,1))
    F_alpha = np.fft.fftshift(F_alpha, axes=(0,1))

    # Multiply coeficients by the functions in dictionary (in Fourier)
    sum_alpha = np.sum( F_alpha*Sk_func_dict , axis=2) # sum up decompositions
    sum_alpha = np.fft.ifftshift(sum_alpha) # axis=(0,1) since axis=2 has been reduced with the sum
    
    # Change from Fourier to physical space
    rec_image = np.fft.ifft2(sum_alpha).real # recover RIR image
 
    return rec_image


def wthresh(x, thresh):
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0)


def common_operations(image, mask, Sk, alpha, threshold):
    # Recupera la imagen de las coeficientes
    # rec_image = mask * iffst(alpha, Sk)
    rec_image = iffst(alpha, Sk)
    # Calcula la diferencia entre la imagen original y la recuperada
    diff_image = mask * (image - rec_image)
    # Proyecta la diferencia en el espacio de Fourier
    diff_alpha = ffst(diff_image, Sk)

    # Actualiza la señal y aplica el umbral
    signal = alpha + diff_alpha
    # thresh = beta * np.max(np.abs(alpha))

    # Aplica el umbral suave a la señal
    alpha_new = wthresh(signal, threshold)

    return alpha_new 


# %%

def computePareto(image, mask, Sk, beta_set, f_verbose=False, f_plot=False):
    from scipy.interpolate import splrep, splev, splder
    import matplotlib.pyplot as plt

    # preallocate:
    eta = np.zeros_like(beta_set)  # sparsity norm
    rho = np.zeros_like(beta_set)  # residual norm

    image = mask*image
    alpha0 = ffst(image, Sk)  # initialize solution vector
    norm_inf_alpha0 = np.linalg.norm(alpha0.ravel(), np.inf)

    for bb, beta in enumerate(beta_set):
        threshold = beta * norm_inf_alpha0
        # _, eta[bb], rho[bb] = common_operations(image, mask, Sk, alpha0, threshold)
        alpha_new = common_operations(image, mask, Sk, alpha0, threshold)

        eta[bb] = np.linalg.norm(alpha_new.ravel(), 1)       # ||alpha_new||_1
        rec_image_new = iffst(alpha_new, Sk)
        diff_image_new = mask * (image - rec_image_new)
        rho[bb] = np.linalg.norm(diff_image_new.ravel(), 2)  # || diff_image ||_2

        if f_verbose:
            print(f'Pareto iteration {bb+1}/{len(beta_set)}.')

    # Interpolación spline cúbica para eta y rho
    eta_sp = splrep(beta_set, np.log(eta))
    rho_sp = splrep(beta_set, np.log(rho))

    # Derivadas primera y segunda de eta y rho
    eta_prime1 = splev(beta_set, splder(eta_sp, 1)) # eta'
    eta_prime2 = splev(beta_set, splder(eta_sp, 2)) # eta''
    rho_prime1 = splev(beta_set, splder(rho_sp, 1)) # rho'
    rho_prime2 = splev(beta_set, splder(rho_sp, 2)) # rho''

    # Cálculo de la curva J
    Jcurve = (rho_prime2 * eta_prime1 - rho_prime1 * eta_prime2) / (rho_prime1**2 + eta_prime1**2)**1.5

    # Encontrar el beta óptimo
    idx_max_curv = np.argmax(Jcurve)
    beta_star = beta_set[idx_max_curv]

    # Graficar la función de curvatura y la curva L
    if f_plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].plot(np.log(rho), np.log(eta), 'ko', linewidth=2, markersize=8)
        axes[0].axvline(np.log(rho[idx_max_curv]), color='r', linestyle='--', linewidth=2)
        axes[0].axhline(np.log(eta[idx_max_curv]), color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel(r'$\rho(\beta) = \| \hat{\mathbf{y}} - \mathbf{\Phi}\alpha \|_2$', fontsize=20, labelpad=10)
        axes[0].set_ylabel(r'$\eta(\beta) = \| \alpha \|_1$', fontsize=20, labelpad=10)
        axes[0].set_title('L-curve', fontsize=22)
        axes[0].tick_params(axis='both', which='major', labelsize=20)
        axes[0].grid(True)

        axes[1].semilogx(beta_set, Jcurve, 'ko', linewidth=2, markersize=8)
        axes[1].set_xlabel(r'$\beta$', fontsize=20, labelpad=10)
        axes[1].set_ylabel(r'$\mathcal{J}(\beta)$', fontsize=20, labelpad=10)
        axes[1].set_title('Curvature Function', fontsize=22)
        axes[1].tick_params(axis='both', which='major', labelsize=20)
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    return beta_star, Jcurve


# %%
def ista(image, mask, Sk, beta, epsilon, max_iterations=15, f_verbose=False, f_plot=False ):

    # Initialize solution vector 
    image = mask*image
    alpha = ffst(image, Sk) # here, alpha = alpha0

    # Decreasing thresholds: C*zeta0 == C[i] * (beta * ||alpha_0||_inf)
    zeta0 = beta * np.linalg.norm(alpha.ravel(), np.inf) # beta * ||alpha_0||_inf
    C = np.linspace(1, epsilon / beta, max_iterations + 1)

    # Initial residual norm
    rec_image = iffst(alpha, Sk)
    diff_image = mask * (image - rec_image)
    res_norm = np.linalg.norm(diff_image.ravel()) 

    its = 0
    while res_norm > epsilon and its < max_iterations:
        its += 1
        # alpha, _, res_norm = common_operations(image, mask, Sk, alpha, threshold = C[its]*zeta0)
        alpha = common_operations(image, mask, Sk, alpha, threshold = C[its]*zeta0)
        rec_image_new = iffst(alpha, Sk)
        diff_image_new = mask * (image - rec_image_new)
        res_norm = np.linalg.norm(diff_image_new.ravel(), 2)  # || diff_image ||_2

        if f_verbose:
            print(f'Iteration no. {its}/{max_iterations}, res_norm: {res_norm}')

    if f_verbose:
        print('Interpolating image from thresholded coefficients...')
        print('\n---------------- INTERPOLATION DONE! ----------------\n')

    return alpha

# %%
def load_DB_ZEA(path):
    import pymatreader as pymat
    #Load data
    RIR=[]
    RIR = pymat.read_mat(path)["out"]["image"]
    fs = pymat.read_mat(path)["out"]["fs"] # Hz
    
    T, M = RIR.shape
    
    x0 = 0.0
    dx = 0.03 # m
    x = np.arange(0,M).reshape((1,M))*dx + x0

    t0 = 0.0
    dt = 1.0/fs # s
    t = np.arange(0,T).reshape((T,1))*dt + t0
    return RIR, x, t

def rand_downsamp_RIR(shape, ratio_t=1, ratio_x=0.5):
    import random
    # choose a ratio of samples in time/space from RIR
    # random choice
    T, M = shape
    tsamples = int(T*ratio_t)
    xMics  = int(M*ratio_x)

    id_T = np.sort(random.sample(range(0,T), tsamples)) # rows to take
    id_X = np.sort(random.sample(range(0,M), xMics)) # cols to take

    mask_T = np.zeros([T, 1], dtype=bool)
    mask_X = np.zeros([1, M], dtype=bool)

    mask_T[id_T,0] = True
    mask_X[0,id_X] = True

    return mask_X, mask_T

def jitter_downsamp_RIR(shape, ratio_t=1, ratio_x=0.5):
    # shape: shape of masks. mask = mask_X*mask_T
    # ratio_t: ratio of rows to choose
    # ratio_x: ratio of columns to choose
    
    T, M = shape
    tsamples = int(T*ratio_t)
    xMics  = int(M*ratio_x)

    deltaT = T/tsamples
    deltaX = M/xMics

    id_T = np.rint(np.arange(0,T,deltaT)).astype(int) # rows to take
    id_X = np.rint(np.arange(0,M,deltaX)).astype(int) # cols to take

    # # Add randomness of 1 position to the right, except last element
    epsX = np.random.randint(0, 2, size=id_X.shape)
    epsX[-1] = 0
    epsT = np.random.randint(0, 2, size=id_T.shape)
    epsT[-1] = 0

    # Mask to check that id_X[i]+1 is not the same as id_X[i+1]
    mask_epsX = ~(id_X[1:]==id_X[0:-1]+1)
    mask_epsT = ~(id_T[1:]==id_T[0:-1]+1)
    mask_epsX = np.append(mask_epsX, False)
    mask_epsT = np.append(mask_epsT, False)

    # Valid epsilons
    epsX = epsX*mask_epsX
    epsT = epsT*mask_epsT

    if ratio_t != 1:
        id_T += epsT
    if ratio_x != 1:
        id_X += epsX
    
    mask_T = np.zeros([T, 1], dtype=bool)
    mask_X = np.zeros([1, M], dtype=bool)

    mask_T[id_T,0] = True
    mask_X[0,id_X] = True

    return mask_X, mask_T


def load_sk(folder, file, build_dict=None):
    # Construct the file path
    file_path = os.path.join(folder, file)

    # If the file exists, load it
    if os.path.exists(file_path):
        print("Loading dictionary")
        print(file_path)
        Sk = sio.loadmat(file_path)['Psi']
    # If build_dict is not None and dictionary type is specified
    elif build_dict is not None and build_dict.get('Sk_type') == "boostlet":
        N = build_dict["N"]
        S = build_dict["S"]
        n_thetas = build_dict["n_thetas"]

        print(f"Generating boostlet dictionary. N={N}, S={S}, n_thetas={n_thetas}")
        a_grid = 2 ** np.arange(S)
        theta_grid = np.linspace(-np.pi/2, np.pi/2, n_thetas)
        Sk = get_boostlets_dict(N, a_grid, theta_grid)
    else:
        raise ValueError("File not found and build_dict is not properly defined to generate the dictionary.")

    return Sk

# %% [markdown]
# ## Linear interpolation

# %%
def linear_interpolation_fft(image, dx=3e-2, fs=11250, cs=340):
    """
    Perform linear interpolation (hourglass filtering) of the FFT spectrum of an image.
    
    Parameters:
    image (numpy.ndarray): 2D array representing the input image.
    dx (float): Spatial resolution.
    fs (float): Sampling frequency.
    cs (float, optional): Speed of wave propagation. Default is 340 (speed of sound in air).
    
    Returns:
    numpy.ndarray: The processed image after linear interpolation.
    """

    # Get image dimensions
    nt, nx = image.shape

    # Calculate physical dimensions
    Lx = nx * dx
    Lt = nt * 1 / fs

    # Compute the FFT of the image and shift the zero-frequency component to the center
    yFFT = np.fft.fftshift(np.fft.fft2(image))

    # Calculate wave numbers and frequencies
    ks = 2 * np.pi / dx
    dk = 2 * np.pi / Lx
    df = fs / nt

    # Generate frequency and wave vectors
    freqVec = np.linspace(-fs / 2, fs / 2 - df, nt)
    waveVec = np.linspace(-ks / 2, ks / 2 - dk, nx)
    KK, FF = np.meshgrid(waveVec, freqVec)

    # Distinguish propagating from evanescent waves
    term = (2 * np.pi * FF / cs) ** 2 - KK ** 2
    mask = np.zeros_like(term)
    mask[term >= 0] = 1
    PAF = mask

    # Compute interpolated responses
    image_linear = np.real(np.fft.ifft2(np.fft.ifftshift(yFFT * PAF)))

    return image_linear


# %% [markdown]
# ## Performance Metrics

# %%
def calculate_NMSE(y_true, y_est):
    """
    Calculate the Normalized Mean Squared Error (NMSE) in dB between y_true and y_est signals.
    
    Parameters:
    y_true (np.ndarray): The reference signal with shape (T, M).
    y_est (np.ndarray): The estimated signal with shape (T, M).
    
    Returns:
    float: The NMSE in dB.
    """
    _, M = y_true.shape
    NMSE = 0
    valid_columns = 0
    
    for mm in range(M):
        norm_ref = np.linalg.norm(y_true[:, mm])
        if norm_ref != 0:
            NMSE += np.linalg.norm(y_true[:, mm] - y_est[:, mm]) ** 2 / norm_ref ** 2
            valid_columns += 1
    
    if valid_columns > 0:
        with np.errstate(divide='ignore', invalid='ignore'):
            NMSE = 10 * np.log10(NMSE / valid_columns)
    else:
        NMSE = float('inf')  # If all columns have zero norm, return inf to indicate invalid NMSE calculation
    
    return NMSE


def calculate_MAC(y_tru, y_est, fs):
    """
    Calculate the Modal Assurance Criterion (MAC) between the true signal and 
    an estimated signal.
    
    Parameters:
    y_tru (np.ndarray): The reference signal with shape (T, M0).
    y_est (np.ndarray): The estimated signal with shape (T, M0).
    fs (float): Temporal sampling frequency (Hz).
    
    Returns:
    tuple: Tuple containing:
        - frqMAC (np.ndarray): The frequency axis vector for MAC (Hz).
        - MAC (np.ndarray): MAC values for the estimated signal.
    """
    T, M0 = y_tru.shape
    
    # Perform FFT and normalize
    FFT_Ptru = np.fft.fft(y_tru, axis=0) / T
    FFT_Prec = np.fft.fft(y_est, axis=0) / T
    
    # Frequency axis vector
    frqMAC = np.linspace(0, fs / 2, T // 2, endpoint=False)
    
    # Initialize MAC array
    MAC = np.zeros(len(frqMAC))
    
    # Calculate MAC
    for ff in range(len(frqMAC)):
        dot_product = np.dot(FFT_Ptru[ff, :], FFT_Prec[ff, :].conj())
        norm_tru = np.dot(FFT_Ptru[ff, :], FFT_Ptru[ff, :].conj())
        norm_est = np.dot(FFT_Prec[ff, :], FFT_Prec[ff, :].conj())
        
        if norm_tru != 0 and norm_est != 0:
            MAC[ff] = (np.abs(dot_product) ** 2) / (np.abs(norm_tru) * np.abs(norm_est))
    
    return frqMAC, MAC

# NMSE_lin = calculate_NMSE(image, image_linear)
# NMSE_nlin = calculate_NMSE(image, image_recov)
# frqMAC, MAC_lin = calculate_MAC(image, image_linear, fs)
# _, MAC_nlin = calculate_MAC(image, image_recov, fs)
    

# %%

def perforMetrics(image, image_recov, image_under, fs, u, dx, room, f_plot=False):
    # PERFORMETRICS Assessment function with performance metrics and figures
    
    T, M = image.shape

    image_linear = linear_interpolation_fft(image_under, dx, fs, cs=340)

    # define space-time grid
    t_idxs = np.arange(min( T, 128 ))
    XX, TT = np.meshgrid(dx * np.arange(M), 1000 * (t_idxs) / fs)

    NMSE_lin = calculate_NMSE(image, image_linear)
    NMSE_nlin = calculate_NMSE(image, image_recov)
    frqMAC, MAC_lin = calculate_MAC(image, image_linear, fs)
    _, MAC_nlin = calculate_MAC(image, image_recov, fs)

    # Arrange outputs
    NMSE_nlin = [NMSE_lin, NMSE_nlin]
    MAC = [MAC_lin, MAC_nlin]


    if f_plot:
        # plot image results
        aux = image[t_idxs, :]
        zscale = [np.min(np.real(aux)), np.max(np.real(aux))]
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))

        img0 = axs[0].imshow(image_under[t_idxs, :], aspect='auto', cmap='gray', extent=[XX.min(), XX.max(), TT.max(), TT.min()])
        axs[0].set_title('Under-sampled', fontsize=18)
        axs[0].set_xlabel('$x$ (m)', fontsize=18)
        axs[0].set_ylabel('$t$ (ms)', fontsize=18)
        img0.set_clim(zscale)

        img1 = axs[1].imshow(image_linear[t_idxs, :], aspect='auto', cmap='gray', extent=[XX.min(), XX.max(), TT.max(), TT.min()])
        axs[1].set_title('Linear int.', fontsize=18)
        axs[1].set_xlabel('$x$ (m)', fontsize=18)
        axs[1].set_ylabel('$t$ (ms)', fontsize=18)
        img1.set_clim(zscale)

        img2 = axs[2].imshow(image_recov[t_idxs, :], aspect='auto', cmap='gray', extent=[XX.min(), XX.max(), TT.max(), TT.min()])
        axs[2].set_title('Diccionary', fontsize=18)
        axs[2].set_xlabel('$x$ (m)', fontsize=18)
        axs[2].set_ylabel('$t$ (ms)', fontsize=18)
        img2.set_clim(zscale)

        img3 = axs[3].imshow(image[t_idxs, :], aspect='auto', cmap='gray', extent=[XX.min(), XX.max(), TT.max(), TT.min()])
        axs[3].set_title('Reference', fontsize=18)
        axs[3].set_xlabel('$x$ (m)', fontsize=18)
        axs[3].set_ylabel('$t$ (ms)', fontsize=18)
        img3.set_clim(zscale)

        plt.suptitle(room)
        plt.show()

        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.plot(frqMAC, MAC_lin, color=[0.38, 0.38, 0.38], linewidth=2)
        plt.xlabel('$f$ (Hz)', fontsize=15)
        plt.ylabel('MAC', fontsize=15)
        plt.ylim([0, 1])
        plt.xlim([0, fs / 2])
        plt.axvline(x=fs / (2 * u), color='r', linestyle='--', linewidth=2)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.title('Linear')

        plt.subplot(1, 2, 2)
        plt.plot(frqMAC, MAC_nlin, color=[0.38, 0.38, 0.38], linewidth=2)
        plt.xlabel('$f$ (Hz)', fontsize=15)
        plt.ylabel('MAC', fontsize=15)
        plt.ylim([0, 1])
        plt.xlim([0, fs / 2])
        plt.axvline(x=fs / (2 * u), color='r', linestyle='--', linewidth=2)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.title('Multiscale Dict')
        plt.show()

    return NMSE_nlin, MAC, frqMAC


# %% [markdown]
# # Main

# %%

def main():

    import matplotlib.pyplot as plt
    import os
    from boostlets_mod import get_boostlets_dict

    room = "Balder"
    # Sk_type = "shearlet"
    Sk_type = "boostlet"
    S = 2 # number of decomposition scales
    ratio_undersampling = 0.3

    extrap_mode = "extrapolate" # or pad

    dx=3e-2
    fs=11250
    cs=340
    u = round(1/ratio_undersampling)


    if Sk_type == "shearlet":
        print("Shearlet decomposition")
        folder = "./dependencies/basisFunctions/"
        file = f"{room}_tau{S}.mat"

        # Tamaño diccionario // Tamaño imagen interpolada
        M, N  = 128, 128
        # Imagen tamaño:
        # M0, N0 = 100, 100

        # Seleccionar subimage 
        Tstart = 0
        Tend = None

        build_dict = dict(Sk_type=Sk_type)

    elif Sk_type == "boostlet":
        print("Boostlet decomposition")

        n_thetas = 7

        # Tamaño diccionario // Tamaño imagen interpolada
        M, N  = 128, 128
        # Imagen tamaño:
        M0, N0 = 100, 100

        # Seleccionar subimage 
        Tstart = 0
        Tend = Tstart+M0

        folder = "./dependencies/basisFunctions/boostlets"
        file = f"boostlets_N_{N}_S_{S}_nthetas_{n_thetas}.mat"
        build_dict = dict(Sk_type=Sk_type, n_thetas=n_thetas, N=N, S=S)



    # %% [markdown]
    # ## Cargar diccionario e imagen

    # ---------- LOAD Dictionary -----------------------

    Sk = load_sk(folder, file, build_dict)
    print(Sk.shape)
    extr_size = Sk.shape[:2]

    # ---------- LOAD Image ---------------------------
    folder = "./dependencies/measurementData"
    file = room+"RIR.mat"
    file_path = os.path.join(folder, file)
    print("Image loaded:")
    print(file_path)

    # Load full image and select a subimage to apply decomposition
    full_image = load_DB_ZEA(file_path)[0]
    orig_image = full_image[Tstart:Tend, :N0]

    mask0, _ = rand_downsamp_RIR(orig_image.shape, ratio_t=1, ratio_x=ratio_undersampling)

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(orig_image[Tstart:Tstart+100, :])
    ax[1].imshow((orig_image*mask0)[Tstart:Tstart+100, :])

    ax[0].axis('off')
    ax[1].axis('off')
    plt.show()




    # %% [markdown]
    # ### Eliminar elementos del diccionario
    # 



    # %% [markdown]
    # ## Extrapolation // Padding

    # ----------------------------------------------------
    # Extrapolation
    # ----------------------------------------------------

    extr_size = Sk.shape[:2]
    imOps = ImageOps(orig_image.shape, mask=mask0, extrap_shape=extr_size, mode=extrap_mode) 

    image = imOps.expand_image(orig_image)
    mask = imOps.get_mask(image)

    print(image.shape)
    print(mask.shape)

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
    # ## Proyection

    # ----------------------------------------------------
    # Proyection
    # ----------------------------------------------------
    image = mask*image
    alphas = ffst(image, Sk)
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(image)
    ax[1].imshow(mask*image)

    ax[0].axis('off')
    ax[1].axis('off')
    plt.show()
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



    # %%
    # recover inpainted image from sparse coefficients (Eq. 19)
    image_recov = iffst(alpha, Sk)
    final_image = imOps.recover_image(image_recov)


    # %% [markdown]
    # ## Linear interpolation
    # 
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

    images = [orig_image[:100,:], (orig_image*mask0)[:100,:], final_image[:100,:], image_lin[:100,:]]
    titles = ['Original Image', 'Masked Image', 'Final reconst image', "Linear reconst"]

    fig, ax = plt.subplots(1, len(images), figsize=(18, 6))
    for i in range(len(images)):
        ax[i].imshow(images[i])
        ax[i].set_title(titles[i])
        ax[i].axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
