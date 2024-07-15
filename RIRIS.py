import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os

# ------------------------------------------------------------------------------
# IMAGE FUNCTIONS
# ------------------------------------------------------------------------------

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


# ------------------------------------------------------------------------------
# PROYECTION FUNCTIONS
# ------------------------------------------------------------------------------

# Coefs of the image whose base are the functions in the dictionary
# ------------------------------------------------------------------------------
def ffst(image, Sk_func_dict):
    # image (M,N)
    # Sk_func_dict (M,N,d)
    # alpha (coefs in physical space of Sk functions)

    F2d_im = np.fft.fft2(image) 
    F2d_im = np.fft.fftshift(F2d_im) # order coefs from negative to positive axes=(0,1)

    # Proyection in Fourier space for all functions in dictionary
    Fproyection = Sk_func_dict * F2d_im[:, :, np.newaxis]
    Fproyection = np.fft.ifftshift(Fproyection, axes=(0,1))  # no ifftshift in 3rd axis
    
    # alpha contains the projection of the image in each decomposition/scale in physical space
    alpha = np.fft.ifft2(Fproyection, axes=(0, 1)).real  # by default, fft2 operates in the last 2 axes
    
    # Vectorize output 
    # order='F' like in Matlab, 
    # alpha = alpha.ravel(order='F')

    return alpha


# Reconstruction of the image usin coefs & dictionary
# ------------------------------------------------------------------------------

def iffst(alpha, Sk_func_dict):
    # alpha (M,N, d): coefs in phys space of the Sk functions
    # Sk_func_dict (M,N,d): Functions/kernels in dictionary
    # rec_image: reconstruction of the image in physical space

    # Change alpha to Fourier space
    F_alpha = np.fft.fft2(alpha, axes=(0,1))
    F_alpha = np.fft.fftshift(F_alpha, axes=(0,1))

    # Multiply coeficients by the functions in dictionary (in Fourier)
    sum_alpha = np.sum( F_alpha*Sk_func_dict , axis=2) # sum up decompositions
    sum_alpha = np.fft.ifftshift(sum_alpha) # axis=(0,1) since axis=2 has been reduced with the sum
    
    # Change from Fourier to physical space
    rec_image = np.fft.ifft2(sum_alpha).real # recover RIR image
 
    return rec_image


# ------------------------------------------------------------------------------
# OPTIMIZATION: Pareto & Iterative Soft Thresholding Algorithm (ISTA)
# ------------------------------------------------------------------------------

# Pareto
# ------------------------------------------------------------------------------


# soft thresholding 
def wthresh(x, thresh):
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0)


def computePareto(image, Sk):
        
    from scipy.interpolate import splrep, splev, splder
    
    # ----------------------  function starts here  ----------------------------------
    beta_set = np.logspace(-2.5, -1, 50)

    # preallocate:
    eta = np.zeros((len(beta_set), 1))  # sparsity norm
    rho = np.zeros((len(beta_set), 1))  # residual norm

    alpha0 = ffst(image, Sk)  # initialize solution vector

    # -----------------------------------------------------------------------
    # LAMBDA.T se encarga de eliminar los falsos
    # Y despues al revertir la operacion, esas posic se rellenan con zeros
    # LAMBDA.T @ (image - rec_image0) 
    # -----------------------------------------------------------------------
    for bb in range(len(beta_set)):
        rec_image0 = iffst(alpha0, Sk)
        diff_image0 = image - rec_image0 
        diff_alpha = ffst( diff_image0, Sk)

        signal = alpha0 + diff_alpha
        thresh = beta_set[bb] * np.linalg.norm(alpha0.ravel(), np.inf) # beta*||alpha0||_inf == beta*max(abs(x))

        alpha1 = wthresh( signal, thresh)

        eta[bb] = np.linalg.norm(alpha1.ravel(), 1) # sum(abs(x))

        rec_image1 = iffst(alpha1, Sk)
        diff_image1 = image - rec_image1
        
        rho[bb] = np.linalg.norm( diff_image1.ravel(), 2)
        print(f'Pareto iteration {bb+1}/{len(beta_set)}.')

    # Interpolación spline cúbica para eta y rho
    eta_sp = splrep(beta_set, np.log(eta).flatten())
    rho_sp = splrep(beta_set, np.log(rho).flatten())

    # Derivadas primera y segunda de eta
    eta_der1 = splder(eta_sp, 1)
    eta_prime1 = splev(beta_set, eta_der1)  # eta'
    eta_der2 = splder(eta_der1, 1)
    eta_prime2 = splev(beta_set, eta_der2)  # eta''

    # Derivadas primera y segunda de rho
    rho_der1 = splder(rho_sp, 1)
    rho_prime1 = splev(beta_set, rho_der1)  # rho'
    rho_der2 = splder(rho_der1, 1)
    rho_prime2 = splev(beta_set, rho_der2)  # rho''

    # Cálculo de la curva J
    Jcurve = (rho_prime2 * eta_prime1 - rho_prime1 * eta_prime2) / (rho_prime1**2 + eta_prime1**2)**1.5


    # Función para configurar los ejes de las gráficas
    def configure_plot():
        plt.axis('tight')
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.grid(True)

    # Graficar la función de curvatura
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 2)
    plt.semilogx(beta_set, Jcurve, 'ko', linewidth=2, markersize=8)
    configure_plot()
    plt.xlabel(r'$\beta$', fontsize=20, labelpad=10)
    plt.ylabel(r'$\mathcal{J}(\beta)$', fontsize=20, labelpad=10)
    plt.title('Curvature Function', fontsize=22)
    plt.draw()

    # Encontrar el beta óptimo
    idx_max_curv = np.argmax(Jcurve)
    beta_star = beta_set[idx_max_curv]

    # Graficar la curva L
    plt.subplot(1, 2, 1)
    plt.plot(np.log(rho), np.log(eta), 'ko', linewidth=2, markersize=8)
    configure_plot()
    plt.xlabel(r'$\rho(\beta) = \| \hat{\mathbf{y}} - \mathbf{\Phi}\alpha \|_2$', fontsize=20, labelpad=10)
    plt.ylabel(r'$\eta(\beta) = \| \alpha \|_1$', fontsize=20, labelpad=10)
    plt.title('L-curve', fontsize=22)
    plt.axvline(np.log(rho[idx_max_curv]), color='r', linestyle='--', linewidth=2)
    plt.axhline(np.log(eta[idx_max_curv]), color='r', linestyle='--', linewidth=2)
    plt.draw()

    # Mostrar las gráficas
    plt.tight_layout()
    plt.show()

    return beta_star, Jcurve, beta_set


# ISTA
# ------------------------------------------------------------------------------

def ista(image, Sk, beta, epsilon, max_iterations=15):
    
    alpha0 = ffst(image, Sk)  # initialize solution vector
    its = 0
    C = np.linspace(1, epsilon/beta, max_iterations+1);  # decreasing threshold function

    zeta0 = beta * np.linalg.norm(alpha0.ravel(), np.inf) # zeta0 = beta * || alpha0 ||_inf

    diff_image0 = image - iffst(alpha0, Sk)
    res_norm = np.linalg.norm( diff_image0 ) # || diff_image0 ||_2


    alpha = alpha0
    while (res_norm > epsilon) and (its < max_iterations):
        
        # Actualizar contador de iteraciones
        its += 1
        
        # Actualizar la solución
        rec_image0 = iffst(alpha, Sk)
        diff_image0 = image - rec_image0 
        diff_alpha = ffst( diff_image0, Sk) 
                
        signal = alpha + diff_alpha
        thresh = C[its] * zeta0 

        alpha = wthresh( signal, thresh)
        
        # Actualizar norma del residuo
        rec_image = image - iffst(alpha, Sk)
        res_norm = np.linalg.norm( rec_image )
        
        # Mostrar el progreso de la iteración
        print(f'Iteration no. {its}/{max_iterations},  res_norm: {res_norm}')


    print('Interpolating image from thresholded coefficients...')
    print('\n---------------- INTERPOLATION DONE! ----------------\n')

    return alpha


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def main():

    room = "Balder"
    Tstart = 0
    N = 100
    S = 3
    n_thetas = 7

    # ---------- LOAD Image ---------------------------

    folder = "./dependencies/measurementData"
    file = room+"RIR.mat"
    file_path = os.path.join(folder, file)
    print("Image loaded:")
    print(file_path)
    
    full_image = load_DB_ZEA(file_path)[0]
    Image = full_image[Tstart:Tstart+N, :N]
    mask, _ = rand_downsamp_RIR(Image.shape, 1, 0.3)

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(Image)
    ax[1].imshow(Image*mask)

    ax[0].axis('off')
    ax[1].axis('off')
    plt.show()

    # ---------- LOAD Dictionary -----------------------

    folder = "./"
    file = f"boostlets_N_{N}_S_{S}_nthetas_{n_thetas}.mat"
    file_path = os.path.join(folder, file)
    print("Loaded Dictionary:")
    print(file_path)

    Sk = sio.loadmat(file_path)['Psi']
    print("Dictionary shape:")
    print(Sk.shape)

    # ---------- Proyection into Dictionary Func ------

    alpha = ffst(Image, Sk)
    print("Coefs shape:")
    print(alpha.shape)

    ind=0
    fig, ax = plt.subplots(1,3, figsize=(12, 12) )
    ax[0].imshow(Image)
    ax[1].imshow(np.abs(Sk[:,:,ind]))
    ax[2].imshow(alpha[:,:,ind])
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    plt.show()

    # ---------- Recovered Image ---------------------

    rec_image = iffst(alpha=alpha, Sk_func_dict=Sk)
    fig, ax = plt.subplots(1,2, figsize=(12, 12) )
    ax[0].imshow(Image)
    ax[1].imshow(rec_image)
    ax[0].axis('off')
    ax[1].axis('off')
    plt.show()

    # --------- Optimization Pareto--------------------

    print("Optimization. Pareto")
    image_und = Image*mask
    beta_star, Jcurve, beta_set = computePareto(image_und, Sk)
    print("Optimal beta (beta_star):")
    print(beta_star)

    # --------- ISTA recovery -------------------------

    max_iterations = 15
    epsilon = 9.4e-6

    alpha = ista(image_und, Sk, beta_star, epsilon, max_iterations)





if __name__ == "__main__":
    main()
