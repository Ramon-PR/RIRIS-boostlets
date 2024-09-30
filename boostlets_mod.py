import numpy as np
import matplotlib.pyplot as plt
from mod_plotting_utilities import plot_array_images
from scipy.io import savemat
import os

# Meyer Functions
# ---------------

# Funcion auxiliar de Meyer para usar dentre de cos() y sin()
def meyeraux(x):
    v = np.zeros_like(x)
    
    # Apply conditions vectorized
    below_zero = (x < 0.0)
    above_one = (x > 1.0)
    between_zero_and_one = (~below_zero & ~above_one)

    v[below_zero] = 0.0
    v[above_one] = 1.0
    v[between_zero_and_one] = 35.0 * x[between_zero_and_one]**4 - 84.0 * x[between_zero_and_one]**5 + \
                              70.0 * x[between_zero_and_one]**6 - 20.0 * x[between_zero_and_one]**7
    
    return v

# General b(omega) function that grows with sin(m*omega) and decreases with cos(alpha*m*omega)
def meyerhelper(omega, a = 1.0, b = 2.0, c = 4.0):
    """
    omega: float.
    a: float a>0 where the function start to increases from 0 to 1 with a sin()
    b: float b>a where the function is 1 and joins a sin() segment with a cos() segment
    c: float c>b where the function reaches 0 after decreasing as a cos()
    Returns:
        a symmetric function around omega=0
        It is zero except in the support omega in [a,c]
        Increases from 0 to 1 as sin( omega/(b-a) ) for omega in [a,b]
        Decreases from 1 to 0 as cos( omega/(c-b) ) for omega in [b,c]
    """
    abs_omega = np.abs(omega)
    between_a_and_b = (a <= abs_omega) & (abs_omega <= b)
    between_b_and_c = (b < abs_omega) & (abs_omega <= c)
    result = np.zeros_like(abs_omega)
    result[between_a_and_b] = np.sin(np.pi/2 * meyeraux((abs_omega[between_a_and_b] - a)/(b-a)))
    result[between_b_and_c] = np.cos(np.pi/2 * meyeraux(((abs_omega[between_b_and_c] - b)/(c-b))))
    return result

def psi_1_fou(omega): 
    a1 = meyerhelper(2*omega)**2
    a2 = meyerhelper(omega)**2
    return np.sqrt(a1 + a2)    

def psi_2_fou(omega): 
    return np.sqrt(meyeraux(1-np.abs(omega)))

def psi_12_fou(omega1, omega2):
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = omega2 / omega1
        ratio[omega1 == 0] = 0  # Ajuste manual para evitar la división por cero
        result = psi_1_fou(omega1) * psi_2_fou(ratio)
        result[omega1 == 0] = 0  # Establecer 0 donde omega1 es 0
    return result

def meyerShearletSpect(x, y, a, s):
    # Transformaciones de las coordenadas
    y_trans = np.sqrt(a) * (s * x + y)
    x_trans = a * x

    # Evitar división por cero reemplazando 0 por 1 en x_trans
    x_safe = np.where(np.abs(x_trans) == 0, 1, x_trans)

    # Cálculo del espectro Meyer-Shearlet
    Psi = psi_1_fou(x_trans) * psi_2_fou(y_trans / x_safe)
    
    return Psi

# Scaling Function de Meyer 
def meyerScalingFun(theta):
    """
    Calcula la función de escalamiento de Meyer.
    - theta: un vector 1D o una matriz 2D.

    Retorna un array psi_2 con la función de escalamiento.

        
    ___(-thetaB2)___________(-thetaB1)________0________(thetaB1)___________(thetaB2)___
                 lateral_int           central interval         lateral_int     
    
    """
    
    thetaB1 = 1.0 / 6.0
    thetaB2 = 2.0 * thetaB1
    
    # Aplanar theta si es 2D (matriz) para realizar cálculos
    original_shape = theta.shape
    theta_flat = theta.flatten()

    central_interval = np.abs(theta_flat) < thetaB1
    lateral_interval = (np.abs(theta_flat) >= thetaB1) & (np.abs(theta_flat) < thetaB2)

    psi_2 = np.zeros_like(theta_flat)
    psi_2[central_interval] = 1 / np.sqrt(2 * np.pi)
    psi_2[lateral_interval] = 1 / np.sqrt(2 * np.pi) * \
                              np.cos(np.pi / 2 * meyeraux(np.abs(theta_flat[lateral_interval]) / thetaB1 - 1))
    
    # Volver a la forma original de theta
    psi_2 = psi_2.reshape(original_shape)
    
    return psi_2

# Wavelet Function de Meyer
def meyerWaveletFun(a):
    """
    Calcula la función wavelet de Meyer.
    
    - a: un vector 1D o una matriz 2D.
    
    Retorna un array psi_1 con la función wavelet de Meyer,
    manteniendo la misma forma que `a`.

    Intervals:
    
    ___(-aB3)_______________(-aB2)______________(-aB1)______0______(aB1)______________(aB2)_______________(aB3)___
             lateral_int_out      lateral_int_in                        lateral_int_in     lateral_int_out
         
    """
    
    aB1 = 1 / 3
    aB2 = 2 * aB1
    aB3 = 4 * aB1
    
    # Aplanar a si es 2D (matriz) para realizar cálculos
    original_shape = a.shape
    a_flat = a.flatten()

    lateral_int_in = (np.abs(a_flat) >= aB1) & (np.abs(a_flat) < aB2)
    lateral_int_out = (np.abs(a_flat) >= aB2) & (np.abs(a_flat) < aB3)

    psi_1 = np.zeros(a_flat.shape, dtype=complex)
    psi_1[lateral_int_in] = (1 / np.sqrt(2 * np.pi)) * np.exp(1j * a_flat[lateral_int_in] / 2) * \
                            np.sin(np.pi / 2 * meyeraux(np.abs(a_flat[lateral_int_in]) / aB1 - 1))
    psi_1[lateral_int_out] = (1 / np.sqrt(2 * np.pi)) * np.exp(1j * a_flat[lateral_int_out] / 2) * \
                             np.cos(np.pi / 2 * meyeraux(np.abs(a_flat[lateral_int_out]) / aB2 - 1))
    
    # Volver a la forma original de a
    psi_1 = psi_1.reshape(original_shape)
    
    return psi_1

# System that uses meyer function to fill the space where more scales fill the space closer to 0
class Meyer_system:
    def __init__(self, b_point, alpha) -> None:
        self.alpha = alpha
        self.b = b_point
        self.a = self.b * self.alpha 
        self.c = self.b/self.alpha
        self.gamma = alpha**2

    def b_om(self, omega):
        """ 
        support: [a,c]
        ----------------------a------------b---------------c----------------------
                  0               sin()          cos()               0
        """
        f = meyerhelper(omega, a=self.a, b=self.b, c=self.c)
        return f
    
    def psi_1(self, omega, scale):
        """ 
        Psi_1( omega ) = sqrt( b^2(omega) +  b^2(omega/alpha) )
        If we include the scale parameter S:
        Psi_1( omega / gamma^S) with S = scale
        (defined in Fourier space)
        gamma < 1  so (omega / gamma^S) = a makes the support to start at omega = gamma^S*a < a
        therefore, increasing the scale make the function to have a support closer to 0

        support: [gamma^S * a ,  ]
        ----------0-----------     sin()           cos()     ----------0------------
        ----------------------a--------------b---------------c----------------------
        ------a*alfa--------b*alfa-------c*alpha-------------------------------------
          0            sin          cos                     0 
        --------I-------------I--------------I----------------------------------------
                       sin    I      1       I     cos()     I          0
        """
        x = omega / self.gamma**scale 
        psi = np.sqrt( self.b_om(x)**2 + self.b_om(x /self.alpha)**2 )
        return psi
    
    def supp_psi_1(self, scale):
        """ 
        psi(omega/gamma^S) = b*[alpha^(2S+2) -- alpha^(2S+1) -- alpha^(2S) -- alpha^(2S-1)]
                                            left             mid          right 
        """
        return self.a*self.alpha**(2*scale+1), self.a*self.alpha**(2*scale-2)

    def left_psi_1(self, scale):
        """ 
        psi(omega/gamma^S) = b*[alpha^(2S+2) -- alpha^(2S+1) -- alpha^(2S) -- alpha^(2S-1)]
                                            left             mid          right 
        """
        return self.b*self.alpha**(2*scale+2), self.b*self.alpha**(2*scale+1)

    def mid_psi_1(self, scale):
        """ 
        psi(omega/gamma^S) = b*[alpha^(2S+2) -- alpha^(2S+1) -- alpha^(2S) -- alpha^(2S-1)]
                                            left             mid          right 
        """
        return self.b*self.alpha**(2*scale+1), self.b*self.alpha**(2*scale)

    def right_psi_1(self, scale):
        """ 
        psi(omega/gamma^S) = b*[alpha^(2S+2) -- alpha^(2S+1) -- alpha^(2S) -- alpha^(2S-1)]
                                            left             mid          right 
        """
        return self.b*self.alpha**(2*scale), self.b*self.alpha**(2*scale-1)

    def ones_psi_1(self, scale):
        return self.mid_psi_1(scale)
    
    def supp_system(self, scale):
        """ 
        support of union of psi for scales from 0 to S
        psi(omega/gamma^S) = a*[alpha^(2S+1), alpha^(-2)]
        """
        return self.a*self.alpha**(2*scale+1), self.a*self.alpha**(-2)

    def ones_system(self, scale):
        """ 
        support of union of psi for scales from 0 to S
        psi(omega/gamma^S) = a*[alpha^(2S), alpha^(-1)]
        """
        return self.a*self.alpha**(2*scale), self.b

    def max_scales(self, m_points):
        # b*alpha^(2*s+1)>1
        # omega_max = m/2 = b (so Psi**2(b)=1)
        # omega_max = m_points/2
        # b = omega_max
        s = np.floor( 0.5*(np.emath.logn(n=self.alpha, x=2/m_points) - 1) ).astype('int') 
        return s

    def get_omega(self, m_points):
        # omega = [-0.5, ..., 0, ...,0.5-domega]*b
        omega = np.fft.fftshift( np.fft.fftfreq(m_points) )*2*self.b 
        return omega

# Theta distribution for Boostlets:
# ----------------------------------------------------------------------
def ang_segmento(n_ondas):
    return np.pi/4/n_ondas

def ang_centros(n_ondas):
    d_alp = ang_segmento(n_ondas)
    indxs = np.arange(1,n_ondas+1)
    phis = np.pi/4 - (2*indxs - 1)*d_alp
    return phis

def theta_dist(n_ondas):
    phis = ang_centros(n_ondas)
    m = np.tan(phis)
    h_thetas = np.log(1+m)/2 - np.log(1-m)/2
    return h_thetas
# ----------------------------------------------------------------------

# Boost operation:
# ----------------------------------------------------------------------
def boost_points(om, k, a, theta):
    k_b  = a*(  k*np.cosh(theta) - om*np.sinh(theta)) 
    om_b = a*( om*np.cosh(theta) -  k*np.sinh(theta)) 
    
    return k_b, om_b
# ----------------------------------------------------------------------

# Diffeo for horizontal cone
# ----------------------------------------------------------------------
def diffeo_hor_cone(k, om):
    """
    Ad no no depende de theta. No lo defino para k==om incluido (k=0, om=0)
    Th es:
       inf en k==om 
      -inf para k==-om
       nan para k=om=0
    I don not use np.nan or np.inf since python needs to explicitly set how to add np.nan 
    and other numbers
    """
    # Crear una máscara para el caso cuando |k| > |om|
    hor_cone = np.abs(k)>np.abs(om)
    
    Ad = np.sqrt(np.abs(k**2 - om**2))*hor_cone

    # Lo que está en el cono horiz lo dejo tal cual el resto lo pongo en 0 (arctanh no quiere 1, o -1)
    ratio = np.divide(om, k, where=(k != 0))*hor_cone + np.zeros_like(k)*(~hor_cone)

    Th = np.arctanh(ratio)  # Calcula Th evitando la división por cero
    Th[~hor_cone] = 0  # Asignar 0 fuera del cono

    return Ad, Th
# ----------------------------------------------------------------------

# Boostlet in horizontal cone
# ----------------------------------------------------------------------
def get_boostlet_h(om, kx, a_i, theta_j, wavelet_fun=meyerWaveletFun, scaling_fun=meyerScalingFun):
    KX_b, OM_b = boost_points(om, kx, a_i, theta_j)
    Ad_h, Th_h = diffeo_hor_cone(k=KX_b, om=OM_b)
    Phi = wavelet_fun(Ad_h)*scaling_fun(Th_h)
    Phi /= np.max(np.abs(Phi)) 
    return Phi, KX_b, OM_b
# ----------------------------------------------------------------------


# Boostlet system con Meyer System  
# ----------------------------------------------------------------------
class Boostlet_syst:
    def __init__(self, dx, dt, cs,
                 M=100, N=100, 
                 n_v_scales=1, n_h_scales=1,
                 n_v_thetas=3, n_h_thetas=3,  
                 base_v=0.5, base_h=0.5, 
                 wavelet_fun=meyerWaveletFun, scaling_fun=meyerScalingFun 
                 ) -> None:
        """ 
        M filas
        N columnas
        dx: space sampling
        dt: time sampling
        cs: sound velocity
        n_v_scales: number of vertical scales
        n_h_scales: number of horizontal scales
        base_v: alfa for Meyer_system for vertical scales
        base_h: alfa for Meyer_system for horizontal scales
        """

        self.M = M
        self.N = N

        self.dx = dx
        self.dt = dt
        self.cs = cs

        self.n_v_thetas = n_v_thetas
        self.n_h_thetas = n_h_thetas  

        self.base_v = base_v
        self.base_h = base_h

        # Define axis om[1/s], kx[1/m] and k[1/s]. 
        # Use of om & k to have an acoustic cone not dependent on cs. 
        self.om = np.fft.fftshift( np.fft.fftfreq(n=self.M, d=self.dt) )
        self.kx = np.fft.fftshift( np.fft.fftfreq(n=self.N, d=self.dx) )
        self.k = self.cs * self.kx
    
        # Meyer system for horizontal and vertical cones
        self.ms_v = Meyer_system(b_point=np.max(np.abs(self.om)), alpha=base_v)
        self.ms_h = Meyer_system(b_point=np.max(np.abs(self.k)), alpha=base_h)

        # number of scales:
        self.max_sc_v = min(n_v_scales, self.ms_v.max_scales(M))
        self.max_sc_h = min(n_h_scales, self.ms_h.max_scales(N))

        # theta distribution to equidivide the acoustic cone
        self.v_thetas = theta_dist(n_v_thetas)
        self.h_thetas = theta_dist(n_h_thetas)

        self.n_boostlets = (self.max_sc_v+1)*n_v_thetas + (self.max_sc_h+1)*n_h_thetas + 1

        self.wavelet_fun = wavelet_fun
        self.scaling_fun = scaling_fun

    def get_boostlet_dict(self):
        K, OM = np.meshgrid(self.k, self.om)

        # Primero, la función de escala (boost_type=1)
        Psi = np.zeros((self.M, self.N, self.n_boostlets), dtype=complex)

        count=1
        # Cono horizontal
        for isc in range(self.max_sc_h + 1):
            for theta_j in self.h_thetas:
                K_b, OM_b = boost_points(om=OM, k=K, a=1, theta=theta_j)
                Ad, Th = diffeo_hor_cone(k=K_b, om=OM_b)
                Phi = self.ms_h.psi_1(Ad, scale=isc)*self.scaling_fun(Th)
                Phi /= np.max(np.abs(Phi)) 
                Psi[:,:,count] = Phi
                count += 1

        # Cono vertical
        for isc in range(self.max_sc_v + 1):
            for theta_j in self.v_thetas:
                K_b, OM_b = boost_points(om=K, k=OM, a=1, theta=theta_j)
                Ad, Th = diffeo_hor_cone(k=K_b, om=OM_b)
                Phi = self.ms_v.psi_1(Ad, scale=isc)*self.scaling_fun(Th)
                Phi /= np.max(np.abs(Phi)) 
                Psi[:,:,count] = Phi
                count += 1

        # Check sum of squares for all scales
        Phi = np.sum(Psi**2, axis=2)
        # Add the scaling function to the dictionary to complete R2
        mask = ~(np.abs(Phi) > 0.0)
        Psi[:,:,0] = np.ones_like(Phi)*mask

        # Check sum of squares for all scales
        Phi = np.sum(Psi**2, axis=2)
        # Divide each scale by the sqrt of the sum, to ensure Parseval
        Psi /= np.sqrt(Phi)[:, :, np.newaxis]  

        return Psi 

    def plot_dict_boostlets(self):
        Psi = self.get_boostlet_dict()
        plot_array_images(Psi, num_cols=5)
        
        
    def gen_boostlet_h(self, theta, isc):
        """
        En vez de usar a, uso MeyerSystem 
        """
        K, OM = np.meshgrid(self.k, self.om)
        K_b, OM_b = boost_points(om=OM, k=K, a=1, theta=theta)
        Ad, Th = diffeo_hor_cone(k=K_b, om=OM_b)
        # Phi = self.wavelet_fun(Ad)*self.scaling_fun(Th)
        Phi = self.ms_h.psi_1(Ad, scale=isc)*self.scaling_fun(Th)
        Phi /= np.max(np.abs(Phi)) 
        return Phi #, K_b, OM_b
    
    def plot_boostlet(self, itheta, isc):
        Phi = self.gen_boostlet_h(theta=self.h_thetas[itheta], isc=isc)
        fig, ax = plt.subplots(1,1)
        ax.contourf(self.k, self.om, Phi)



    def plot_psi_1(self, scale):
        psi_v = self.ms_v.psi_1(self.om, scale)
        psi_h = self.ms_h.psi_1(self.k, scale)

        fig, axs = plt.subplots(1,2)
        axs[0].plot(self.om, psi_v)
        axs[1].plot(self.k, psi_h)

        xlabs = [r'$\omega$', r'$k$']
        ylabs = [r'$\psi_1$', r'$\psi_1$']
        titles = [rf'scale={scale}', rf'scale={scale}']

        for (ax, xl, yl, ttl) in zip(axs, xlabs, ylabs, titles):
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_title(ttl)
        plt.tight_layout()
        plt.show()

    def print_max_scales(self):
        print(rf"vertical scales: {self.max_sc_v}")
        print(rf"horizontal scales: {self.max_sc_h}")

    def get_axis(self):
        return self.om, self.k
    
    def save_dict(self, folder=".saved_dicts/"):
        # Get boostlet dictionary
        Psi = self.get_boostlet_dict()

        # Create a dict with booslets and label fields
        mdic = {"Psi": Psi, 
        "label": f"boostlets with: \n"
        f"{self.max_sc_v} vert. scales, "
        f"{self.max_sc_h} hor. scales, \n" 
        f"{len(self.v_thetas)} vert. angles, " 
        f"{len(self.h_thetas)} hor. angles, \n\n"
        "factor for supports: \n"
        f"vert. base: {self.base_v}, \n"
        f"hor. base: {self.base_h}"
        }

        # file = rf"dict_BS_m_{self.M}_n_{self.N}_hsc_{self.max_sc_h}_vsc_{self.max_sc_v}_thV_{len(self.v_thetas)}_thH_{len(self.h_thetas)}.mat"
        file = rf"BS_m_{self.M}_n_{self.N}_vsc_{self.max_sc_v}_hsc_{self.max_sc_h}" \
               rf"_bases_{self.base_v}_{self.base_h}" \
               rf"_thV_{self.n_v_thetas}_thH_{self.n_h_thetas}.mat"
        file_path = os.path.join(folder, file)

        print("Saving dictionary in \n"
               f"{file_path} \n" )
        print(mdic['label'])

        os.makedirs(folder, exist_ok=True)
        # Save 
        savemat(file_path, mdic)


# ------------------------------------------------------------------------
# Indexes of the dict to eliminate. Based on convolutions with the mask
# ------------------------------------------------------------------------

def indexs_max_conv_mask_dict(mask, Sk):
    F2d_im = np.fft.fft2(mask) 
    F2d_im = np.fft.fftshift(F2d_im) # order coefs from negative to positive in axes=(0,1)
    Specs = Sk * F2d_im[:, :, np.newaxis].conj()

    B = np.max( abs(Specs), axis=(0,1)) 
    # Encuentra los índices de los elementos que no son cero
    nonzero_indices = np.where(B != 0)[0]
    # Ordena los índices en base a los valores correspondientes en B en orden descendente
    sorted_indices = nonzero_indices[np.argsort(-B[nonzero_indices])]

    return sorted_indices


def indexs_max_conv_mask_maskedDict(mask, Sk):
    Fproyection = np.fft.ifftshift( Sk*Sk , axes=(0,1))  # no ifftshift in 3rd axis
    phys_Psi2 = np.fft.ifft2(Fproyection, axes=(0, 1)).real  # by default, fft2 operates in the last 2 axes
    phys_Psi2 = np.fft.fftshift(phys_Psi2, axes=(0,1))

    MSk = mask[...,np.newaxis] * phys_Psi2 # masked physical dict kernels
    Specs = np.fft.fft2(mask)[...,np.newaxis] * np.fft.fft2(MSk, axes=(0, 1)).conj() # conv -> mult in Fou space

    B = np.max( abs(Specs), axis=(0,1)) 
    # Encuentra los índices de los elementos que no son cero
    nonzero_indices = np.where(B != 0)[0]
    # Ordena los índices en base a los valores correspondientes en B en orden descendente
    sorted_indices = nonzero_indices[np.argsort(-B[nonzero_indices])]

    return sorted_indices

# -------------------------------------------------
# Difeomorfismo
# -------------------------------------------------

def computeDiffeo(OM, KX, vert_hor_cone):
    """
    Compute the diffeomorphism for the given wavenumber-frequency grids.

    Parameters:
    OM (ndarray): Omega values (frequency components).
    KX (ndarray): KX values (wavenumber components).
    vert_hor_cone (int): Determines the field type:
        0 for vertical cone (far field, |omega| > |k| ) 
        1 for horizontal cone (near field, |omega| < |k|)

    Returns:
    Ad (ndarray): Diffeomorphism amplitude values.
    Th (ndarray): Diffeomorphism angle values.
    """
    
    # Calculate Z
    Z = OM**2 - KX**2
    
    if vert_hor_cone == 0:  # Vertical cone / far field
        Ad = np.sqrt(Z.astype(complex))
        with np.errstate(divide='ignore', invalid='ignore'):
            div = np.divide(KX, OM).astype(complex)
            Th = np.arctanh(div)
    else:  # Horizontal cone / near field
        Ad = np.sqrt(-Z.astype(complex))
        with np.errstate(divide='ignore', invalid='ignore'):
            div = np.divide(OM, KX).astype(complex)
            Th = np.arctanh(div)
    
    return Ad, Th


# -------------------------------------------------
# Generate Boostlets
# -------------------------------------------------

def genBoostlet(N, a_i, theta_j, far_or_near, boost_type):
    """
    Generate Boostlets

    Parameters:
    N (int): Number of samples per dimension
    a_i (float): Dilation parameter, (2^0, 2^1, ... 2^(S-1) )
    theta_j (float): angle between -pi/2 and pi/2
    far_or_near (int): Determines the field type (0 for far field, 1 for near field)
    boost_type (int): Type of boost (1 for scaling function, 2 for boostlet functions)

    Returns:
    tuple: phi (ndarray), KX (ndarray), OM (ndarray)
    """

    # Create Cartesian wavenumber-frequency space
    om = np.linspace(-1, 1, N)
    kx = np.linspace(-1, 1, N)
    KX, OM = np.meshgrid(kx, om)

    # Define boost/dilation matrix
    M_a_theta = np.array([
        [a_i * np.cosh(theta_j), -a_i * np.sinh(theta_j)],
        [-a_i * np.sinh(theta_j), a_i * np.cosh(theta_j)]
    ])

    # Apply boost and dilation to each point
    # --------------------------------------
    # for i in range(N):
    #     for j in range(N):
    #         boosted_points = M_a_theta @ np.array([KX[i, j], OM[i, j]])
    #         KX_atheta[i, j] = boosted_points[0]
    #         OM_atheta[i, j] = boosted_points[1]
            
    boosted_points = np.einsum('ij,xyj->xyi', M_a_theta, np.dstack((KX, OM)))
    KX_atheta = boosted_points[:, :, 0]
    OM_atheta = boosted_points[:, :, 1]


    # Apply diffeomorphism to boosted/dilated points
    Ad, Th = computeDiffeo(OM_atheta, KX_atheta, far_or_near)

    # Generate the appropriate function based on boost_type
    if boost_type == 1:
        phi = meyerScalingFun(Ad)
    elif boost_type == 2:
        PHI_1 = meyerWaveletFun(Ad)
        PHI_2 = meyerScalingFun(Th)
        phi = PHI_1 * PHI_2
    else:
        raise ValueError("boost_type must be 1 or 2")

    # Normalize phi
    phi /= np.max(np.abs(phi))

    return phi, KX, OM

# Grid para los parametros a y theta (dilatacion y boosts)


def get_boostlets_dict(N, a_grid, theta_grid):
    """ 
    a_grid = 2 ** np.arange(S)
    theta_grid = np.linspace(-np.pi/2, np.pi/2, n_thetas)
    """
    
    S = len(a_grid)

    far_or_near = [0, 1]
    n_boostlets = len(far_or_near)*len(a_grid)*len(theta_grid) + 1

    # Primero, la función de escala (boost_type=1)
    phi0 = genBoostlet(N, a_i = S-1, theta_j=0.0, far_or_near=0, boost_type=1)[0]

    Psi = np.zeros((N, N, n_boostlets), dtype=complex)
    Psi[:,:,0] = phi0

    # Luego cada uno de los boostlets en distintas escalas
    boost_type = 2
    count=1
    for f_farNear in far_or_near:
        for a_i in a_grid:
            for theta_j in theta_grid:
                phi = genBoostlet(N, a_i, theta_j, f_farNear, boost_type)[0]
                Psi[:,:,count] = phi
                count += 1

    return Psi
