import numpy as np

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

    # def get_Psi_system(self, m_points):
    #     # omega = [-0.5, ..., 0, ...,0.5-domega]*b
    #     self.omega = self.get_omega(m_points)
    #     self.S = self.max_scales(m_points)
    #     self.
    #     return omega







    
#  EXAMPLE
# a, alpha = 1, 0.5
# b, c = a/alpha, a/alpha**2     
# mw = Meyer_system(a=a, alpha=alpha)
# omega = np.linspace(-4.5, 4.5, 1001)*1
# f = mw.b_om(omega)
# psi_s0 = mw.psi_1(omega, 0)
# psi_s1 = mw.psi_1(omega, 1)
# psi_s2 = mw.psi_1(omega, 2)
# x_ticks = [-c, -b, -a, 0, a, b, c] 
# x_labels = ["-c", "-b", "-a", "0", "a", "b", "c"]
# fig, ax = plt.subplots(1,3, figsize=(25,6) )
# ax[0].plot(omega, f, '-', label=r'$b(\omega)$', linewidth=2.5)
# ax[0].plot(omega, psi_s0, '--',label=rf'$\Psi_1(\omega, S={0}) $', linewidth=1.5)
# ax[0].scatter(omega[psi_s0 == 1.0], psi_s0[psi_s0 == 1.0], color='k', label=r'$\Psi_1(\omega)=1$')
# ax[0].scatter(omega[f == 1.0], f[f == 1.0], color='red', label=r'$b(\omega)=1$')
# ax[0].set_xticks(x_ticks, labels=x_labels)
# ax[0].grid(visible=True)
# ax[0].legend()
# ax[1].plot(omega, psi_s0, '--',label=rf'$\Psi_1(\omega, S={0}) $', linewidth=1.5)
# ax[1].plot(omega, psi_s1, '--',label=rf'$\Psi_1(\omega, S={1}) $', linewidth=1.5)
# ax[1].plot(omega, psi_s2, '--',label=rf'$\Psi_1(\omega, S={2}) $', linewidth=1.5)
# ax[1].grid(visible=True)
# ax[1].legend()
# ax[2].plot(omega, psi_s0**2 + psi_s1**2 + psi_s2**2, '--',label=r'$\sum_{S=0}^{S=2} \Psi_1^2(\omega, S) $', linewidth=1.5)
# ax[2].grid(visible=True)
# ax[2].legend()
# plt.show()

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
        Ad = np.sqrt(Z.astype(complex)).flatten()
        with np.errstate(divide='ignore', invalid='ignore'):
            div = np.divide(KX, OM).astype(complex)
            Th = np.arctanh(div).flatten()
    else:  # Horizontal cone / near field
        Ad = np.sqrt(-Z.astype(complex)).flatten()
        with np.errstate(divide='ignore', invalid='ignore'):
            div = np.divide(OM, KX).astype(complex)
            Th = np.arctanh(div).flatten()
    
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
        phi = meyerScalingFun(N, Ad)
    elif boost_type == 2:
        PHI_1 = meyerWaveletFun(N, Ad)
        PHI_2 = meyerScalingFun(N, Th)
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
