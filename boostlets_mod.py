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

# Scaling Function de Meyer 
def MeyerScalingFun(N, theta):
    # N: number of samples per dimension
    # theta: a vector obtained by flattening an NxN array
    # returns an array psi_2[N,N] with the Scaling Function
    # 
    # ___(-thetaB2)___________(-thetaB1)________0________(thetaB1)___________(thetaB2)___
    #              lateral_int           central interval         lateral_int     
    #            
    thetaB1 = 1.0 / 6.0
    thetaB2 = 2.0 * thetaB1

    central_interval = np.abs(theta) < thetaB1
    lateral_interval = (np.abs(theta) >= thetaB1) & (np.abs(theta) < thetaB2)

    psi_2 = np.zeros_like(theta)
    psi_2[central_interval] = 1 / np.sqrt(2 * np.pi)
    psi_2[lateral_interval] = 1 / np.sqrt(2 * np.pi) * \
                              np.cos(np.pi / 2 * meyeraux(np.abs(theta[lateral_interval]) / thetaB1 - 1))
    
    psi_2 = psi_2.reshape(N, N)
    
    return psi_2

# Wavelet Function de Meyer
def MeyerWaveletFun(N, a):
    # 
    # Intervals:
    # 
    # ___(-aB3)_______________(-aB2)______________(-aB1)______0______(aB1)______________(aB2)_______________(aB3)___
    #          lateral_int_out      lateral_int_in                        lateral_int_in     lateral_int_out
    #      

    aB1 = 1 / 3
    aB2 = 2 * aB1
    aB3 = 4 * aB1

    lateral_int_in = (np.abs(a) >= aB1) & (np.abs(a) < aB2)
    lateral_int_out = (np.abs(a) >= aB2) & (np.abs(a) < aB3)

    psi_1 = np.zeros(a.shape, dtype=complex)
    psi_1[lateral_int_in] = (1 / np.sqrt(2 * np.pi)) * np.exp(1j * a[lateral_int_in] / 2) * \
                  np.sin(np.pi / 2 * meyeraux(np.abs(a[lateral_int_in]) / aB1 - 1))
    psi_1[lateral_int_out] = (1 / np.sqrt(2 * np.pi)) * np.exp(1j * a[lateral_int_out] / 2) * \
                  np.cos(np.pi / 2 * meyeraux(np.abs(a[lateral_int_out]) / aB2 - 1))
    
    psi_1 = psi_1.reshape(N, N)
    return psi_1


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
        phi = MeyerScalingFun(N, Ad)
    elif boost_type == 2:
        PHI_1 = MeyerWaveletFun(N, Ad)
        PHI_2 = MeyerScalingFun(N, Th)
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
