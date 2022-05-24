import numpy as np
from math import pi, sqrt
from astropy.constants import sigma_sb, L_sun, R_sun

# Global parameters:
sigma = sigma_sb.value


# Calc. L from  and R
def lum_eqn(T, R):
    """
    T = stellar effective temp df/array [K]
    R = stellar radius df/array [solar radius]
    
    returns luminosity array [solar lum]
    """

    T = T.to_numpy()
    R = R.to_numpy() * R_sun.value
    L = 4*pi*sigma * np.multiply(R**2, T**4) # in W
    L_solar = L / (L_sun.value) # in solar lum
    
    return L_solar


# Calc. dL from dT and dR
def lum_unc_eqn(T, dT, R, dR, positive_unc=True):
    """
    T = stellar effective temp df/array [K]    
    dT = uncs in T df/array [K]
    R = stellar radius df/array [solar radius]
    dR = uncs in R df/array [solar rad]
    positive_unc = True if calculating +ve dL, = False if calculating -ve dL
    
    returns luminosity uncs df/array [solar lum]
    """

    T = T.to_numpy()
    dT = dT.to_numpy()
    R = R.to_numpy() * R_sun.value
    dR = dR.to_numpy() * R_sun.value
    
    dL = 4*pi*sigma * np.sqrt( ( 2*np.multiply(R, np.multiply(T**4, dR)) )**2 + ( 4*np.multiply(R**2, np.multiply(T**3, dT)) )**2 ) 
    dL_solar = dL / (L_sun.value)
    
    if (positive_unc==False):
        dL_solar = -dL_solar
    
    return dL_solar


# Calculate the stellar effective temperature from stellar radius and luminosity
def teff_eqn(L, R):
    """
    R = stellar radius df/array [solar radius]
    L = stellar luminosity df/array [solar lum]
    
    returns stellar effective temperature array
    """
    
    # Convert R to m:
    R = R.to_numpy() * R_sun.value
    
    # Convert L to W:
    L = L.to_numpy() * L_sun.value
    
    T = ( np.multiply(L, R**(-2)) / (4*pi*sigma) )**(0.25) 
    
    return T


# Calculate uncertainties in T from dL and dR
def teff_unc_eqn(L, dL, R, dR, positive_unc=True):
    """
    R = stellar radii df/array [solar radius]
    dR = uncs in R df/array [solar rad]
    L = stellar luminosies df/array [solar lum]
    dL = uncs in L df/array [solar rad]
    positive_unc = True if calculating +ve dT, = False if calculating -ve dT

    returns array of uncertainties in stellar effective temperature
    """
    
    # Convert R to m:
    R = R.to_numpy() * R_sun.value
    dR = dR.to_numpy() * R_sun.value
    
    # Convert L to W:
    L = L * L_sun.value
    dL = dL * L_sun.value
    
    dT = 0.5*(4*pi*sigma)**(-0.25) * np.sqrt ( ( 0.5*np.divide(np.multiply(L**(-0.75), dL), np.sqrt(R) ) )**2 + ( np.multiply(np.multiply(L**(-0.25), R**(-1.5)), dR) )**2 )
    
    if (positive_unc==False):
        dT = -dT
        
    return dT