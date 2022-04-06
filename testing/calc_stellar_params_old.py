import numpy as np
from math import pi, sqrt
from astropy.constants import sigma_sb, L_sun, R_sun

# Global parameters:
sigma = sigma_sb.value

# Calc. L from R and T
def calc_luminosity(T,  dT1, dT2, R, dR1, dR2):
    """
    T = stellar effective temperature [K]
    dT1 = positive error
    dT2 = negative error
    R = stellar radius [solar radius]
    dR1 = positive error
    dR2 = negative error
    returns log(stellar luminosity) [log(solar)], positive error, percentage error, negative error, percentage error
    """
    
    dR1 = dR1 * R_sun.value
    dR2 = dR2 * R_sun.value 

    # Calculate L if T and R =/ NaN:
    if np.isfinite(T) and np.isfinite(R):
        
        # Convert R, dR1 and dR2 to meters:
        R = R * R_sun.value
        #dR1 = dR1 * R_sun.value # These didn't work when they were here
        #dR2 = dR2 * R_sun.value 
        
        L = 4 * pi * sigma * R**2 * T**4 # in W
        L_solar = L / L_sun.value # in Solar Luminosities
        log_L = np.log10(L_solar) # L listed in units log(Solar) in NASA EA
        
        # Calculate positive and negative errors in L if T and R have uncs.
        # Faster than calcing dlog_L for every value just to get NaN?
        if np.isfinite(dT1) and np.isfinite(dR1):
            dL1 = 4*pi*sigma * sqrt( (2*R*T**4*dR1)**2 + (R**2*4*T**3*dT1)**2 ) # in W
            dL1_solar = dL1 / (L_sun.value) # in Solar Lum
            # TODO: whya re these uncertainties coming out so big??
            dlog_L1 = dL1_solar / ( L_solar*np.log(10) ) # in log(Solar). np.log() is ln()
            percent_err1 = abs(dlog_L1 / log_L) * 100 #log_L can be +/-ve
            
        # If both dT1 and dR1 are not available, we can't calc dL1_solar:
        else:
            #dL1 = np.nan
            dL1_solar = np.nan # TODO: won't need this saved
            dlog_L1 = np.nan
            percent_err1 = np.nan
            
        if np.isfinite(dT2) and np.isfinite(dR2):
            dL2 = (-1) * 4*pi*sigma_sb.value * sqrt( (2*R*T**4*dR2)**2 + (R**2*4*T**3*dT2)**2 ) # in W
            dL2_solar = dL2 / (L_sun.value) # in Solar Lum
            dlog_L2 = dL2_solar / ( L_solar*np.log(10) ) # in log(Solar)
            percent_err2 = abs(dlog_L2 / log_L) * 100 #log_L can be +/-ve
        
        # If both dT1 and dR1 are not available, we can't calc dL1_solar:
        else:
            #dL2 = np.nan
            dL2_solar = np.nan # TODO: won't need this saved
            dlog_L2 = np.nan
            percent_err2 = np.nan
        
    # If either T or R are Nan, then L and dL are NaN:
    else:
        #L = np.nan
        L_solar = np.nan # TODO: won't need this saved
        log_L = np.nan
        dL1_solar = np.nan
        dL2_solar = np.nan
        dlog_L1 = np.nan
        dlog_L2 = np.nan
        percent_err1 = np.nan
        percent_err2 = np.nan
    
    #return log_L, dlog_L1, percent_err1, dlog_L2, percent_err2
    return L_solar, dL1_solar, dL2_solar
#, log_L, dlog_L1, dlog_L2

    
    
# Calculate the stellar effective temperature from stellar radius and luminosity:
def calc_temp(R, dR1, dR2, log_L, dlog_L1, dlog_L2):
    """
    R = stellar radius [solar radius]
    dR1 = positive error
    dR2 = negative error
    log_L = stellar luminosity
    dlog_L1 = positive error
    dlog_L2 = negative error
    
    returns stellar effective temperature, positive error, percent err, negative error, percent err
    """
    
    sigma = sigma_sb.value
    
    # Calculate T if R and L /= NaN
    if np.isfinite(R) and np.isfinite(log_L):
        
        # Convert R to meters:
        R = R * R_sun.value # [m]
    
        # Convert log_L, dlog_L1 and dlog_L2 to W:
        L = 10**log_L * L_sun.value # [W]
        
        #T = (2*R)**(-2) * ( L / (pi*sigma) )**(0.25)
        T = (4*pi*sigma)**(-0.25) * L**(0.25) * R**(-0.5)
        
        # Calculate negative and positive errors in T if errors in R and dlog_L are available:
        if np.isfinite(dR1) and np.isfinite(dlog_L1):
            # Convert dR1 to W:
            dR1 = dR1 * R_sun.value
            # Convert dlog_L1 to W
            dL1 = 10**log_L * np.log(10) * dlog_L1 * L_sun.value
            
            # Calc dT1:
            dT1 = 0.5*(4*pi*sigma)**(-0.25) * sqrt( ( 0.5*L**(-0.75)*dL1 / sqrt(R) )**2 + ( L**(-0.25)*R**(-1.5)*dR1 )**2 )
            percent_err1 = dT1 / T * 100
            
        # If dR1 and dlog_L1 aren't both avaiable, we can't calc dT1
        else:
            dT1 = np.nan
            percent_err1 = np.nan
        
        if np.isfinite(dR2) and np.isfinite(dlog_L2):
            # Convert dR1 to W:
            dR2 = dR2 * R_sun.value
            # Convert dlog_L1 to W
            dL2 = 10**log_L * np.log(10) * dlog_L2 * L_sun.value
            
            # Calc dT2:
            dT2 = (-1) * 0.5*(4*pi*sigma)**(-0.25) * sqrt( ( 0.5*L**(-0.75)*dL2 / sqrt(R) )**2 + ( L**(-0.25)*R**(-1.5)*dR2 )**2 ) # Mmultiply by (-1), otherwise dT2 would be +ve
            percent_err2 = abs(dT2 / T) * 100
            
            
        # If dR2 and dlog_L2 aren't both avaiable, we can't calc dT2:
        else:
            dT2 = np.nan
            percent_err2 = np.nan
        
    # If either R or L are NaN, then we can't calc T or its errors:
    else:
        T = np.nan
        dT1 = np.nan
        dT2 = np.nan
        percent_err1 = np.nan
        percent_err2 = np.nan
    
    return T, dT1, percent_err1, dT2, percent_err2