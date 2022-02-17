
   
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Program to compute the Statistical-likelihood Exo-Planetary 
Habitability Index (SEPHI) as published in Rodrıguez-Mozos & Moya, 
MNRAS 471, 4628–4636 (2017).
SEPHI can be estimated with only seven physical characteristics: 
planetary mass, planetary radius, planetary orbital period, stellar mass, 
stellar radius, stellar effective temperature and planetary system age.
"""

import numpy as np
from astropy import units as u
from astropy import constants as const
import math as m
#from utils.stars import f_sma
#from utils.utils import add_fields


#Function to determine the likelihood that a planet has telluric 
#composition. It makes use of the model grid of planet composition in 
#Zeng & Sasselov, PASP, 125, 925 (2013).
def likelihood_telluric(planet_mass, planet_radius, verbose):
    
    if(verbose):
        print("Composition")
    
    # Determining radius of a 100% MgSiO3 planet of the same mass
    # Using result of 3rd-order polynomial fit to Table 1 in Zeng & Sasselov 2013
    log10_MgSiO3_radius = np.poly1d( np.array([-0.0066252 , -0.02274424,  0.30285182,  0.02052977]))
    MgSiO3_radius = 10.**(log10_MgSiO3_radius(planet_mass.to(u.Mearth).value))
   
    # Determining radius of a 50% MgSiO3, 50% H2O planet of the same mass
    # Using result of 3rd-order polynomial fit to Table 1 in Zeng & Sasselov 2013
    log10_MgSiO3_H2O_radius = np.poly1d( np.array([-0.00637393, -0.01837899,  0.28980072,  0.10391018]))
    MgSiO3_H2O_radius = 10.**(log10_MgSiO3_H2O_radius(planet_mass.to(u.Mearth).value))

    sigma = (MgSiO3_H2O_radius - MgSiO3_radius) / 3.0
    
    #--------------------------
    # Calculating likelihood

    c1 = np.where(planet_radius.to(u.Rearth).value <= MgSiO3_radius) #condition 1
    c2 = np.where(np.logical_and( planet_radius.to(u.Rearth).value > MgSiO3_radius, \
                                  planet_radius.to(u.Rearth).value <= MgSiO3_H2O_radius) ) #condition 2
    
    likelihood = np.zeros(planet_mass.size)  #initialising with zero likelihood   
    if(c1[0].size>0):
        likelihood[c1] = 1.0
    if(c2[0].size>0):
        likelihood[c2] = np.exp( -0.5 * ( (planet_radius.to(u.Rearth).value[c2] - MgSiO3_radius[c2])/sigma[c2] )**2 )
    
    if(verbose):
        print("Sim 1: (Telluric planet) = ", likelihood, "\n")

    return likelihood


#Function to determine likelihood that planet has an atmosphere.
def likelihood_atmosphere(planet_mass, planet_radius, verbose):
    
    if(verbose):
        print("Normalised physical characteristics")
    
    #calculating the relative escape velocity of planet wrt Earth
    v_e = (planet_mass.to(u.kg) * const.R_earth / (const.M_earth * planet_radius.to(u.m)) )**0.5
    if(verbose):
        print ("escape velocity relative to Earth",v_e)
        
    # calculating likelihood    
    c1 = np.where(v_e.value < 1.0)  #condition 1
    c2 = np.where(v_e.value >= 1.0) #condition 2

    likelihood = np.zeros(planet_mass.size)  #initialising with zero likelihood  
    if(c1[0].size>0):
        likelihood[c1] = np.exp(-0.5*(3.*(v_e.value[c1]-1.))**2)
    if(c2[0].size>0):
        likelihood[c2] = np.exp(-0.5*(3.*(v_e.value[c2]-1.)/7.66)**2)
    
    if(verbose):
        print("Sim 2: (Atmosphere) = ", likelihood, "\n")
    return likelihood


#Function to determine likelihood that planet has an liquid water on surface.
#Uses models in Kopparapu+13,ApJ,765,131 and Kopparapu+14,ApJL,787,29
#Note: only works for stars with effective stellar temperatures of 2600K < T_eff <7200K,
#corresponding to F,G,K,M stars.
# SNL 3/8: At the moment the function deals 'correctly' with remnants, in that if 
# L_star = 0 and T_eff_star = nan, the function return a likelihood value = '0.'
# However, the code produces warnings and errors, so in the future it would be 
# better to deal with this more elegantly. 
def likelihood_surface_liquid_water(T_eff_star, L_star, planet_mass, planet_semi_major_axis, verbose):

    if(verbose):
        print("Habitability")
    
    possible_water = (L_star > 0.) & ~np.isinf(planet_semi_major_axis) # where the host star has non-zero luminosity and the planet is not free-floating

    #Making sure the stellar effective temperature is in the correct range
    if( np.any(T_eff_star[possible_water].value < 2600.) or np.any(T_eff_star[possible_water].value > 7200.) ):
        # print("Warning: stellar effective temperature outside correct range of 2600K < T_eff <7200K. Truncating.")
        T_eff_star.value[np.where( np.logical_and(T_eff_star.value<2700., possible_water) )] = 2700.0 
        T_eff_star.value[np.where( np.logical_and(T_eff_star.value>7200., possible_water) )] = 7200.0 
        # return -1.0
    
    #Coefficient updates from Table 1 of Kopparapu+14,ApJL,787,29
    #These are similar to Kopparapu+13,ApJ,765,131 but the runaway greenhouse depends on planet mass.
    #To get coefficients: download HZs.f90 from Kop+14 electronic file associated with the paper, 
    #then compile and run that to generate HZs.dat and HZ_coefficients.dat. 
    #Coeffs copied here to avoid speed penalty in opening/reading from file
    ##numbers: 1=recent venus,2=runaway greenhouse,3=max. greenhouse,4=early mars,5=run. ghouse (5Mearth), 6=run. ghouse (0.1Mearth)
    
    S_eff_sol=np.repeat(np.array([[1.77600E+00,1.10700E+00,3.56000E-01,3.20000E-01,1.18800E+00,9.90000E-01]]), T_eff_star.size, axis=0)
    a=np.array([[2.13600E-04,1.33200E-04,6.17100E-05,5.54700E-05,1.43300E-04,1.20900E-04]])
    b=np.array([[2.53300E-08,1.58000E-08,1.69800E-09,1.52600E-09,1.70700E-08,1.40400E-08]])
    c=np.array([[-1.33200E-11,-8.30800E-12,-3.19800E-12,-2.87400E-12,-8.96800E-12,-7.41800E-12]])
    d=np.array([[-3.09700E-15,-1.93100E-15,-5.57500E-16,-5.01100E-16,-2.08400E-15,-1.71300E-15]])
    
    T_star_K = T_eff_star - 5780*u.K
    T_star = np.reshape(T_star_K.value, (1,T_star_K.size))
  
    # Calculating the distance from the star for each of the boundaries. First the mass-independent ones
    S_eff = S_eff_sol.T + np.dot(a.T,T_star) + np.dot(b.T,T_star**2) + np.dot(c.T,T_star**3) + np.dot(d.T,T_star**4)
    D = (L_star.to(u.W)/const.L_sun/S_eff)**0.5 * u.au

    #-------------------
    # Calculating distance from star at which planet undergoes runaway greenhouse. In Kopparapu+14 this is planet-mass-dependent
    # SNL: note the mass boundaries where the switch between different regimes occurs are not defined in SEPHI paper.
    #      I've made the boundary roughly mid-range between the discrete masses but this may vary from the online calculator.
 
    c1 = np.where( (planet_mass.value<0.5) & possible_water)
    c2 = np.where( (planet_mass.value>2.) & possible_water)
    c3 = np.where( np.logical_and(planet_mass.value>=0.5, planet_mass.value <=2.) & possible_water)
    
    D_RG = np.zeros(planet_mass.size)*u.au
    if(c1[0].size>0):
        D_RG[c1] = D[5,c1]; #print (c1[0].size, "Planet mass <0.5Msun")
    if(c2[0].size>0):
        D_RG[c2] = D[4,c2]; #print (c2[0].size, "Planet mass >2Msun")
    if(c3[0].size>0):
        D_RG[c3] = D[1,c3]; #print (c3[0].size, "Planet mass 0.5 - 2 Msun")
    
    # Checking distances make sense, i.e. D1<D_RG<D3<D4
    # Note, these distances get screwed up if the temperature is outside the 2700K < T < 7200K range
    # if not ( np.all(D[0,:]<D_RG) and np.all(D_RG<D[2,:]) and np.all(D[2,:]<D[3,:])):
    #     print("Distance calculations incorrect, because temperature outside correct range of 2600K < T_eff <7200K")
        #return -1.
 
    if(verbose):
        print("Green Zone Inner Radius = ", D_RG)
    if(verbose):
        print("Green Zone Outer Radius = ", D[2,:])
    if(verbose):
        print("Incident Effective Flux = ", S_eff)
    
    #-------------------
    # Determining the likelihood
        
    likelihood = np.zeros(planet_mass.size)  #initialising with zero likelihood 
        
    c4 = np.where( (planet_semi_major_axis < D_RG) & possible_water)
    c5 = np.where( np.logical_and(planet_semi_major_axis >= D_RG, planet_semi_major_axis <= D[2,:]) & possible_water)
    c6 = np.where( (planet_semi_major_axis > D[2,:]) & possible_water)
 
    if(c4[0].size>0): 
        sigma_31=(D_RG-D[0,:])/3.0   
        likelihood[c4]=np.exp(-0.5*((planet_semi_major_axis[c4] - D_RG[c4])/sigma_31[c4])**2)
        if(verbose):
            print("Zone = Hot")
        
    if(c5[0].size>0): 
        likelihood[c5] = 1.0
        if(verbose):
            print("Zone = Green")
    
    if(c6[0].size>0): 
        sigma_32=(D[3,:]-D[2,:])/3.0 
        likelihood[c6]=np.exp(-0.5*((planet_semi_major_axis[c6] - D[2,c6])/sigma_32[c6])**2)
        if(verbose):
            print("Zone = Cold")
 
    if(verbose):
        print("Sim 3: (Liquid Water) = ", likelihood, "\n")

    return likelihood


#Function to determine the likelihood that a planet has a magnetic moment similar to the Earth.
def likelihood_magnetic_moment(stellar_mass, planet_semi_major_axis, planet_system_age, planet_radius, planet_mass, verbose):
    
    if(verbose):
        print("Normalised Magnetic Field")
    
    #---------------------------
    #Pre-amble preparation
    
    #Adding mass and radius of Neptune and Jupiter which are needed in defining planet types
    neptune_radius = 24622000 * u.m; neptune_mass   = 1.02413E26 * u.kg
    jupiter_radius = 71492000 * u.m; jupiter_mass   = 1.8982E27 * u.kg
    
    #calculating density ratio of planet with solar system objects as this is needed several times
    planet_density_ratio_Earth   = (planet_mass.to(u.kg) * const.R_earth**3/planet_radius.to(u.m)**3/const.M_earth)
    planet_density_ratio_Jupiter = (planet_mass.to(u.kg) * jupiter_radius **3/planet_radius.to(u.m)**3/jupiter_mass)
    planet_density_ratio_Neptune = (planet_mass.to(u.kg) * neptune_radius **3/planet_radius.to(u.m)**3/neptune_mass)
    
    #---------------------------
    #Determining whether the planet tidally locked. Using Eq 1 from 
    #Griessmeier+09, Icarus,199,526 to determine how long it takes a planet of given 
    #structure and radius from host star of given mass to become tidally locked.
    #There are *many* assumptions and simplifications that go into this.
    #However, from the discussion in G+09, many of the factors are not expected to vary 
    #by more than factors of a few for reasonable exoplanet properties.
    #The tidal locking timescale is dominated primarily by the star-planet distance (^6) and 
    #secondly the star-planet mass ratio (^2)
    #3 fiducial cases: 1. Earth, 2."small & big super Earth" (6M_E, 1.63R_E & 10M_E, 1.86R_E), 3. "Ocean planet" (6M_E, 2R_E) 
    
    #Determining factors of order unity
    alpha_G07 = 1./3. #structure of planet. Assume planet structure similar to Earth.
    omega_final = 0.0 / u.s #ignore final angular velocity of the planet ("can be neglected...for the planets of interest in this work")
    #omega_initial_upper = 1.8 #upper value of initial angular velocity (based on early Earth-moon system, day length = 13.1 hours).
    #omega_initial_lower = 0.8 #lower value of initial angular velocity (day length = 30 hours).
    omega_initial = 1.0 / u.s #SNL: taking rough average between upper and lower values. Can therefore ignore from tau_synch calculation below.

    #Calculating the tidal dissipation factor
    k_2_p = 0.3     #Apparently this is suitable for Earth, "small super Earth", and "ocean planet" cases (need to check relevance of those for our work) 
    Q_P_Earth = 12  #planetary tidal dissipation factor for Earth (was larger in the past but Earth's shallow sees dissipataed energy. Was high when single continent)
    Q_P_other = 100 #planetary tidal dissipation factor for both the super Earths and Ocean planet
    Q_P_prime_Earth = (3. * Q_P_Earth)/(2.*k_2_p) #modified Q value for Earth
    Q_P_prime_other = (3. * Q_P_other)/(2.*k_2_p) #modified Q value for super Earths and ocean world
    
    #First step of calculating tidal synchronisation time (how long till a planet gets tidally locked)
    tau_synch = (4./9.) * alpha_G07 * (planet_radius.to(u.m)**3/const.G/planet_mass.to(u.kg)) \
                 * (omega_initial - omega_final) \
                 * (planet_mass.to(u.kg)/stellar_mass.to(u.kg))**2 \
                 * (planet_semi_major_axis.to(u.m)/planet_radius.to(u.m))**6
    
    #Next step: determining whether to use Earth-like, or "other" (super-Earth/Ocean) tidal dissipation factor
    c1 = np.where(  np.logical_and(planet_mass.to(u.kg) < 2.0 * const.M_earth, planet_radius.to(u.m) < 1.5 * const.R_earth))
    c2 = np.where(~(np.logical_and(planet_mass.to(u.kg) < 2.0 * const.M_earth, planet_radius.to(u.m) < 1.5 * const.R_earth)))
  
    if(c1[0].size>0):
        tau_synch[c1] *=  Q_P_prime_Earth
    if(c2[0].size>0):
        tau_synch[c2] *=  Q_P_prime_other

    #------------------------------
    #Now determining the magnetic moment ratio between the planet and Earth that
    #is used in the likelihood calculation. The ratio depends on whether the 
    #planet is tidally locked, and how its radius and density compare to 
    #terrestrial-like, ice giant and gas giant planets.
    
    # Determining conditions if planets tidally locked and density ranges wrt solar system planets
    tidally_locked     = tau_synch.to(u.Gyr) < planet_system_age.to(u.Gyr)
    earth_like_density = planet_density_ratio_Earth >= 1.0
    ice_giant_density  = np.logical_and(planet_density_ratio_Earth < 1.0, planet_density_ratio_Earth > 0.18)
    neptune_density    = np.logical_and(planet_density_ratio_Earth <=0.18, planet_density_ratio_Earth >0.16) 
    jupiter_density    = planet_density_ratio_Earth <=0.16 
    
    c3 = np.where(tidally_locked)
    c4 = np.where(np.logical_and(~tidally_locked, earth_like_density))
    c5 = np.where(np.logical_and(~tidally_locked, ice_giant_density))
    c6 = np.where(np.logical_and(~tidally_locked, neptune_density))
    c7 = np.where(np.logical_and(~tidally_locked, jupiter_density))
        
    mag_moment_ratio = np.zeros(planet_mass.size) #initialising magnetic moment ratio array
  
    # ---------------------------
    # Planet tidally locked: use Eq7 in Rodriguez-Mozoz & Moya (2017) (originally from Sano93)
    if(c3[0].size>0):
        
        if(verbose):
            print("Planet rotation = locked")
        
        #SNL: need to assume something about angular frequency ratio of planet with that of Earth. Assume omega ratio = 1.
        omega_ratio = 1.0
        mag_moment_ratio[c3] = planet_density_ratio_Earth[c3]**0.5 \
                                 * (planet_radius.to(u.m)[c3]/const.R_earth)**(7./2.) \
                                     * omega_ratio 
                                     
    else: 
        if(verbose):
            print("Planet rotation = Free")  
                               
    # ---------------------------
    # Planet not tidally locked: use Eq 9 & 10 and Table 3 from Rodriguez-Mozoz & Moya (2017)     
    
    # Calculations below require knowing nature of dynamo (multipolar, internally heated, dipolar) to determine alpha_SOC: 
    #  - based on Olson & Christensen, 2006, Earth & Planet Science Letters, 250, 561
    #  - These required obs pars are difficult/unknown even for SS objects -> now way can know these for exoplanets
    #  - Pragmatic solution = use the discussion in section 6 and Figure 7 to determine a planet's SOC by similarity with SS objects
    #       - Earth & Jupiter like = bipolor (alpha_SOC=1.0)
    #       - Neptune-like & between Earth and ice giant = internally heated dynamo (alpha_SOC=0.15)
                       
    if(c4[0].size>0):                                  # Earth-like  
        if(verbose):
            print("Planet density = Earth-like \nEstimated regime = Dipolar") 
        alpha_SOC = 1.0 # dipolar        
        beta_1 = planet_radius.to(u.m)/const.R_earth
        mag_moment_ratio[c4] = alpha_SOC * beta_1[c4]**(10./3.) * beta_1[c4]**(1./3.)
        
    
    if(c5[0].size>0):                                 # Between telluric and ice giant
        if(verbose):
            print("Planet density = Between telluric and ice giant \nEstimated regime = Internally heated dynamo") 
        alpha_SOC = 0.15 # internally heated dynamo
        beta_1 = planet_radius.to(u.m)/const.R_earth
        mag_moment_ratio[c5] = alpha_SOC * 0.45**0.5 * (1.8 * beta_1[c5])**(10./3.) * (4.*beta_1[c5])**(1./3.)
               
    if(c6[0].size>0):                                 # Neptune-like ice giant
        if(verbose):
            print("Planet density = Neptune-like \nEstimated regime = Internally heated dynamo") 
        alpha_SOC = 0.15 # internally heated dynamo
        beta_1 = planet_radius.to(u.m)/neptune_radius
        mag_moment_ratio[c6] = alpha_SOC * 0.18**0.5 * (4.5 * beta_1[c6])**(10./3.) * (20.*beta_1[c6])**(1./3.)
   
    if(c7[0].size>0):                                 # Jupiter-like gas giant
        if(verbose):
            print("Planet density = Jupiter-like \nEstimated regime = Dipolar") 
        alpha_SOC = 1.0 # dipolar 
        beta_1 = planet_radius.to(u.m)/jupiter_radius
        beta_2 = planet_density_ratio_Jupiter
        mag_moment_ratio[c7] = alpha_SOC * 0.16**0.5 * (16.*beta_1[c7]*beta_2[c7])**(10./3.) * (100.*beta_1[c7]*beta_2[c7])**(1./3.)
  
    #-------------------------------
    #Finally, calculating and returning likelihood
    #If magnetic moment ratio > 1.0, planet has stronger magnetic field than Earth so is protected: likelihood = 1.0.
    #If magnetic moment ratio < 1.0, planet has weaker magnetic field, so likelihood drops         
    
    if(verbose):
        print("magnetic moment ratio = ", mag_moment_ratio)
    c8 = np.where(mag_moment_ratio < 1.0)    
        
    likelihood = np.ones(planet_mass.size)
    if(c8[0].size>0):
        likelihood[c8] = np.exp(-0.5*( 3.*(mag_moment_ratio[c8]-1.0))**2 ) 

    if(verbose):
        print("Sim 4: (Magnetic Field) = ", likelihood, "\n")

    return likelihood


#Function to estimate SEPHI.
#Can be estimated with only seven physical characteristics: 
#planetary mass, planetary radius, planetary orbital period, stellar mass, 
#stellar radius, stellar effective temperature and planetary system age.
def get_sephi_RM17(config, this_level, sfh, stellarpop, planetpop):
    
    #Determine likelihoods at 4 different stages
    verbose = False
    likelihood_1 = likelihood_telluric(planet_mass, planet_radius, verbose)
    likelihood_2 = likelihood_atmosphere(planet_mass, planet_radius, verbose)
    likelihood_3 = likelihood_surface_liquid_water(T_eff_star, L_star, planet_mass,  planet_semi_major_axis, verbose)
    likelihood_4 = likelihood_magnetic_moment(stellar_mass, planet_semi_major_axis, planet_system_age, planet_radius, planet_mass, verbose)
    combined     = (likelihood_1 * likelihood_2 * likelihood_3 * likelihood_4)**(1./4.) # Determine SEPHI as geometric mean of the 4 different likelihoods

    return combined
