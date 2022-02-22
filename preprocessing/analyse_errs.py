import numpy as np
import pandas as pd

# TODO: consider only recording mean % unc in data frame / calcing % errors in the function only

# A function for error classification
def classify_err(og_value, og_err1, og_err2, calc_err1, calc_err2): #calc_value, 
    """
    og_value: the original stellar parameter value [any units]
    og_err1: the original positive error (on the given values) [any units]
    og_err2: the original negative error (on the given values) [any units]
    calc_value: [TODO: calc % errors here instead?] [any units]
    calc_err1: the calculated (new) positive error (on the calculated values) [%]
    calc_err2: the calculated (new) negative error (on the calculated values) [%]
    
    Returns a classification according to the value with the best (lowest) errors:
    - 0 (equal)
    - 1 (st_lum_combined is 'best')
    - 2 (calc_L_combined is 'best')
    - 3 (no classification, neither have both +ve and -ve errors avaiable)
    """
    
    # If both positive and negative errors are available for both the original and calculated errors:
    if( pd.notna(og_err1) and pd.notna(og_err2) and pd.notna(calc_err1) and pd.notna(calc_err2) ):
        # Calculate the percentage errors in og_err:
        percent_err1 = og_err1 / og_value * 100
        percent_err2 = (-1) * og_err2 / og_value * 100
        
        # The mean percentage error on the original value:
        og_combined = ( percent_err1 + percent_err2 ) / 2
        
        # The mean percentage error on the calculated value:
        calc_combined = ( calc_err1 + calc_err2 ) / 2
        
        # Compare the mean percentage errors and assign a classification accorsing to which is smallest:
        if og_combined == calc_combined:
            classification = 0
        elif og_combined < calc_combined:
            classification = 1
        else: #calc_combined < og_combined:
            classification = 2
            
    # If the error is available on og_value but not on calc_value, count og_value as the 'better' one:
    elif( pd.notna(og_err1) and pd.notna(og_err2) and pd.isna(calc_err1) and pd.isna(calc_err2) ):
        classification = 1
    
     # If the error is available on calc_value and but not on og_value, count calc_value as the 'better' one:
    elif( pd.isna(og_err1) and pd.isna(og_err2) and pd.notna(calc_err1) and pd.notna(calc_err2) ):
        classification = 2
    
    # If neither og_value nor calc_value have both positive and negative errors, then they get no classification (3):
    else:
        classification = 3
        
        
    return classification