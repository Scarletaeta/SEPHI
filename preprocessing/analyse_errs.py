import numpy as np

def calc_percent_errs(values, errs):
    """
    values = values array/df
    err = errors array/df
    
    returns percent_errs = df of percentage errors
    """
    
    values.to_numpy()
    errs.to_numpy()
    
    percent_errs =  np.absolute( np.multiply(errs, values**(-1)) ) * 100
    return percent_errs.to_numpy()

def combined_percent_errs(values, errs1, errs2):
    """
    values = values array/df
    errs1 = +ve errors array/df
    errs2 = -ve errors array/df
    
    returns a df of mean percentage errors
    """
    
    #values.to_numpy() These get converted to numpy in the percent_err function above
    #errs1.to_numpy()
    #errs2.to_numpy()
    
    means = ( calc_percent_errs(values, errs1) + calc_percent_errs(values, errs2) ) / 2
    return means