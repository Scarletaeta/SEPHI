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

def classify(means_a, means_b, means_c, means_d, means_e):
    
    values_a, errs_a1, errs_a2, values_b, errs_b1, errs_b2, values_c, errs_c1, errs_c2, values_d, errs_d1, errs_d2, values_e, errs_e1, errs_e2
    
    means_a = combined_percent_errs(values_a, errs_a1, errs_a2)
    means_b = combined_percent_errs(values_b, errs_b1, errs_b2)
    means_c = combined_percent_errs(values_c, errs_c1, errs_c2)
    means_d = combined_percent_errs(values_d, errs_d1, errs_d2)
    means_e = combined_percent_errs(values_e, errs_e1, errs_e2)
    
    # Data frame of percentage errors from all catalogues (NEA, GDR2, Q16, CKSI, CKSII)
    percentage_errs_df = pd.DataFrame(data=[means_a, means_b, means_c, means_d, means_e]).transpose()
    
    # Array of column names indicating which column has the lowest percentage error:
    flags = percentage_errs_df.idxmin(axis=1, skipna=True)
    # Returns the column name/index of the smallest value in each row
    # With skipna=True, NaNs are ignored if there are real values in the row
    # If all values in the row are NaN, then NaN is returned
    
    # Array of all the 'best' values for that parameter:
    #best = 