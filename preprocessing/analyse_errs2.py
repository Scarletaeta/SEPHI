import numpy as np

def calc_percent_errs(values, errs):
    """
    values = values numpy array
    err = errors numpy array
    
    returns percent_errs = df of percentage errors
    """
    
    #values.to_numpy()
    #errs.to_numpy()
    
    percent_errs =  np.absolute( np.multiply(errs, values**(-1)) ) * 100
    return percent_errs

def mean_percent_errs(data):
    """
    data = data frame: values, errs1, errs2
    returns an array of mean percentage errors
    """
    
    values = data.iloc[:,0].to_numpy() # values
    errs1 = data.iloc[:,1].to_numpy() # errs1
    errs2 = data.iloc[:,2].to_numpy() # errs2
    
    #values = data[0].to_numpy() # values
    #errs1 = data[1].to_numpy() # errs1
    #errs2 = data[2].to_numpy() # errs2
    
    percent_errs1 =  np.absolute( np.multiply(errs1, values**(-1)) ) * 100
    percent_errs2 =  np.absolute( np.multiply(errs2, values**(-1)) ) * 100
    
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