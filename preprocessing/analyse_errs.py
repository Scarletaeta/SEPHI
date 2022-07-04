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


#def classify():


def get_best_values(df, param):
    """
    Retrieves the parameter value from the exoplanets df that corresponds to the flagged 'best' parameter.
    
    df = input data frame (exoplanets)
    param = the parameter (e.g. 'st_age'
    """
    
    param = param + '_best'
    best_values = np.zeros(len(df), dtype=np.float64)
    
    for m in range(len(df)):
        if df[param].iloc[m] != df[param].iloc[m]:
            
            best_values[m] = np.nan
            
        else:
            best_values[m] = df[ df[param].iloc[m] ].iloc[m]
            
    return best_values