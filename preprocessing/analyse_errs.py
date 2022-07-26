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
    
    #TODO: If the value is not nan but the uncertainties are, then give it a mean unc of 1 million so that it will get picked if no others are available
    
    return means


def get_best_values(df, param):
    """
    Retrieves the parameter value from the exoplanets df that corresponds to the flagged 'best' parameter.
    
    df = input data frame (exoplanets) that contains columns with string identifiers for the heading of the column containing the 'best' version of a particular parameter for each exoplanet 
    param = the parameter (e.g. 'st_age')
    """
    
    # The column name for the list/df of best parameters (e.g. 'st_age_best') in the exoplanets df:
    param = param + '_best'
    
    # Empty array for the 'best' parameters:
    best_values = np.zeros(len(df), dtype=np.float64)
    
    # For all items in the column names 'best'
    for m in range(len(df)):
        
        # If entry m is np.nan make the best value np.nan:
        if df[param].iloc[m] != df[param].iloc[m]:
            # one of the properties of np.nan is that np.nan != np.nan
            best_values[m] = np.nan
        
        # If the best values contain a string (not nan) then go to the corresponding column (with the heading 'param') and get the value listed for that exoplanet (index m):
        else:
            best_values[m] = df[ df[param].iloc[m] ].iloc[m]
            
    return best_values