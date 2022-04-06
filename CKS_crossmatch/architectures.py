import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import ticker
from copy import deepcopy
import time
from scipy.stats import ks_2samp as ks
from scipy.stats import pearsonr
from scipy.stats import spearmanr

# from matplotlib import rc
# rc('font',**{'family':'serif','serif':['Times New Roman'],'size':16})
# rc('text',usetex=True)
# rc('patch',antialiased=False)
# from matplotlib import rcParams
# rcParams['font.size'] = 16.0
# plt.rc('font', size=16)

import seaborn as sns
import pandas as pd
import pymc3 as pm
import theano.tensor as tt



# OPTIONS

calc_model = 1


# READ

data = pd.read_csv('../data/planetdata.csv')
star_labels = ['hostnames', 'fst_mass','fst_masserr1','fst_masserr2','fst_age','fst_ageerr1','fst_ageerr2','fst_met','fst_meterr1','fst_meterr2','fst_dist','fst_disterr1','fst_disterr2']
planet_labels = ['fpl_letter','fpl_bmasse','fpl_bmasseerr1','fpl_bmasseerr2','fpl_smax','fpl_smaxerr1','fpl_smaxerr2','fpl_orbper','fpl_orbpererr1','fpl_orbpererr2',
                 'fpl_eccen','fpl_eccenerr1','fpl_eccenerr2','fpl_rade','fpl_radeerr1','fpl_radeerr2','fpl_dens','fpl_denserr1','fpl_denserr2']
analysis_labels = ['Nss','Pnull','BIC1','rhoN','Phigh','HJflag','insamp']
stars = deepcopy(data[star_labels])
planets = deepcopy(data[planet_labels])
analysis = deepcopy(data[analysis_labels])


# FORMAT

for label in star_labels[1:]:
    stars[label] = pd.to_numeric(stars[label], errors='coerce')
    if 'err2' in label:
        stars[label] = -stars[label]
for label in planet_labels[1:]:
    planets[label] = pd.to_numeric(planets[label], errors='coerce')
    if 'err2' in label:
        planets[label] = -planets[label]
for label in analysis_labels[:5]:
    analysis[label] = pd.to_numeric(analysis[label], errors='coerce')


# FIX OVERDENSITY STATS OF MULTIPLES

unique_all, counts_all = np.unique(stars['hostnames'].values, return_counts=True)
multiples = unique_all[counts_all > 1]
for sys in multiples:
    idx = (stars['hostnames'] == sys)
    analysis['Pnull'][idx] = np.mean(analysis['Pnull'][idx])
    analysis['rhoN'][idx] = np.mean(analysis['rhoN'][idx])
    analysis['Phigh'][idx] = np.mean(analysis['Phigh'][idx])


# SUBSAMPLES

giants = planets['fpl_bmasse'] >= 50.
neptunes = (planets['fpl_bmasse'] < 50.) & (planets['fpl_bmasse'] > 5.)
rockys = planets['fpl_bmasse'] <= 5.
low_orig = analysis['Phigh'] < 0.16
high_orig = analysis['Phigh'] > 0.84
ambiguous = (analysis['Phigh'] >= 0.16) & (analysis['Phigh'] <= 0.84)
HJs = analysis['HJflag'] == 'Y'
cuts = (analysis['Pnull'] < 0.05) & (analysis['Nss'] >= 400) & (stars['fst_age'] >= 1.) & (stars['fst_age'] <= 4.5) & (stars['fst_mass'] >= 0.7) & (stars['fst_mass'] <= 2.)
masses = ~np.isnan(planets['fpl_bmasseerr1'])
radii = ~np.isnan(planets['fpl_radeerr1'])

unique_all, counts_all = np.unique(stars['hostnames'].values[cuts], return_counts=True)
unique_low, counts_low = np.unique(stars['hostnames'].values[cuts & low_orig], return_counts=True)
unique_high, counts_high = np.unique(stars['hostnames'].values[cuts & high_orig], return_counts=True)

np_all = np.arange(1, np.max(counts_all)+1)
nsys_all = np.bincount(counts_all)[1:]
np_low = np.arange(1, np.max(counts_low)+1)
nsys_low = np.bincount(counts_low)[1:]
np_high = np.arange(1, np.max(counts_high)+1)
nsys_high = np.bincount(counts_high)[1:]

closein = planets['fpl_smax'] <= 30.
unique_all30, counts_all30 = np.unique(stars['hostnames'].values[cuts & closein], return_counts=True)
unique_low30, counts_low30 = np.unique(stars['hostnames'].values[cuts & low_orig & closein], return_counts=True)
unique_high30, counts_high30 = np.unique(stars['hostnames'].values[cuts & high_orig & closein], return_counts=True)

np_all30 = np.arange(1, np.max(counts_all30)+1)
nsys_all30 = np.bincount(counts_all30)[1:]
np_low30 = np.arange(1, np.max(counts_low30)+1)
nsys_low30 = np.bincount(counts_low30)[1:]
np_high30 = np.arange(1, np.max(counts_high30)+1)
nsys_high30 = np.bincount(counts_high30)[1:]

npair_per = np.zeros_like(np_all)
for i in range(1,np.size(np_all)):
    npair_per[i] = np.math.factorial(np_all[i])/(np.math.factorial(2)*np.math.factorial(np_all[i]-2))
npair_all = np.sum(nsys_all*npair_per)
npair_low = np.sum(nsys_low*npair_per[0:np.max(np_low)])
npair_high = np.sum(nsys_high*npair_per[0:np.max(np_high)])

subs = [unique_low, unique_high]
for sname in subs:
    ns = len(sname)
    samp = np.array([])
    for i in range(ns):
        samp = np.append(samp, np.where(stars['hostnames'] == sname[i])[0])
    ms_m = np.percentile(stars['fst_mass'][samp],50.)
    ms_0 = ms_m-np.percentile(stars['fst_mass'][samp],16.)
    ms_2 = np.percentile(stars['fst_mass'][samp],84.)-ms_m
    str_m = '$'+str(np.round(ms_m,2))+'_{-'+str(np.round(ms_0,2))+'}^{+'+str(np.round(ms_2,2))+'}$'
    met_m = np.nanpercentile(stars['fst_met'][samp],50.)
    met_0 = met_m-np.nanpercentile(stars['fst_met'][samp],16.)
    met_2 = np.nanpercentile(stars['fst_met'][samp],84.)-met_m
    str_met = '$'+str(np.round(met_m,2))+'_{-'+str(np.round(met_0,2))+'}^{+'+str(np.round(met_2,2))+'}$'
    age_m = np.nanpercentile(stars['fst_age'][samp],50.)
    age_0 = age_m-np.nanpercentile(stars['fst_age'][samp],16.)
    age_2 = np.nanpercentile(stars['fst_age'][samp],84.)-age_m
    str_age = '$'+str(np.round(age_m,1))+'_{-'+str(np.round(age_0,1))+'}^{+'+str(np.round(age_2,1))+'}$'
    d_m = np.percentile(stars['fst_dist'][samp],50.)
    d_0 = d_m-np.percentile(stars['fst_dist'][samp],16.)
    d_2 = np.percentile(stars['fst_dist'][samp],84.)-d_m
    str_d = '$'+str(np.round(d_m,0))+'_{-'+str(np.round(d_0,0))+'}^{+'+str(np.round(d_2,0))+'}$'
    str_ntotal = '$'+str(np.size(planets['fpl_radeerr1'][samp]))+'$'
    str_ntransit = '$'+str(np.sum(~np.isnan(planets['fpl_radeerr1'][samp])))+'$'
    str_nrv = '$'+str(np.sum(~np.isnan(planets['fpl_bmasseerr1'][samp])))+'$'
    str_ntotal_sys = '$'+str(ns)+'$'
    str_ntransit_sys = '$'+str(np.size(np.unique(stars['hostnames'][samp][~np.isnan(planets['fpl_radeerr1'][samp])])))+'$'
    str_nrv_sys = '$'+str(np.size(np.unique(stars['hostnames'][samp][~np.isnan(planets['fpl_bmasseerr1'][samp])])))+'$'
    print(str_m+' & '+str_met+' & '+str_age+' & '+str_d+' & '+str_ntotal_sys+' & '+str_ntransit_sys+' & '+str_nrv_sys)

multiple_all = unique_all[counts_all > 1]
multiple_low = unique_low[counts_low > 1]
multiple_high = unique_high[counts_high > 1]

subs = [multiple_low, multiple_high]
for sname in subs:
    ns = len(sname)
    samp = np.array([])
    for i in range(ns):
        samp = np.append(samp, np.where(stars['hostnames'] == sname[i])[0])
    ms_m = np.percentile(stars['fst_mass'][samp],50.)
    ms_0 = ms_m-np.percentile(stars['fst_mass'][samp],16.)
    ms_2 = np.percentile(stars['fst_mass'][samp],84.)-ms_m
    str_m = '$'+str(np.round(ms_m,2))+'_{-'+str(np.round(ms_0,2))+'}^{+'+str(np.round(ms_2,2))+'}$'
    met_m = np.nanpercentile(stars['fst_met'][samp],50.)
    met_0 = met_m-np.nanpercentile(stars['fst_met'][samp],16.)
    met_2 = np.nanpercentile(stars['fst_met'][samp],84.)-met_m
    str_met = '$'+str(np.round(met_m,2))+'_{-'+str(np.round(met_0,2))+'}^{+'+str(np.round(met_2,2))+'}$'
    age_m = np.nanpercentile(stars['fst_age'][samp],50.)
    age_0 = age_m-np.nanpercentile(stars['fst_age'][samp],16.)
    age_2 = np.nanpercentile(stars['fst_age'][samp],84.)-age_m
    str_age = '$'+str(np.round(age_m,1))+'_{-'+str(np.round(age_0,1))+'}^{+'+str(np.round(age_2,1))+'}$'
    d_m = np.percentile(stars['fst_dist'][samp],50.)
    d_0 = d_m-np.percentile(stars['fst_dist'][samp],16.)
    d_2 = np.percentile(stars['fst_dist'][samp],84.)-d_m
    str_d = '$'+str(np.round(d_m,0))+'_{-'+str(np.round(d_0,0))+'}^{+'+str(np.round(d_2,0))+'}$'
    str_ntotal = '$'+str(np.size(planets['fpl_radeerr1'][samp]))+'$'
    str_ntransit = '$'+str(np.sum(~np.isnan(planets['fpl_radeerr1'][samp])))+'$'
    str_nrv = '$'+str(np.sum(~np.isnan(planets['fpl_bmasseerr1'][samp])))+'$'
    str_ntotal_sys = '$'+str(ns)+'$'
    str_ntransit_sys = '$'+str(np.size(np.unique(stars['hostnames'][samp][~np.isnan(planets['fpl_radeerr1'][samp])])))+'$'
    str_nrv_sys = '$'+str(np.size(np.unique(stars['hostnames'][samp][~np.isnan(planets['fpl_bmasseerr1'][samp])])))+'$'
    print(str_m+' & '+str_met+' & '+str_age+' & '+str_d+' & '+str_ntotal_sys+' & '+str_ntransit_sys+' & '+str_nrv_sys)

multiple = np.concatenate([multiple_low,multiple_high])
ns = len(multiple)
samp = np.array([])
for i in range(ns):
    samp = np.append(samp, np.where(stars['hostnames'] == multiple[i])[0])
masserr = np.nanmean(0.5*(stars['fst_masserr1'][samp]+stars['fst_masserr2'][samp]))
meterr = np.nanmean(0.5*(stars['fst_meterr1'][samp]+stars['fst_meterr2'][samp]))
ageerr = np.nanmean(0.5*(stars['fst_ageerr1'][samp]+stars['fst_ageerr2'][samp]))
disterr = np.nanmean(0.5*(stars['fst_disterr1'][samp]+stars['fst_disterr2'][samp]))
print(masserr, meterr, ageerr, disterr)

rosetta = pd.read_csv('../data/KOI-Kepler.csv')
for i in range(len(rosetta)):
    rosetta['Name'][i] = np.char.split(rosetta['Name'].values.astype('str'))[i][0]
    rosetta['KOI'][i] = rosetta['KOI'][i][:-3]
rosetta.drop_duplicates(subset=['Name'], inplace=True)
multiple_low_K = np.array([])
multiple_low_KOI = np.array([])
for i in range(len(multiple_low)):
    if multiple_low[i][0:3] == 'Kep':
        multiple_low_K = np.append(multiple_low_K, multiple_low[i])
        multiple_low_KOI = np.append(multiple_low_KOI, rosetta['KOI'].values[np.where(rosetta['Name'].values == multiple_low[i])])
multiple_high_K = np.array([])
multiple_high_KOI = np.array([])
for i in range(len(multiple_high)):
    if multiple_high[i][0:3] == 'Kep':
        multiple_high_K = np.append(multiple_high_K, multiple_high[i])
        multiple_high_KOI = np.append(multiple_high_KOI, rosetta['KOI'].values[np.where(rosetta['Name'].values == multiple_high[i])])

print('')
cks = pd.read_table('../data/Fulton_stars.txt', sep=' ')
subs = [multiple_low_K, multiple_high_K]
subs_KOI = [multiple_low_KOI, multiple_high_KOI]
for j,sname in enumerate(subs):
    sname_KOI = subs_KOI[j]
    ns = len(sname)
    samp = np.array([])
    samp_KOI = np.array([])
    for i in range(ns):
        match = np.where(cks['KOI'] == sname_KOI[i])[0]
        if np.size(match) > 0:
            samp = np.append(samp, np.where(stars['hostnames'] == sname[i])[0])
    ms_m = np.percentile(stars['fst_mass'][samp],50.)
    ms_0 = ms_m-np.percentile(stars['fst_mass'][samp],16.)
    ms_2 = np.percentile(stars['fst_mass'][samp],84.)-ms_m
    str_m = '$'+str(np.round(ms_m,2))+'_{-'+str(np.round(ms_0,2))+'}^{+'+str(np.round(ms_2,2))+'}$'
    met_m = np.nanpercentile(stars['fst_met'][samp],50.)
    met_0 = met_m-np.nanpercentile(stars['fst_met'][samp],16.)
    met_2 = np.nanpercentile(stars['fst_met'][samp],84.)-met_m
    str_met = '$'+str(np.round(met_m,2))+'_{-'+str(np.round(met_0,2))+'}^{+'+str(np.round(met_2,2))+'}$'
    age_m = np.nanpercentile(stars['fst_age'][samp],50.)
    age_0 = age_m-np.nanpercentile(stars['fst_age'][samp],16.)
    age_2 = np.nanpercentile(stars['fst_age'][samp],84.)-age_m
    str_age = '$'+str(np.round(age_m,1))+'_{-'+str(np.round(age_0,1))+'}^{+'+str(np.round(age_2,1))+'}$'
    d_m = np.percentile(stars['fst_dist'][samp],50.)
    d_0 = d_m-np.percentile(stars['fst_dist'][samp],16.)
    d_2 = np.percentile(stars['fst_dist'][samp],84.)-d_m
    str_d = '$'+str(np.round(d_m,0))+'_{-'+str(np.round(d_0,0))+'}^{+'+str(np.round(d_2,0))+'}$'
    str_ntotal = '$'+str(np.size(planets['fpl_radeerr1'][samp]))+'$'
    str_ntransit = '$'+str(np.sum(~np.isnan(planets['fpl_radeerr1'][samp])))+'$'
    str_nrv = '$'+str(np.sum(~np.isnan(planets['fpl_bmasseerr1'][samp])))+'$'
    str_ntotal_sys = '$'+str(len(np.unique(stars['hostnames'][samp])))+'$'
    str_ntransit_sys = '$'+str(np.size(np.unique(stars['hostnames'][samp][~np.isnan(planets['fpl_radeerr1'][samp])])))+'$'
    str_nrv_sys = '$'+str(np.size(np.unique(stars['hostnames'][samp][~np.isnan(planets['fpl_bmasseerr1'][samp])])))+'$'
    print(str_m+' & '+str_met+' & '+str_age+' & '+str_d+' & '+str_ntotal_sys+' & '+str_ntransit_sys+' & '+str_nrv_sys)

print('')
subs = [multiple_low_K, multiple_high_K]
subs_KOI = [multiple_low_KOI, multiple_high_KOI]
for j,sname in enumerate(subs):
    sname_KOI = subs_KOI[j]
    ns = len(sname)
    samp = np.array([])
    samp_KOI = np.array([])
    for i in range(ns):
        match = np.where(cks['KOI'] == sname_KOI[i])[0]
        if np.size(match) > 0:
            match_NASA = np.where(stars['hostnames'] == sname[i])[0]
            samp = np.append(samp, match_NASA)
            samp_KOI = np.append(samp_KOI, np.repeat(match[0], np.size(match_NASA))).flatten()
    ms_m = np.percentile(cks['mass'][samp_KOI],50.)
    ms_0 = ms_m-np.percentile(cks['mass'][samp_KOI],16.)
    ms_2 = np.percentile(cks['mass'][samp_KOI],84.)-ms_m
    str_m = '$'+str(np.round(ms_m,2))+'_{-'+str(np.round(ms_0,2))+'}^{+'+str(np.round(ms_2,2))+'}$'
    met_m = np.nanpercentile(cks['FeH'][samp_KOI],50.)
    met_0 = met_m-np.nanpercentile(cks['FeH'][samp_KOI],16.)
    met_2 = np.nanpercentile(cks['FeH'][samp_KOI],84.)-met_m
    str_met = '$'+str(np.round(met_m,2))+'_{-'+str(np.round(met_0,2))+'}^{+'+str(np.round(met_2,2))+'}$'
    age_m = np.nanpercentile(10.**(cks['age'][samp_KOI]-9.),50.)
    age_0 = age_m-np.nanpercentile(10.**(cks['age'][samp_KOI]-9.),16.)
    age_2 = np.nanpercentile(10.**(cks['age'][samp_KOI]-9.),84.)-age_m
    str_age = '$'+str(np.round(age_m,1))+'_{-'+str(np.round(age_0,1))+'}^{+'+str(np.round(age_2,1))+'}$'
    d_m = np.percentile(stars['fst_dist'][samp],50.)
    d_0 = d_m-np.percentile(stars['fst_dist'][samp],16.)
    d_2 = np.percentile(stars['fst_dist'][samp],84.)-d_m
    str_d = '$'+str(np.round(d_m,0))+'_{-'+str(np.round(d_0,0))+'}^{+'+str(np.round(d_2,0))+'}$'
    str_ntotal = '$'+str(np.size(planets['fpl_radeerr1'][samp]))+'$'
    str_ntransit = '$'+str(np.sum(~np.isnan(planets['fpl_radeerr1'][samp])))+'$'
    str_nrv = '$'+str(np.sum(~np.isnan(planets['fpl_bmasseerr1'][samp])))+'$'
    str_ntotal_sys = '$'+str(len(np.unique(stars['hostnames'][samp])))+'$'
    str_ntransit_sys = '$'+str(np.size(np.unique(stars['hostnames'][samp][~np.isnan(planets['fpl_radeerr1'][samp])])))+'$'
    str_nrv_sys = '$'+str(np.size(np.unique(stars['hostnames'][samp][~np.isnan(planets['fpl_bmasseerr1'][samp])])))+'$'
    print(str_m+' & '+str_met+' & '+str_age+' & '+str_d+' & '+str_ntotal_sys+' & '+str_ntransit_sys+' & '+str_nrv_sys)

quit()


def find_pairs(stars, indices, multiple_list, npair_per, tar_array):

    count = 0
    for name in multiple_list:
        use = np.where(stars['hostnames'] == name)[0]
        nhere = npair_per[len(use)-1]
        tar_array[count:count+nhere, :] = np.array([[i, j] for i in indices[use] for j in indices[use] if i<j])
        count += nhere

    return tar_array


pair_all = np.zeros([npair_all, 2])
pair_low = np.zeros([npair_low, 2])
pair_high = np.zeros([npair_high, 2])
idx_all = np.arange(np.size(stars['hostnames']))

pair_all = find_pairs(stars, idx_all, multiple_all, npair_per, pair_all)
pair_low = find_pairs(stars, idx_all, multiple_low, npair_per, pair_low)
pair_high = find_pairs(stars, idx_all, multiple_high, npair_per, pair_high)
pout_pin_all = np.zeros(npair_all)
pout_pin_low = np.zeros(npair_low)
pout_pin_high = np.zeros(npair_high)
mpair_all = np.zeros(npair_all)
mpair_low = np.zeros(npair_low)
mpair_high = np.zeros(npair_high)
mjme = 317.83
for i in range(npair_all):
    pout_pin_all[i] = np.max(planets['fpl_orbper'][pair_all[i,:]])/np.min(planets['fpl_orbper'][pair_all[i,:]])
    mpair_all[i] = (planets['fpl_bmasse'][pair_all[i,1]]+planets['fpl_bmasse'][pair_all[i,0]])/mjme
for i in range(npair_low):
    pout_pin_low[i] = np.max(planets['fpl_orbper'][pair_low[i,:]])/np.min(planets['fpl_orbper'][pair_low[i,:]])
    mpair_low[i] = (planets['fpl_bmasse'][pair_low[i,1]]+planets['fpl_bmasse'][pair_low[i,0]])/mjme
for i in range(npair_high):
    pout_pin_high[i] = np.max(planets['fpl_orbper'][pair_high[i,:]])/np.min(planets['fpl_orbper'][pair_high[i,:]])
    mpair_high[i] = (planets['fpl_bmasse'][pair_high[i,1]]+planets['fpl_bmasse'][pair_high[i,0]])/mjme

print(nsys_all[0]/np.sum(nsys_all[1:]), nsys_low[0]/np.sum(nsys_low[1:]), nsys_high[0]/np.sum(nsys_high[1:]))
print(nsys_all30[0]/np.sum(nsys_all30[1:]), nsys_low30[0]/np.sum(nsys_low30[1:]), nsys_high30[0]/np.sum(nsys_high30[1:]))

# PLOTS

xs = 8
ys = 6
edgepad = 0.15
textpad = 0.03
msize = 5
alp = 1

dx = 0.02
xmin = 0.5
xmax = np.max(counts_all)+0.5
ymin = 0.3
ymax = 10.**np.ceil(np.log10(np.max(nsys_all)))

fig, ax = plt.subplots(1, 1, figsize=(xs, ys))
ax.errorbar(np_all, nsys_all, yerr=np.sqrt(nsys_all), fmt='o', c='k', ls='-', ms=msize, label='All systems')
ax.errorbar(np_low-dx, nsys_low, yerr=np.sqrt(nsys_low), fmt='o', c='b', ls='-', ms=msize, label='Field')
ax.errorbar(np_high+dx, nsys_high, yerr=np.sqrt(nsys_high), fmt='o', c='r', ls='-', ms=msize, label='Overdensities')
ax.set_yscale('log')
ax.set_xlim((xmin,xmax))
ax.set_ylim((ymin,ymax))
ax.set_xlabel('Planets per system')
ax.set_ylabel('Observed systems')
ax.legend(loc=1, frameon=0)
fig.subplots_adjust(left=edgepad, right=1-edgepad, top=1-edgepad, bottom=edgepad)
figsave = plt.gcf()
figsave.set_size_inches(xs,ys)
plt.savefig('figures/planets_per_system.pdf',bbox_inches='tight',dpi=300)

xs = 8
ys = 6
edgepad = 0.15
msize = 5
alp = 1

dx = 0.02
xmin = 0.5
xmax = np.max(counts_all)+0.5
ymin = 0.3
ymax = 10.**np.ceil(np.log10(np.max(nsys_all)))

fig, ax = plt.subplots(1, 1, figsize=(xs, ys))
ax.errorbar(np_all30, nsys_all30, yerr=np.sqrt(nsys_all30), fmt='o', c='k', ls='-', ms=msize, label='All systems')
ax.errorbar(np_low30-dx, nsys_low30, yerr=np.sqrt(nsys_low30), fmt='o', c='b', ls='-', ms=msize, label='Field')
ax.errorbar(np_high30+dx, nsys_high30, yerr=np.sqrt(nsys_high30), fmt='o', c='r', ls='-', ms=msize, label='Overdensities')
ax.set_yscale('log')
ax.set_xlim((xmin,xmax))
ax.set_ylim((ymin,ymax))
ax.set_xlabel('Planets per system with $a<30$ AU')
ax.set_ylabel('Observed systems')
ax.legend(loc=1, frameon=0)
fig.subplots_adjust(left=edgepad, right=1-edgepad, top=1-edgepad, bottom=edgepad)
figsave = plt.gcf()
figsave.set_size_inches(xs,ys)
plt.savefig('figures/planets_per_system_30au.pdf',bbox_inches='tight',dpi=300)

xmin = 0.5
xmax = 3.5
ymin = 0.3
ymax = 3.

fig, ax = plt.subplots(1, 1, figsize=(xs, ys))
ylow = nsys_low[0:3]/nsys_all[0:3]*nsys_all[0]/nsys_low[0]
yhigh = nsys_high[0:3]/nsys_all[0:3]*nsys_all[0]/nsys_high[0]
ax.errorbar(np_all[0:3], nsys_all[0:3]/nsys_all[0:3], yerr=0., fmt='o', c='k', ls='-', ms=msize, label='All systems')
ax.errorbar(np_low[0:3]-dx, ylow, yerr=ylow*np.sqrt(1./nsys_low[0:3]+1./nsys_all[0:3]), fmt='o', c='b', ls='-', ms=msize, label='Field')
ax.errorbar(np_high[0:3]+dx, yhigh, yerr=yhigh*np.sqrt(1./nsys_high[0:3]+1./nsys_all[0:3]), fmt='o', c='r', ls='-', ms=msize, label='Overdensities')
ax.set_yscale('log')
ax.set_xlim((xmin,xmax))
ax.set_ylim((ymin,ymax))
ax.set_xlabel('Planets per system')
ax.set_ylabel('Frequency excess (normalised at 1 planet)')
ax.legend(loc=1, frameon=0)
fig.subplots_adjust(left=edgepad, right=1-edgepad, top=1-edgepad, bottom=edgepad)
figsave = plt.gcf()
figsave.set_size_inches(xs,ys)
plt.savefig('figures/planets_per_system_relative.pdf',bbox_inches='tight',dpi=300)


xmin = 0.1
xmax = 100
ymin = .01
ymax = 40
msize = 25

resonances = np.array([5., 4., 3., 2., 5./3., 3./2., 4./3., 5./4., 6./5.])
res_name = np.array(['5:1', '4:1', '3:1', '2:1', '5:3', '3:2', '4:3', '5:4', '6:5'])
resonances_short = np.array([2., 3./2., 4./3.])
poiarr = np.logspace(np.log10(xmin), np.log10(xmax), 10000)

fig, ax = plt.subplots(1, 1, figsize=(xs, ys))
ax.scatter(pout_pin_all-1, mpair_all, marker='o', c='k', s=msize, label='All systems', zorder=3)
ax.scatter(pout_pin_low-1, mpair_low, marker='o', c='b', s=msize, label='Field', zorder=3)
ax.scatter(pout_pin_high-1, mpair_high, marker='o', c='r', s=msize, label='Overdensities', zorder=3)
for res in resonances_short:
    j = 1./(res-1.)
    ax.plot(poiarr, 0.05**(-2.)*np.abs(poiarr-1./j)**2., 'k', lw=0.5, zorder=1)
for i, res in enumerate(resonances):
    j = 1./(res-1.)
    ax.plot([1./j, 1./j], [ymin,20./1.4**i], c='k', ls='--', lw=1, alpha=0.3, zorder=0)
    ax.text(1./j*1.02, 20./1.4**i*1.05, res_name[i], {'ha': 'center', 'va': 'bottom'}, color='k', alpha=0.3)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim((xmin,xmax))
ax.set_ylim((ymin,ymax))
ax.set_xlabel('$P_{\\rm out}/P_{\\rm in}-1$')
ax.set_ylabel('$M_1+M_2$ [M$_{\\rm J}$]')
ax.tick_params(axis='x', which='both', top=True)
ax.tick_params(axis='y', which='both', right=True)
ax.tick_params(which='major', length=6)
ax.tick_params(which='minor', length=3)
ax.legend(loc=1, frameon=0)
fig.subplots_adjust(left=edgepad, right=1-edgepad, top=1-edgepad, bottom=edgepad)
figsave = plt.gcf()
figsave.set_size_inches(xs,ys)
plt.savefig('figures/period_ratios.pdf',bbox_inches='tight',dpi=300)


xmin = 0.1
xmax = 100
ymin = 0
ymax = 15
nbins = 37
edges = np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
left = edges[:-1]
right = edges[1:]
bins = np.sqrt(left*right)
hist_all = np.zeros_like(bins)
hist_low = np.zeros_like(bins)
hist_high = np.zeros_like(bins)
for i in range(nbins):
    hist_all[i] = np.size(np.where((pout_pin_all-1. < right[i]) & (pout_pin_all-1. >= left[i])))
    hist_low[i] = np.size(np.where((pout_pin_low-1. < right[i]) & (pout_pin_low-1. >= left[i])))
    hist_high[i] = np.size(np.where((pout_pin_high-1. < right[i]) & (pout_pin_high-1. >= left[i])))

fig, ax = plt.subplots(1, 1, figsize=(xs, ys))
ax.hist(pout_pin_all-1., color='k', bins=edges, histtype='step', label='All systems', zorder=3)
ax.hist(pout_pin_low-1., color='b', bins=edges, alpha=0.5, label='Field', zorder=3)
ax.hist(pout_pin_high-1., color='r', bins=edges, alpha=0.5, label='Overdensities', zorder=3)
for i, res in enumerate(resonances):
    j = 1./(res-1.)
    ax.plot([1./j, 1./j], [ymin,ymax-.7*i-1.25], c='k', ls='--', lw=1, alpha=0.3, zorder=0)
    ax.text(1./j*1.02, ymax-.7*i-1., res_name[i], {'ha': 'center', 'va': 'bottom'}, color='k', alpha=0.3)
ax.set_xscale('log')
# ax.set_yscale('log')
ax.set_xlim((xmin,xmax))
ax.set_ylim((ymin,ymax))
ax.set_xlabel('$P_{\\rm out}/P_{\\rm in}-1$')
ax.set_ylabel('Number')
ax.tick_params(axis='x', which='both', top=True)
ax.tick_params(axis='y', which='both', right=True)
ax.tick_params(which='major', length=6)
ax.tick_params(which='minor', length=3)
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.legend(loc=1, frameon=0)
fig.subplots_adjust(left=edgepad, right=1-edgepad, top=1-edgepad, bottom=edgepad)
figsave = plt.gcf()
figsave.set_size_inches(xs,ys)
plt.savefig('figures/period_ratios_hist.pdf',bbox_inches='tight',dpi=300)


def is_within_fac(values, tar_values, fac):

    within = np.zeros_like(values)
    for i, val in enumerate(values):
        if np.min(np.abs(np.log10(val/tar_values))) < np.log10(fac):
            within[i] = 1

    return within

minratio = .1
maxratio = 5.
use_all = (pout_pin_all-1. < maxratio) & (pout_pin_all-1. > minratio)
use_low = (pout_pin_low-1. < maxratio) & (pout_pin_low-1. > minratio)
use_high = (pout_pin_high-1. < maxratio) & (pout_pin_high-1. > minratio)
pr_all = pout_pin_all[use_all]
pr_low = pout_pin_low[use_low]
pr_high = pout_pin_high[use_high]

ntol = 10000
tolmin = 1.01
tolmax = 3.
tolerance = 1.+np.logspace(np.log10(tolmin-1.), np.log10(tolmax-1.), ntol)
nwithin_all = np.zeros_like(tolerance)
nwithin_low = np.zeros_like(tolerance)
nwithin_high = np.zeros_like(tolerance)
for i in range(ntol):
    nwithin_all[i] = np.sum(is_within_fac(pr_all-1., resonances, tolerance[i]))
    nwithin_low[i] = np.sum(is_within_fac(pr_low-1., resonances, tolerance[i]))
    nwithin_high[i] = np.sum(is_within_fac(pr_high-1., resonances, tolerance[i]))

pwithin_all = nwithin_all/float(np.size(pr_all))
pwithin_low = nwithin_low/float(np.size(pr_low))
pwithin_high = nwithin_high/float(np.size(pr_high))

xmin = tolmin-1
xmax = tolmax-1
ymin = -0.05
ymax = 1.05

fig, ax = plt.subplots(1, 1, figsize=(xs, ys))
ax.plot(tolerance-1, pwithin_all, '-k', label='All systems', zorder=3)
ax.plot(tolerance-1, pwithin_low, '-b', label='Field', zorder=3)
ax.plot(tolerance-1, pwithin_high, '-r', label='Overdensities', zorder=3)
ax.set_xscale('log')
# ax.set_yscale('log')
ax.set_xlim((xmin,xmax))
ax.set_ylim((ymin,ymax))
ax.set_xlabel('$f-1$')
ax.set_ylabel('Fraction within a factor $f$ of resonance')
ax.tick_params(axis='x', which='both', top=True)
ax.tick_params(axis='y', which='both', right=True)
ax.tick_params(which='major', length=6)
ax.tick_params(which='minor', length=3)
ax.legend(loc=2, frameon=0)
fig.subplots_adjust(left=edgepad, right=1-edgepad, top=1-edgepad, bottom=edgepad)
figsave = plt.gcf()
figsave.set_size_inches(xs,ys)
plt.savefig('figures/resonant_fraction.pdf',bbox_inches='tight',dpi=300)

def get_adj_inn(pair, planets, stars):

    adjacent = np.ones_like(pair[:,0]).astype('bool')
    inner = np.zeros_like(pair[:,0]).astype('int')
    for i in range(len(adjacent)):
        pairs_in_sys = (stars['hostnames'][pair[:,0]] == stars['hostnames'][pair[:,0]].values[i]).values
        if np.sum(pairs_in_sys) > 2:
            periods_in_sys = np.sort(np.unique(np.ndarray.flatten(np.array([planets['fpl_orbper'][pair[:,0]][pairs_in_sys],planets['fpl_orbper'][pair[:,1]][pairs_in_sys]]))))
            idx_period1 = np.where(periods_in_sys == planets['fpl_orbper'][pair[i,0]])[0]
            idx_period2 = np.where(periods_in_sys == planets['fpl_orbper'][pair[i,1]])[0]
            adjacent[i] = (np.abs(idx_period1-idx_period2) == 1)[0]
        else:
            adjacent[i] = True
        inner[i] = int(planets['fpl_orbper'][pair[i,0]] > planets['fpl_orbper'][pair[i,1]])
    outer = 1-inner
    inner_arr = np.zeros([np.size(inner), 2], dtype=bool)
    for i in range(np.size(inner)):
        for j in range(2):
            inner_arr[i,j] = (inner[i] == j)
    outer_arr = (1-inner_arr).astype('bool')

    return adjacent, inner, outer, inner_arr, outer_arr

adjacent_all, inner_all, outer_all, inner_all_arr, outer_all_arr = get_adj_inn(pair_all, planets, stars)
adjacent_low, inner_low, outer_low, inner_low_arr, outer_low_arr = get_adj_inn(pair_low, planets, stars)
adjacent_high, inner_high, outer_high, inner_high_arr, outer_high_arr = get_adj_inn(pair_high, planets, stars)


xs = 8
ys = 8

xmin = 0.2
xmax = 30
ymin = 0.2
ymax = 30
msize = 5
dy = 1.2

fig, ax = plt.subplots(1, 1, figsize=(xs, ys))
ax.plot([xmin,xmax], [ymin,ymax], ':k', alpha=0.3, zorder=0)
var = planets['fpl_rade']
errmin = planets['fpl_radeerr2']
errmax = planets['fpl_radeerr1']

x = var[pair_all[:,0]][adjacent_all]
y = var[pair_all[:,1]][adjacent_all]
use = ~np.isnan(x).values & ~np.isnan(y).values & (x.values >= xmin) & (x.values <= xmax) & (y.values >= ymin) & (y.values <= ymax)
x = x[use]
y = y[use]
xerrmin = errmin[pair_all[:,0]][adjacent_all][use]
xerrmax = errmax[pair_all[:,0]][adjacent_all][use]
yerrmin = errmin[pair_all[:,1]][adjacent_all][use]
yerrmax = errmax[pair_all[:,1]][adjacent_all][use]
sr, sp = spearmanr(x, y)
pr, pp = pearsonr(x, y)
ax.errorbar(x, y, xerr=[xerrmin,xerrmax], yerr=[yerrmin,yerrmax], fmt='o', c='k', ms=msize, label='All systems', zorder=3)
ax.text(1.-textpad, textpad, 'Pearson $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(pr, 2), np.round(np.log10(pp), 2)), color='k', ha='right', va='bottom', transform = ax.transAxes)
ax.text(1.-textpad, textpad+1.*dy*textpad, 'Spearman $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(sr, 2), np.round(np.log10(sp), 2)), color='k', ha='right', va='bottom', transform = ax.transAxes)

x = var[pair_low[:,0]][adjacent_low]
y = var[pair_low[:,1]][adjacent_low]
use = ~np.isnan(x).values & ~np.isnan(y).values & (x.values >= xmin) & (x.values <= xmax) & (y.values >= ymin) & (y.values <= ymax)
x = x[use]
y = y[use]
xerrmin = errmin[pair_low[:,0]][adjacent_low][use]
xerrmax = errmax[pair_low[:,0]][adjacent_low][use]
yerrmin = errmin[pair_low[:,1]][adjacent_low][use]
yerrmax = errmax[pair_low[:,1]][adjacent_low][use]
sr, sp = spearmanr(x, y)
pr, pp = pearsonr(x, y)
ax.errorbar(x, y, xerr=[xerrmin,xerrmax], yerr=[yerrmin,yerrmax], fmt='o', c='b', ms=msize, label='Field', zorder=3)
ax.text(1.-textpad, textpad+2.*dy*textpad, 'Pearson $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(pr, 2), np.round(np.log10(pp), 2)), color='b', ha='right', va='bottom', transform = ax.transAxes)
ax.text(1.-textpad, textpad+3.*dy*textpad, 'Spearman $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(sr, 2), np.round(np.log10(sp), 2)), color='b', ha='right', va='bottom', transform = ax.transAxes)

x = var[pair_high[:,0]][adjacent_high]
y = var[pair_high[:,1]][adjacent_high]
use = ~np.isnan(x).values & ~np.isnan(y).values & (x.values >= xmin) & (x.values <= xmax) & (y.values >= ymin) & (y.values <= ymax)
x = x[use]
y = y[use]
xerrmin = errmin[pair_high[:,0]][adjacent_high][use]
xerrmax = errmax[pair_high[:,0]][adjacent_high][use]
yerrmin = errmin[pair_high[:,1]][adjacent_high][use]
yerrmax = errmax[pair_high[:,1]][adjacent_high][use]
sr, sp = spearmanr(x, y)
pr, pp = pearsonr(x, y)
ax.errorbar(x, y, xerr=[xerrmin,xerrmax], yerr=[yerrmin,yerrmax], fmt='o', c='r', ms=msize, label='Overdensities', zorder=3)
ax.text(1.-textpad, textpad+4.*dy*textpad, 'Pearson $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(pr, 2), np.round(np.log10(pp), 2)), color='r', ha='right', va='bottom', transform = ax.transAxes)
ax.text(1.-textpad, textpad+5.*dy*textpad, 'Spearman $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(sr, 2), np.round(np.log10(sp), 2)), color='r', ha='right', va='bottom', transform = ax.transAxes)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim((xmin,xmax))
ax.set_ylim((ymin,ymax))
ax.set_xlabel('Inner planet radius [R$_\\oplus$]')
ax.set_ylabel('Outer planet radius [R$_\\oplus$]')
ax.set_aspect(1.)
ax.tick_params(axis='x', which='both', top=True)
ax.tick_params(axis='y', which='both', right=True)
ax.tick_params(which='major', length=6)
ax.tick_params(which='minor', length=3)
ax.legend(loc='upper left', frameon=0)
fig.subplots_adjust(left=edgepad, right=1-edgepad, top=1-edgepad, bottom=edgepad)
figsave = plt.gcf()
figsave.set_size_inches(xs,ys)
plt.savefig('figures/pod_radii.pdf',bbox_inches='tight',dpi=300)


xs = 8
ys = 8

xmin = 0.2
xmax = 30
ymin = 0.2
ymax = 30
msize = 5
dy = 1.2

fig, ax = plt.subplots(1, 1, figsize=(xs, ys))
ax.plot([xmin,xmax], [ymin,ymax], ':k', alpha=0.3, zorder=0)
var = planets['fpl_rade']
errmin = planets['fpl_radeerr2']
errmax = planets['fpl_radeerr1']

x = var[pair_all[:,0]][adjacent_all]
y = var[pair_all[:,1]][adjacent_all]
err = 0.5*(errmin+errmax)[pair_all[:,0]][adjacent_all]
use = ~np.isnan(x).values & ~np.isnan(y).values & (x.values >= xmin) & (x.values <= xmax) & (y.values >= ymin) & (y.values <= ymax) & (err.values > 0.)
x = x[use]
y = y[use]
xerrmin = errmin[pair_all[:,0]][adjacent_all][use]
xerrmax = errmax[pair_all[:,0]][adjacent_all][use]
yerrmin = errmin[pair_all[:,1]][adjacent_all][use]
yerrmax = errmax[pair_all[:,1]][adjacent_all][use]
if len(x) > 1:
    sr, sp = spearmanr(x, y)
    pr, pp = pearsonr(x, y)
else:
    sr, sp, pr, pp = 0., 0., 0., 0.
ax.errorbar(x, y, xerr=[xerrmin,xerrmax], yerr=[yerrmin,yerrmax], fmt='o', c='k', ms=msize, label='All systems', zorder=3)
ax.text(1.-textpad, textpad, 'Pearson $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(pr, 2), np.round(np.log10(pp), 2)), color='k', ha='right', va='bottom', transform = ax.transAxes)
ax.text(1.-textpad, textpad+1.*dy*textpad, 'Spearman $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(sr, 2), np.round(np.log10(sp), 2)), color='k', ha='right', va='bottom', transform = ax.transAxes)

x = var[pair_low[:,0]][adjacent_low]
y = var[pair_low[:,1]][adjacent_low]
err = 0.5*(errmin+errmax)[pair_low[:,0]][adjacent_low]
use = ~np.isnan(x).values & ~np.isnan(y).values & (x.values >= xmin) & (x.values <= xmax) & (y.values >= ymin) & (y.values <= ymax) & (err.values > 0.)
x = x[use]
y = y[use]
xerrmin = errmin[pair_low[:,0]][adjacent_low][use]
xerrmax = errmax[pair_low[:,0]][adjacent_low][use]
yerrmin = errmin[pair_low[:,1]][adjacent_low][use]
yerrmax = errmax[pair_low[:,1]][adjacent_low][use]
if len(x) > 1:
    sr, sp = spearmanr(x, y)
    pr, pp = pearsonr(x, y)
else:
    sr, sp, pr, pp = 0., 0., 0., 0.
ax.errorbar(x, y, xerr=[xerrmin,xerrmax], yerr=[yerrmin,yerrmax], fmt='o', c='b', ms=msize, label='Field', zorder=3)
ax.text(1.-textpad, textpad+2.*dy*textpad, 'Pearson $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(pr, 2), np.round(np.log10(pp), 2)), color='b', ha='right', va='bottom', transform = ax.transAxes)
ax.text(1.-textpad, textpad+3.*dy*textpad, 'Spearman $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(sr, 2), np.round(np.log10(sp), 2)), color='b', ha='right', va='bottom', transform = ax.transAxes)

x = var[pair_high[:,0]][adjacent_high]
y = var[pair_high[:,1]][adjacent_high]
err = 0.5*(errmin+errmax)[pair_high[:,0]][adjacent_high]
use = ~np.isnan(x).values & ~np.isnan(y).values & (x.values >= xmin) & (x.values <= xmax) & (y.values >= ymin) & (y.values <= ymax) & (err.values > 0.)
x = x[use]
y = y[use]
xerrmin = errmin[pair_high[:,0]][adjacent_high][use]
xerrmax = errmax[pair_high[:,0]][adjacent_high][use]
yerrmin = errmin[pair_high[:,1]][adjacent_high][use]
yerrmax = errmax[pair_high[:,1]][adjacent_high][use]
if len(x) > 1:
    sr, sp = spearmanr(x, y)
    pr, pp = pearsonr(x, y)
else:
    sr, sp, pr, pp = 0., 0., 0., 0.
ax.errorbar(x, y, xerr=[xerrmin,xerrmax], yerr=[yerrmin,yerrmax], fmt='o', c='r', ms=msize, label='Overdensities', zorder=3)
ax.text(1.-textpad, textpad+4.*dy*textpad, 'Pearson $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(pr, 2), np.round(np.log10(pp), 2)), color='r', ha='right', va='bottom', transform = ax.transAxes)
ax.text(1.-textpad, textpad+5.*dy*textpad, 'Spearman $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(sr, 2), np.round(np.log10(sp), 2)), color='r', ha='right', va='bottom', transform = ax.transAxes)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim((xmin,xmax))
ax.set_ylim((ymin,ymax))
ax.set_xlabel('Inner planet radius [R$_\\oplus$]')
ax.set_ylabel('Outer planet radius [R$_\\oplus$]')
ax.set_aspect(1.)
ax.tick_params(axis='x', which='both', top=True)
ax.tick_params(axis='y', which='both', right=True)
ax.tick_params(which='major', length=6)
ax.tick_params(which='minor', length=3)
ax.legend(loc='upper left', frameon=0)
fig.subplots_adjust(left=edgepad, right=1-edgepad, top=1-edgepad, bottom=edgepad)
figsave = plt.gcf()
figsave.set_size_inches(xs,ys)
plt.savefig('figures/pod_radii_true.pdf',bbox_inches='tight',dpi=300)


xmin = 0.1
xmax = 2e4
ymin = 0.1
ymax = 2e4
msize = 5
dy = 1.2

fig, ax = plt.subplots(1, 1, figsize=(xs, ys))
ax.plot([xmin,xmax], [ymin,ymax], ':k', alpha=0.3, zorder=0)
var = planets['fpl_bmasse']
errmin = planets['fpl_bmasseerr2']
errmax = planets['fpl_bmasseerr1']

x = var[pair_all[:,0]][adjacent_all]
y = var[pair_all[:,1]][adjacent_all]
use = ~np.isnan(x).values & ~np.isnan(y).values & (x.values >= xmin) & (x.values <= xmax) & (y.values >= ymin) & (y.values <= ymax)
x = x[use]
y = y[use]
xerrmin = errmin[pair_all[:,0]][adjacent_all][use]
xerrmax = errmax[pair_all[:,0]][adjacent_all][use]
yerrmin = errmin[pair_all[:,1]][adjacent_all][use]
yerrmax = errmax[pair_all[:,1]][adjacent_all][use]
sr, sp = spearmanr(x, y)
pr, pp = pearsonr(x, y)
ax.errorbar(x, y, xerr=[xerrmin,xerrmax], yerr=[yerrmin,yerrmax], fmt='o', c='k', ms=msize, label='All systems', zorder=3)
ax.text(1.-textpad, textpad, 'Pearson $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(pr, 2), np.round(np.log10(pp), 2)), color='k', ha='right', va='bottom', transform = ax.transAxes)
ax.text(1.-textpad, textpad+1.*dy*textpad, 'Spearman $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(sr, 2), np.round(np.log10(sp), 2)), color='k', ha='right', va='bottom', transform = ax.transAxes)

x = var[pair_low[:,0]][adjacent_low]
y = var[pair_low[:,1]][adjacent_low]
use = ~np.isnan(x).values & ~np.isnan(y).values & (x.values >= xmin) & (x.values <= xmax) & (y.values >= ymin) & (y.values <= ymax)
x = x[use]
y = y[use]
xerrmin = errmin[pair_low[:,0]][adjacent_low][use]
xerrmax = errmax[pair_low[:,0]][adjacent_low][use]
yerrmin = errmin[pair_low[:,1]][adjacent_low][use]
yerrmax = errmax[pair_low[:,1]][adjacent_low][use]
sr, sp = spearmanr(x, y)
pr, pp = pearsonr(x, y)
ax.errorbar(x, y, xerr=[xerrmin,xerrmax], yerr=[yerrmin,yerrmax], fmt='o', c='b', ms=msize, label='Field', zorder=3)
ax.text(1.-textpad, textpad+2.*dy*textpad, 'Pearson $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(pr, 2), np.round(np.log10(pp), 2)), color='b', ha='right', va='bottom', transform = ax.transAxes)
ax.text(1.-textpad, textpad+3.*dy*textpad, 'Spearman $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(sr, 2), np.round(np.log10(sp), 2)), color='b', ha='right', va='bottom', transform = ax.transAxes)

x = var[pair_high[:,0]][adjacent_high]
y = var[pair_high[:,1]][adjacent_high]
use = ~np.isnan(x).values & ~np.isnan(y).values & (x.values >= xmin) & (x.values <= xmax) & (y.values >= ymin) & (y.values <= ymax)
x = x[use]
y = y[use]
xerrmin = errmin[pair_high[:,0]][adjacent_high][use]
xerrmax = errmax[pair_high[:,0]][adjacent_high][use]
yerrmin = errmin[pair_high[:,1]][adjacent_high][use]
yerrmax = errmax[pair_high[:,1]][adjacent_high][use]
sr, sp = spearmanr(x, y)
pr, pp = pearsonr(x, y)
ax.errorbar(x, y, xerr=[xerrmin,xerrmax], yerr=[yerrmin,yerrmax], fmt='o', c='r', ms=msize, label='Overdensities', zorder=3)
ax.text(1.-textpad, textpad+4.*dy*textpad, 'Pearson $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(pr, 2), np.round(np.log10(pp), 2)), color='r', ha='right', va='bottom', transform = ax.transAxes)
ax.text(1.-textpad, textpad+5.*dy*textpad, 'Spearman $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(sr, 2), np.round(np.log10(sp), 2)), color='r', ha='right', va='bottom', transform = ax.transAxes)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim((xmin,xmax))
ax.set_ylim((ymin,ymax))
ax.set_xlabel('Inner planet mass [M$_\\oplus$]')
ax.set_ylabel('Outer planet mass [M$_\\oplus$]')
ax.set_aspect(1.)
ax.tick_params(axis='x', which='both', top=True)
ax.tick_params(axis='y', which='both', right=True)
ax.tick_params(which='major', length=6)
ax.tick_params(which='minor', length=3)
ax.legend(loc='upper left', frameon=0)
fig.subplots_adjust(left=edgepad, right=1-edgepad, top=1-edgepad, bottom=edgepad)
figsave = plt.gcf()
figsave.set_size_inches(xs,ys)
plt.savefig('figures/pod_masses.pdf',bbox_inches='tight',dpi=300)


xmin = 0.1
xmax = 2e4
ymin = 0.1
ymax = 2e4
msize = 5
dy = 1.2

fig, ax = plt.subplots(1, 1, figsize=(xs, ys))
ax.plot([xmin,xmax], [ymin,ymax], ':k', alpha=0.3, zorder=0)
var = planets['fpl_bmasse']
errmin = planets['fpl_bmasseerr2']
errmax = planets['fpl_bmasseerr1']

x = var[pair_all[:,0]][adjacent_all]
y = var[pair_all[:,1]][adjacent_all]
err = 0.5*(errmin+errmax)[pair_all[:,0]][adjacent_all]
use = ~np.isnan(x).values & ~np.isnan(y).values & (x.values >= xmin) & (x.values <= xmax) & (y.values >= ymin) & (y.values <= ymax) & (err.values > 0.)
x = x[use]
y = y[use]
xerrmin = errmin[pair_all[:,0]][adjacent_all][use]
xerrmax = errmax[pair_all[:,0]][adjacent_all][use]
yerrmin = errmin[pair_all[:,1]][adjacent_all][use]
yerrmax = errmax[pair_all[:,1]][adjacent_all][use]
if len(x) > 1:
    sr, sp = spearmanr(x, y)
    pr, pp = pearsonr(x, y)
else:
    sr, sp, pr, pp = 0., 0., 0., 0.
ax.errorbar(x, y, xerr=[xerrmin,xerrmax], yerr=[yerrmin,yerrmax], fmt='o', c='k', ms=msize, label='All systems', zorder=3)
ax.text(1.-textpad, textpad, 'Pearson $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(pr, 2), np.round(np.log10(pp), 2)), color='k', ha='right', va='bottom', transform = ax.transAxes)
ax.text(1.-textpad, textpad+1.*dy*textpad, 'Spearman $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(sr, 2), np.round(np.log10(sp), 2)), color='k', ha='right', va='bottom', transform = ax.transAxes)

x = var[pair_low[:,0]][adjacent_low]
y = var[pair_low[:,1]][adjacent_low]
err = 0.5*(errmin+errmax)[pair_low[:,0]][adjacent_low]
use = ~np.isnan(x).values & ~np.isnan(y).values & (x.values >= xmin) & (x.values <= xmax) & (y.values >= ymin) & (y.values <= ymax) & (err.values > 0.)
x = x[use]
y = y[use]
xerrmin = errmin[pair_low[:,0]][adjacent_low][use]
xerrmax = errmax[pair_low[:,0]][adjacent_low][use]
yerrmin = errmin[pair_low[:,1]][adjacent_low][use]
yerrmax = errmax[pair_low[:,1]][adjacent_low][use]
if len(x) > 1:
    sr, sp = spearmanr(x, y)
    pr, pp = pearsonr(x, y)
else:
    sr, sp, pr, pp = 0., 0., 0., 0.
ax.errorbar(x, y, xerr=[xerrmin,xerrmax], yerr=[yerrmin,yerrmax], fmt='o', c='b', ms=msize, label='Field', zorder=3)
ax.text(1.-textpad, textpad+2.*dy*textpad, 'Pearson $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(pr, 2), np.round(np.log10(pp), 2)), color='b', ha='right', va='bottom', transform = ax.transAxes)
ax.text(1.-textpad, textpad+3.*dy*textpad, 'Spearman $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(sr, 2), np.round(np.log10(sp), 2)), color='b', ha='right', va='bottom', transform = ax.transAxes)

x = var[pair_high[:,0]][adjacent_high]
y = var[pair_high[:,1]][adjacent_high]
err = 0.5*(errmin+errmax)[pair_high[:,0]][adjacent_high]
use = ~np.isnan(x).values & ~np.isnan(y).values & (x.values >= xmin) & (x.values <= xmax) & (y.values >= ymin) & (y.values <= ymax) & (err.values > 0.)
x = x[use]
y = y[use]
xerrmin = errmin[pair_high[:,0]][adjacent_high][use]
xerrmax = errmax[pair_high[:,0]][adjacent_high][use]
yerrmin = errmin[pair_high[:,1]][adjacent_high][use]
yerrmax = errmax[pair_high[:,1]][adjacent_high][use]
if len(x) > 1:
    sr, sp = spearmanr(x, y)
    pr, pp = pearsonr(x, y)
else:
    sr, sp, pr, pp = 0., 0., 0., 0.
ax.errorbar(x, y, xerr=[xerrmin,xerrmax], yerr=[yerrmin,yerrmax], fmt='o', c='r', ms=msize, label='Overdensities', zorder=3)
ax.text(1.-textpad, textpad+4.*dy*textpad, 'Pearson $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(pr, 2), np.round(np.log10(pp), 2)), color='r', ha='right', va='bottom', transform = ax.transAxes)
ax.text(1.-textpad, textpad+5.*dy*textpad, 'Spearman $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(sr, 2), np.round(np.log10(sp), 2)), color='r', ha='right', va='bottom', transform = ax.transAxes)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim((xmin,xmax))
ax.set_ylim((ymin,ymax))
ax.set_xlabel('Inner planet mass [M$_\\oplus$]')
ax.set_ylabel('Outer planet mass [M$_\\oplus$]')
ax.set_aspect(1.)
ax.tick_params(axis='x', which='both', top=True)
ax.tick_params(axis='y', which='both', right=True)
ax.tick_params(which='major', length=6)
ax.tick_params(which='minor', length=3)
ax.legend(loc='upper left', frameon=0)
fig.subplots_adjust(left=edgepad, right=1-edgepad, top=1-edgepad, bottom=edgepad)
figsave = plt.gcf()
figsave.set_size_inches(xs,ys)
plt.savefig('figures/pod_masses_true.pdf',bbox_inches='tight',dpi=300)

xmin = 0.1
xmax = 2e1
ymin = 0.1
ymax = 2e1
msize = 5
dy = 1.2

fig, ax = plt.subplots(1, 1, figsize=(xs, ys))
ax.plot([xmin,xmax], [ymin,ymax], ':k', alpha=0.3, zorder=0)
var = planets['fpl_dens']
errmin = planets['fpl_denserr2']
errmax = planets['fpl_denserr1']

x = var[pair_all[:,0]][adjacent_all]
y = var[pair_all[:,1]][adjacent_all]
use = ~np.isnan(x).values & ~np.isnan(y).values & (x.values >= xmin) & (x.values <= xmax) & (y.values >= ymin) & (y.values <= ymax)
x = x[use]
y = y[use]
xerrmin = errmin[pair_all[:,0]][adjacent_all][use]
xerrmax = errmax[pair_all[:,0]][adjacent_all][use]
yerrmin = errmin[pair_all[:,1]][adjacent_all][use]
yerrmax = errmax[pair_all[:,1]][adjacent_all][use]
sr, sp = spearmanr(x, y)
pr, pp = pearsonr(x, y)
ax.errorbar(x, y, xerr=[xerrmin,xerrmax], yerr=[yerrmin,yerrmax], fmt='o', c='k', ms=msize, label='All systems', zorder=3)
ax.text(1.-textpad, textpad, 'Pearson $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(pr, 2), np.round(np.log10(pp), 2)), color='k', ha='right', va='bottom', transform = ax.transAxes)
ax.text(1.-textpad, textpad+1.*dy*textpad, 'Spearman $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(sr, 2), np.round(np.log10(sp), 2)), color='k', ha='right', va='bottom', transform = ax.transAxes)

x = var[pair_low[:,0]][adjacent_low]
y = var[pair_low[:,1]][adjacent_low]
use = ~np.isnan(x).values & ~np.isnan(y).values & (x.values >= xmin) & (x.values <= xmax) & (y.values >= ymin) & (y.values <= ymax)
x = x[use]
y = y[use]
xerrmin = errmin[pair_low[:,0]][adjacent_low][use]
xerrmax = errmax[pair_low[:,0]][adjacent_low][use]
yerrmin = errmin[pair_low[:,1]][adjacent_low][use]
yerrmax = errmax[pair_low[:,1]][adjacent_low][use]
sr, sp = spearmanr(x, y)
pr, pp = pearsonr(x, y)
ax.errorbar(x, y, xerr=[xerrmin,xerrmax], yerr=[yerrmin,yerrmax], fmt='o', c='b', ms=msize, label='Field', zorder=3)
ax.text(1.-textpad, textpad+2.*dy*textpad, 'Pearson $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(pr, 2), np.round(np.log10(pp), 2)), color='b', ha='right', va='bottom', transform = ax.transAxes)
ax.text(1.-textpad, textpad+3.*dy*textpad, 'Spearman $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(sr, 2), np.round(np.log10(sp), 2)), color='b', ha='right', va='bottom', transform = ax.transAxes)

x = var[pair_high[:,0]][adjacent_high]
y = var[pair_high[:,1]][adjacent_high]
use = ~np.isnan(x).values & ~np.isnan(y).values & (x.values >= xmin) & (x.values <= xmax) & (y.values >= ymin) & (y.values <= ymax)
x = x[use]
y = y[use]
xerrmin = errmin[pair_high[:,0]][adjacent_high][use]
xerrmax = errmax[pair_high[:,0]][adjacent_high][use]
yerrmin = errmin[pair_high[:,1]][adjacent_high][use]
yerrmax = errmax[pair_high[:,1]][adjacent_high][use]
sr, sp = spearmanr(x, y)
pr, pp = pearsonr(x, y)
ax.errorbar(x, y, xerr=[xerrmin,xerrmax], yerr=[yerrmin,yerrmax], fmt='o', c='r', ms=msize, label='Overdensities', zorder=3)
ax.text(1.-textpad, textpad+4.*dy*textpad, 'Pearson $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(pr, 2), np.round(np.log10(pp), 2)), color='r', ha='right', va='bottom', transform = ax.transAxes)
ax.text(1.-textpad, textpad+5.*dy*textpad, 'Spearman $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(sr, 2), np.round(np.log10(sp), 2)), color='r', ha='right', va='bottom', transform = ax.transAxes)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim((xmin,xmax))
ax.set_ylim((ymin,ymax))
ax.set_xlabel('Inner planet density [g cm$^{-3}$]')
ax.set_ylabel('Outer planet density [g cm$^{-3}$]')
ax.set_aspect(1.)
ax.tick_params(axis='x', which='both', top=True)
ax.tick_params(axis='y', which='both', right=True)
ax.tick_params(which='major', length=6)
ax.tick_params(which='minor', length=3)
ax.legend(loc='upper left', frameon=0)
fig.subplots_adjust(left=edgepad, right=1-edgepad, top=1-edgepad, bottom=edgepad)
figsave = plt.gcf()
figsave.set_size_inches(xs,ys)
plt.savefig('figures/pod_densities.pdf',bbox_inches='tight',dpi=300)


xmin = 0.1
xmax = 2e1
ymin = 0.1
ymax = 2e1
msize = 5
dy = 1.2

fig, ax = plt.subplots(1, 1, figsize=(xs, ys))
ax.plot([xmin,xmax], [ymin,ymax], ':k', alpha=0.3, zorder=0)
var = planets['fpl_dens']
errmin = planets['fpl_denserr2']
errmax = planets['fpl_denserr1']

x = var[pair_all[:,0]][adjacent_all]
y = var[pair_all[:,1]][adjacent_all]
err = 0.5*(errmin+errmax)[pair_all[:,0]][adjacent_all]
use = ~np.isnan(x).values & ~np.isnan(y).values & (x.values >= xmin) & (x.values <= xmax) & (y.values >= ymin) & (y.values <= ymax) & (err.values > 0.)
x = x[use]
y = y[use]
xerrmin = errmin[pair_all[:,0]][adjacent_all][use]
xerrmax = errmax[pair_all[:,0]][adjacent_all][use]
yerrmin = errmin[pair_all[:,1]][adjacent_all][use]
yerrmax = errmax[pair_all[:,1]][adjacent_all][use]
if len(x) > 1:
    sr, sp = spearmanr(x, y)
    pr, pp = pearsonr(x, y)
else:
    sr, sp, pr, pp = 0., 0., 0., 0.
ax.errorbar(x, y, xerr=[xerrmin,xerrmax], yerr=[yerrmin,yerrmax], fmt='o', c='k', ms=msize, label='All systems', zorder=3)
ax.text(1.-textpad, textpad, 'Pearson $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(pr, 2), np.round(np.log10(pp), 2)), color='k', ha='right', va='bottom', transform = ax.transAxes)
ax.text(1.-textpad, textpad+1.*dy*textpad, 'Spearman $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(sr, 2), np.round(np.log10(sp), 2)), color='k', ha='right', va='bottom', transform = ax.transAxes)

x = var[pair_low[:,0]][adjacent_low]
y = var[pair_low[:,1]][adjacent_low]
err = 0.5*(errmin+errmax)[pair_low[:,0]][adjacent_low]
use = ~np.isnan(x).values & ~np.isnan(y).values & (x.values >= xmin) & (x.values <= xmax) & (y.values >= ymin) & (y.values <= ymax) & (err.values > 0.)
x = x[use]
y = y[use]
xerrmin = errmin[pair_low[:,0]][adjacent_low][use]
xerrmax = errmax[pair_low[:,0]][adjacent_low][use]
yerrmin = errmin[pair_low[:,1]][adjacent_low][use]
yerrmax = errmax[pair_low[:,1]][adjacent_low][use]
if len(x) > 1:
    sr, sp = spearmanr(x, y)
    pr, pp = pearsonr(x, y)
else:
    sr, sp, pr, pp = 0., 0., 0., 0.
ax.errorbar(x, y, xerr=[xerrmin,xerrmax], yerr=[yerrmin,yerrmax], fmt='o', c='b', ms=msize, label='Field', zorder=3)
ax.text(1.-textpad, textpad+2.*dy*textpad, 'Pearson $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(pr, 2), np.round(np.log10(pp), 2)), color='b', ha='right', va='bottom', transform = ax.transAxes)
ax.text(1.-textpad, textpad+3.*dy*textpad, 'Spearman $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(sr, 2), np.round(np.log10(sp), 2)), color='b', ha='right', va='bottom', transform = ax.transAxes)

x = var[pair_high[:,0]][adjacent_high]
y = var[pair_high[:,1]][adjacent_high]
err = 0.5*(errmin+errmax)[pair_high[:,0]][adjacent_high]
use = ~np.isnan(x).values & ~np.isnan(y).values & (x.values >= xmin) & (x.values <= xmax) & (y.values >= ymin) & (y.values <= ymax) & (err.values > 0.)
x = x[use]
y = y[use]
xerrmin = errmin[pair_high[:,0]][adjacent_high][use]
xerrmax = errmax[pair_high[:,0]][adjacent_high][use]
yerrmin = errmin[pair_high[:,1]][adjacent_high][use]
yerrmax = errmax[pair_high[:,1]][adjacent_high][use]
if len(x) > 1:
    sr, sp = spearmanr(x, y)
    pr, pp = pearsonr(x, y)
else:
    sr, sp, pr, pp = 0., 0., 0., 0.
ax.errorbar(x, y, xerr=[xerrmin,xerrmax], yerr=[yerrmin,yerrmax], fmt='o', c='r', ms=msize, label='Overdensities', zorder=3)
ax.text(1.-textpad, textpad+4.*dy*textpad, 'Pearson $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(pr, 2), np.round(np.log10(pp), 2)), color='r', ha='right', va='bottom', transform = ax.transAxes)
ax.text(1.-textpad, textpad+5.*dy*textpad, 'Spearman $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(sr, 2), np.round(np.log10(sp), 2)), color='r', ha='right', va='bottom', transform = ax.transAxes)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim((xmin,xmax))
ax.set_ylim((ymin,ymax))
ax.set_xlabel('Inner planet density [g cm$^{-3}$]')
ax.set_ylabel('Outer planet density [g cm$^{-3}$]')
ax.set_aspect(1.)
ax.tick_params(axis='x', which='both', top=True)
ax.tick_params(axis='y', which='both', right=True)
ax.tick_params(which='major', length=6)
ax.tick_params(which='minor', length=3)
ax.legend(loc='upper left', frameon=0)
fig.subplots_adjust(left=edgepad, right=1-edgepad, top=1-edgepad, bottom=edgepad)
figsave = plt.gcf()
figsave.set_size_inches(xs,ys)
plt.savefig('figures/pod_densities_true.pdf',bbox_inches='tight',dpi=300)

xmin = 1e-2
xmax = 1
ymin = 1e-2
ymax = 1
msize = 5
dy = 1.2

fig, ax = plt.subplots(1, 1, figsize=(xs, ys))
ax.plot([xmin,xmax], [ymin,ymax], ':k', alpha=0.3, zorder=0)
var = planets['fpl_eccen']
errmin = planets['fpl_eccenerr2']
errmax = planets['fpl_eccenerr1']

x = var[pair_all[:,0]][adjacent_all]
y = var[pair_all[:,1]][adjacent_all]
use = ~np.isnan(x).values & ~np.isnan(y).values & (x.values >= xmin) & (x.values <= xmax) & (y.values >= ymin) & (y.values <= ymax)
x = x[use]
y = y[use]
xerrmin = errmin[pair_all[:,0]][adjacent_all][use]
xerrmax = errmax[pair_all[:,0]][adjacent_all][use]
yerrmin = errmin[pair_all[:,1]][adjacent_all][use]
yerrmax = errmax[pair_all[:,1]][adjacent_all][use]
sr, sp = spearmanr(x, y)
pr, pp = pearsonr(x, y)
ax.errorbar(x, y, xerr=[xerrmin,xerrmax], yerr=[yerrmin,yerrmax], fmt='o', c='k', ms=msize, label='All systems', zorder=3)
ax.text(1.-textpad, textpad, 'Pearson $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(pr, 2), np.round(np.log10(pp), 2)), color='k', ha='right', va='bottom', transform = ax.transAxes)
ax.text(1.-textpad, textpad+1.*dy*textpad, 'Spearman $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(sr, 2), np.round(np.log10(sp), 2)), color='k', ha='right', va='bottom', transform = ax.transAxes)

x = var[pair_low[:,0]][adjacent_low]
y = var[pair_low[:,1]][adjacent_low]
use = ~np.isnan(x).values & ~np.isnan(y).values & (x.values >= xmin) & (x.values <= xmax) & (y.values >= ymin) & (y.values <= ymax)
x = x[use]
y = y[use]
xerrmin = errmin[pair_low[:,0]][adjacent_low][use]
xerrmax = errmax[pair_low[:,0]][adjacent_low][use]
yerrmin = errmin[pair_low[:,1]][adjacent_low][use]
yerrmax = errmax[pair_low[:,1]][adjacent_low][use]
sr, sp = spearmanr(x, y)
pr, pp = pearsonr(x, y)
ax.errorbar(x, y, xerr=[xerrmin,xerrmax], yerr=[yerrmin,yerrmax], fmt='o', c='b', ms=msize, label='Field', zorder=3)
ax.text(1.-textpad, textpad+2.*dy*textpad, 'Pearson $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(pr, 2), np.round(np.log10(pp), 2)), color='b', ha='right', va='bottom', transform = ax.transAxes)
ax.text(1.-textpad, textpad+3.*dy*textpad, 'Spearman $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(sr, 2), np.round(np.log10(sp), 2)), color='b', ha='right', va='bottom', transform = ax.transAxes)

x = var[pair_high[:,0]][adjacent_high]
y = var[pair_high[:,1]][adjacent_high]
use = ~np.isnan(x).values & ~np.isnan(y).values & (x.values >= xmin) & (x.values <= xmax) & (y.values >= ymin) & (y.values <= ymax)
x = x[use]
y = y[use]
xerrmin = errmin[pair_high[:,0]][adjacent_high][use]
xerrmax = errmax[pair_high[:,0]][adjacent_high][use]
yerrmin = errmin[pair_high[:,1]][adjacent_high][use]
yerrmax = errmax[pair_high[:,1]][adjacent_high][use]
sr, sp = spearmanr(x, y)
pr, pp = pearsonr(x, y)
ax.errorbar(x, y, xerr=[xerrmin,xerrmax], yerr=[yerrmin,yerrmax], fmt='o', c='r', ms=msize, label='Overdensities', zorder=3)
ax.text(1.-textpad, textpad+4.*dy*textpad, 'Pearson $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(pr, 2), np.round(np.log10(pp), 2)), color='r', ha='right', va='bottom', transform = ax.transAxes)
ax.text(1.-textpad, textpad+5.*dy*textpad, 'Spearman $\\{r,\\log{(p)}\\}=\\{%.2f,%.2f\\}$' % (np.round(sr, 2), np.round(np.log10(sp), 2)), color='r', ha='right', va='bottom', transform = ax.transAxes)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim((xmin,xmax))
ax.set_ylim((ymin,ymax))
ax.set_xlabel('Inner planet eccentricity')
ax.set_ylabel('Outer planet eccentricity')
ax.set_aspect(1.)
ax.tick_params(axis='x', which='both', top=True)
ax.tick_params(axis='y', which='both', right=True)
ax.tick_params(which='major', length=6)
ax.tick_params(which='minor', length=3)
ax.legend(loc='upper left', frameon=0)
fig.subplots_adjust(left=edgepad, right=1-edgepad, top=1-edgepad, bottom=edgepad)
figsave = plt.gcf()
figsave.set_size_inches(xs,ys)
plt.savefig('figures/pod_eccentricities.pdf',bbox_inches='tight',dpi=300)


# quit()










zeta1_all = 3.*(1./(pout_pin_all-1.)-np.round(1./(pout_pin_all-1.)))
zeta1_low = 3.*(1./(pout_pin_low-1.)-np.round(1./(pout_pin_low-1.)))
zeta1_high = 3.*(1./(pout_pin_high-1.)-np.round(1./(pout_pin_high-1.)))
zeta2_all = 3.*(2./(pout_pin_all-1.)-np.round(2./(pout_pin_all-1.)))
zeta2_low = 3.*(2./(pout_pin_low-1.)-np.round(2./(pout_pin_low-1.)))
zeta2_high = 3.*(2./(pout_pin_high-1.)-np.round(2./(pout_pin_high-1.)))
zeta1_ad_all = 3.*(1./(pout_pin_all[adjacent_all]-1.)-np.round(1./(pout_pin_all[adjacent_all]-1.)))
zeta1_ad_low = 3.*(1./(pout_pin_low[adjacent_low]-1.)-np.round(1./(pout_pin_low[adjacent_low]-1.)))
zeta1_ad_high = 3.*(1./(pout_pin_high[adjacent_high]-1.)-np.round(1./(pout_pin_high[adjacent_high]-1.)))
zeta2_ad_all = 3.*(2./(pout_pin_all[adjacent_all]-1.)-np.round(2./(pout_pin_all[adjacent_all]-1.)))
zeta2_ad_low = 3.*(2./(pout_pin_low[adjacent_low]-1.)-np.round(2./(pout_pin_low[adjacent_low]-1.)))
zeta2_ad_high = 3.*(2./(pout_pin_high[adjacent_high]-1.)-np.round(2./(pout_pin_high[adjacent_high]-1.)))
zeta1_all = zeta1_all[np.abs(zeta1_all) <= 1.]
zeta1_low = zeta1_low[np.abs(zeta1_low) <= 1.]
zeta1_high = zeta1_high[np.abs(zeta1_high) <= 1.]
zeta2_all = zeta2_all[np.abs(zeta2_all) <= 1.]
zeta2_low = zeta2_low[np.abs(zeta2_low) <= 1.]
zeta2_high = zeta2_high[np.abs(zeta2_high) <= 1.]
zeta1_ad_all = zeta1_ad_all[np.abs(zeta1_ad_all) <= 1.]
zeta1_ad_low = zeta1_ad_low[np.abs(zeta1_ad_low) <= 1.]
zeta1_ad_high = zeta1_ad_high[np.abs(zeta1_ad_high) <= 1.]
zeta2_ad_all = zeta2_ad_all[np.abs(zeta2_ad_all) <= 1.]
zeta2_ad_low = zeta2_ad_low[np.abs(zeta2_ad_low) <= 1.]
zeta2_ad_high = zeta2_ad_high[np.abs(zeta2_ad_high) <= 1.]

xs = 8
ys = 6

xmin = -1
xmax = 1
ymin = 0
ymax = 25
nbins = 20
edges = np.linspace(xmin, xmax, nbins+1)
fig, ax = plt.subplots(1, 1, figsize=(xs, ys))
ax.hist(zeta1_all, color='k', bins=edges, histtype='step', label='All systems', zorder=3)
ax.hist(zeta1_low, color='b', bins=edges, alpha=0.5, label='Field', zorder=3)
ax.hist(zeta1_high, color='r', bins=edges, alpha=0.5, label='Overdensities', zorder=3)
ax.text(textpad, 1.-textpad, 'All pairs', ha='left', va='top', transform = ax.transAxes)
ax.set_xlim((xmin,xmax))
ax.set_ylim((ymin,ymax))
ax.set_xlabel('$\\zeta_1$')
ax.set_ylabel('Number')
ax.tick_params(axis='x', which='both', top=True)
ax.tick_params(axis='y', which='both', right=True)
ax.tick_params(which='major', length=6)
ax.tick_params(which='minor', length=3)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.legend(loc=1, frameon=0)
fig.subplots_adjust(left=edgepad, right=1-edgepad, top=1-edgepad, bottom=edgepad)
figsave = plt.gcf()
figsave.set_size_inches(xs,ys)
plt.savefig('figures/resonant_offset1.pdf',bbox_inches='tight',dpi=300)

fig, ax = plt.subplots(1, 1, figsize=(xs, ys))
ax.hist(zeta2_all, color='k', bins=edges, histtype='step', label='All systems', zorder=3)
ax.hist(zeta2_low, color='b', bins=edges, alpha=0.5, label='Field', zorder=3)
ax.hist(zeta2_high, color='r', bins=edges, alpha=0.5, label='Overdensities', zorder=3)
ax.text(textpad, 1.-textpad, 'All pairs', ha='left', va='top', transform = ax.transAxes)
ax.set_xlim((xmin,xmax))
ax.set_ylim((ymin,ymax))
ax.set_xlabel('$\\zeta_2$')
ax.set_ylabel('Number')
ax.tick_params(axis='x', which='both', top=True)
ax.tick_params(axis='y', which='both', right=True)
ax.tick_params(which='major', length=6)
ax.tick_params(which='minor', length=3)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.legend(loc=1, frameon=0)
fig.subplots_adjust(left=edgepad, right=1-edgepad, top=1-edgepad, bottom=edgepad)
figsave = plt.gcf()
figsave.set_size_inches(xs,ys)
plt.savefig('figures/resonant_offset2.pdf',bbox_inches='tight',dpi=300)

xmin = -1
xmax = 1
ymin = 0
ymax = 25
nbins = 20
edges = np.linspace(xmin, xmax, nbins+1)
fig, ax = plt.subplots(1, 1, figsize=(xs, ys))
ax.hist(zeta1_ad_all, color='k', bins=edges, histtype='step', label='All systems', zorder=3)
ax.hist(zeta1_ad_low, color='b', bins=edges, alpha=0.5, label='Field', zorder=3)
ax.hist(zeta1_ad_high, color='r', bins=edges, alpha=0.5, label='Overdensities', zorder=3)
ax.text(textpad, 1.-textpad, 'Adjacent pairs only', ha='left', va='top', transform = ax.transAxes)
ax.set_xlim((xmin,xmax))
ax.set_ylim((ymin,ymax))
ax.set_xlabel('$\\zeta_1$')
ax.set_ylabel('Number')
ax.tick_params(axis='x', which='both', top=True)
ax.tick_params(axis='y', which='both', right=True)
ax.tick_params(which='major', length=6)
ax.tick_params(which='minor', length=3)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.legend(loc=1, frameon=0)
fig.subplots_adjust(left=edgepad, right=1-edgepad, top=1-edgepad, bottom=edgepad)
figsave = plt.gcf()
figsave.set_size_inches(xs,ys)
plt.savefig('figures/resonant_offset_adjacent1.pdf',bbox_inches='tight',dpi=300)

fig, ax = plt.subplots(1, 1, figsize=(xs, ys))
ax.hist(zeta2_ad_all, color='k', bins=edges, histtype='step', label='All systems', zorder=3)
ax.hist(zeta2_ad_low, color='b', bins=edges, alpha=0.5, label='Field', zorder=3)
ax.hist(zeta2_ad_high, color='r', bins=edges, alpha=0.5, label='Overdensities', zorder=3)
ax.text(textpad, 1.-textpad, 'Adjacent pairs only', ha='left', va='top', transform = ax.transAxes)
ax.set_xlim((xmin,xmax))
ax.set_ylim((ymin,ymax))
ax.set_xlabel('$\\zeta_2$')
ax.set_ylabel('Number')
ax.tick_params(axis='x', which='both', top=True)
ax.tick_params(axis='y', which='both', right=True)
ax.tick_params(which='major', length=6)
ax.tick_params(which='minor', length=3)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.legend(loc=1, frameon=0)
fig.subplots_adjust(left=edgepad, right=1-edgepad, top=1-edgepad, bottom=edgepad)
figsave = plt.gcf()
figsave.set_size_inches(xs,ys)
plt.savefig('figures/resonant_offset_adjacent2.pdf',bbox_inches='tight',dpi=300)

nzeta = 10000
zetamin = 0.
zetamax = 1.
zeta = np.linspace(zetamin, zetamax, nzeta)
zeta1_cdf_all = np.zeros_like(zeta)
zeta1_cdf_low = np.zeros_like(zeta)
zeta1_cdf_high = np.zeros_like(zeta)
zeta2_cdf_all = np.zeros_like(zeta)
zeta2_cdf_low = np.zeros_like(zeta)
zeta2_cdf_high = np.zeros_like(zeta)
zeta1_ad_cdf_all = np.zeros_like(zeta)
zeta1_ad_cdf_low = np.zeros_like(zeta)
zeta1_ad_cdf_high = np.zeros_like(zeta)
zeta2_ad_cdf_all = np.zeros_like(zeta)
zeta2_ad_cdf_low = np.zeros_like(zeta)
zeta2_ad_cdf_high = np.zeros_like(zeta)
for i in range(nzeta):
    zeta1_cdf_all[i] = np.sum(np.abs(zeta1_all) <= zeta[i])/np.size(zeta1_all)
    zeta1_cdf_low[i] = np.sum(np.abs(zeta1_low) <= zeta[i])/np.size(zeta1_low)
    zeta1_cdf_high[i] = np.sum(np.abs(zeta1_high) <= zeta[i])/np.size(zeta1_high)
    zeta2_cdf_all[i] = np.sum(np.abs(zeta2_all) <= zeta[i])/np.size(zeta2_all)
    zeta2_cdf_low[i] = np.sum(np.abs(zeta2_low) <= zeta[i])/np.size(zeta2_low)
    zeta2_cdf_high[i] = np.sum(np.abs(zeta2_high) <= zeta[i])/np.size(zeta2_high)
    zeta1_ad_cdf_all[i] = np.sum(np.abs(zeta1_ad_all) <= zeta[i])/np.size(zeta1_ad_all)
    zeta1_ad_cdf_low[i] = np.sum(np.abs(zeta1_ad_low) <= zeta[i])/np.size(zeta1_ad_low)
    zeta1_ad_cdf_high[i] = np.sum(np.abs(zeta1_ad_high) <= zeta[i])/np.size(zeta1_ad_high)
    zeta2_ad_cdf_all[i] = np.sum(np.abs(zeta2_ad_all) <= zeta[i])/np.size(zeta2_ad_all)
    zeta2_ad_cdf_low[i] = np.sum(np.abs(zeta2_ad_low) <= zeta[i])/np.size(zeta2_ad_low)
    zeta2_ad_cdf_high[i] = np.sum(np.abs(zeta2_ad_high) <= zeta[i])/np.size(zeta2_ad_high)

xmin = zetamin
xmax = zetamax
ymin = -0.05
ymax = 1.05

fig, ax = plt.subplots(1, 1, figsize=(xs, ys))
ax.plot(zeta, zeta1_cdf_all, '-k', label='All systems', zorder=3)
ax.plot(zeta, zeta1_cdf_low, '-b', label='Field', zorder=3)
ax.plot(zeta, zeta1_cdf_high, '-r', label='Overdensities', zorder=3)
ax.plot([0,1], [0,1], ':k', alpha=0.3, zorder=0)
ax.text(textpad, 1.-textpad, 'All pairs', ha='left', va='top', transform = ax.transAxes)
ax.text(textpad, 1.-3.*textpad, '$p_{\\rm KS}=%.3f$'%(np.round(ks(zeta1_low, zeta1_high)[1],3)), ha='left', va='top', transform = ax.transAxes)
ax.set_xlim((xmin,xmax))
ax.set_ylim((ymin,ymax))
ax.set_xlabel('$\\zeta_1$')
ax.set_ylabel('CDF')
ax.tick_params(axis='x', which='both', top=True)
ax.tick_params(axis='y', which='both', right=True)
ax.tick_params(which='major', length=6)
ax.tick_params(which='minor', length=3)
ax.legend(loc='lower right', frameon=0)
fig.subplots_adjust(left=edgepad, right=1-edgepad, top=1-edgepad, bottom=edgepad)
figsave = plt.gcf()
figsave.set_size_inches(xs,ys)
plt.savefig('figures/zeta1_cdf.pdf',bbox_inches='tight',dpi=300)

fig, ax = plt.subplots(1, 1, figsize=(xs, ys))
ax.plot(zeta, zeta2_cdf_all, '-k', label='All systems', zorder=3)
ax.plot(zeta, zeta2_cdf_low, '-b', label='Field', zorder=3)
ax.plot(zeta, zeta2_cdf_high, '-r', label='Overdensities', zorder=3)
ax.plot([0,1], [0,1], ':k', alpha=0.3, zorder=0)
ax.text(textpad, 1.-textpad, 'All pairs', ha='left', va='top', transform = ax.transAxes)
ax.text(textpad, 1.-3.*textpad, '$p_{\\rm KS}=%.3f$'%(np.round(ks(zeta2_low, zeta2_high)[1],3)), ha='left', va='top', transform = ax.transAxes)
ax.set_xlim((xmin,xmax))
ax.set_ylim((ymin,ymax))
ax.set_xlabel('$\\zeta_2$')
ax.set_ylabel('CDF')
ax.tick_params(axis='x', which='both', top=True)
ax.tick_params(axis='y', which='both', right=True)
ax.tick_params(which='major', length=6)
ax.tick_params(which='minor', length=3)
ax.legend(loc='lower right', frameon=0)
fig.subplots_adjust(left=edgepad, right=1-edgepad, top=1-edgepad, bottom=edgepad)
figsave = plt.gcf()
figsave.set_size_inches(xs,ys)
plt.savefig('figures/zeta2_cdf.pdf',bbox_inches='tight',dpi=300)

fig, ax = plt.subplots(1, 1, figsize=(xs, ys))
ax.plot(zeta, zeta1_ad_cdf_all, '-k', label='All systems', zorder=3)
ax.plot(zeta, zeta1_ad_cdf_low, '-b', label='Field', zorder=3)
ax.plot(zeta, zeta1_ad_cdf_high, '-r', label='Overdensities', zorder=3)
ax.plot([0,1], [0,1], ':k', alpha=0.3, zorder=0)
ax.text(textpad, 1.-textpad, 'Adjacent pairs only', ha='left', va='top', transform = ax.transAxes)
ax.text(textpad, 1.-3.*textpad, '$p_{\\rm KS}=%.3f$'%(np.round(ks(zeta1_ad_low, zeta1_ad_high)[1],3)), ha='left', va='top', transform = ax.transAxes)
ax.set_xlim((xmin,xmax))
ax.set_ylim((ymin,ymax))
ax.set_xlabel('$\\zeta_1$')
ax.set_ylabel('CDF')
ax.tick_params(axis='x', which='both', top=True)
ax.tick_params(axis='y', which='both', right=True)
ax.tick_params(which='major', length=6)
ax.tick_params(which='minor', length=3)
ax.legend(loc='lower right', frameon=0)
fig.subplots_adjust(left=edgepad, right=1-edgepad, top=1-edgepad, bottom=edgepad)
figsave = plt.gcf()
figsave.set_size_inches(xs,ys)
plt.savefig('figures/zeta1_ad_cdf.pdf',bbox_inches='tight',dpi=300)

fig, ax = plt.subplots(1, 1, figsize=(xs, ys))
ax.plot(zeta, zeta2_ad_cdf_all, '-k', label='All systems', zorder=3)
ax.plot(zeta, zeta2_ad_cdf_low, '-b', label='Field', zorder=3)
ax.plot(zeta, zeta2_ad_cdf_high, '-r', label='Overdensities', zorder=3)
ax.plot([0,1], [0,1], ':k', alpha=0.3, zorder=0)
ax.text(textpad, 1.-textpad, 'Adjacent pairs only', ha='left', va='top', transform = ax.transAxes)
ax.text(textpad, 1.-3.*textpad, '$p_{\\rm KS}=%.3f$'%(np.round(ks(zeta2_ad_low, zeta2_ad_high)[1],3)), ha='left', va='top', transform = ax.transAxes)
ax.set_xlim((xmin,xmax))
ax.set_ylim((ymin,ymax))
ax.set_xlabel('$\\zeta_2$')
ax.set_ylabel('CDF')
ax.tick_params(axis='x', which='both', top=True)
ax.tick_params(axis='y', which='both', right=True)
ax.tick_params(which='major', length=6)
ax.tick_params(which='minor', length=3)
ax.legend(loc='lower right', frameon=0)
fig.subplots_adjust(left=edgepad, right=1-edgepad, top=1-edgepad, bottom=edgepad)
figsave = plt.gcf()
figsave.set_size_inches(xs,ys)
plt.savefig('figures/zeta2_ad_cdf.pdf',bbox_inches='tight',dpi=300)



plt.close('all')







