#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 18:23:13 2019

@author: jameswilliams
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:30:40 2019

@author: jameswilliams
"""

import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
##############################################################################
# 1. Set working directory
# 2. Assign parameters with fixed values
# 3. Read in data file
# 4. Extract age, depth etc. information 
# 5. Remove sponges, radiolarians, silicoflagellates and Dactyliosolen girdle bands 
# 6. Calculate relative abundance i.e. percentage of diatom species
# 7. Select sub-set based on >=1% abundance 
# 8. Select sub-set of raw counts based on >=1% abundance 
# 9. Plot CRS abundance against depth (cmbsf) and age (cal. yrs BP)
# 9. Select sub-set of raw counts based on >=2% abundance
# 10. Calculate ADA (Warnock and Scherer, 2015)
# 11. Check if any slides need to be counted further
# 12. Calculate species statistical weights
# 13. PCA analysis
##############################################################################
# Section 1. Set working directory 
os.chdir('/Users/jameswilliams/Desktop/Work/Cores/TC46_GC47_Anvers_Shelf/Quantitative_diatom_counts')
##############################################################################
# Section 2. Assign parameters with fixed values
FOV_area = 0.020114286 ##mm^2
beaker_area = 10391.07143 ##mm^2 	

Threshold = 2
##############################################################################
# Section 3. Read in data file
filename_1 = 'TC46GC47_CRS.csv'
filename_2 = 'TC46GC47_CRS_free.csv'
CRS_data = pd.read_csv(filename_1,sep = ',')
CRS_free_data = pd.read_csv(filename_2,sep = ',')
print(CRS_data.shape, 'filename1')
print(CRS_free_data.shape, 'filename2')

core = CRS_data.iloc[:,0] 
cmbsf_mid = CRS_data.iloc[:,1] #cm 
age_best = CRS_data.iloc[:,2] #(calender years BP)
mass = CRS_data.iloc[:,3] #(g)

FieldsOfView_CRS = CRS_data.iloc[:,4]  
FOV_Chaet = CRS_data.iloc[:,4]  
FOV_Chaet_free = CRS_free_data.iloc[:,4] 

CRS_count_raw = CRS_data.iloc[:,5:]
CRS_free_count_raw = CRS_free_data.iloc[:,5:] 
print(CRS_count_raw.shape, 'CRS_count')
print(CRS_free_count_raw.shape, 'CRS_free_count')
##############################################################################
CRS_count_raw = CRS_count_raw.drop('Acnanthes_spp._(R)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('Acnanthes_spp._(S)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('Acanthes_bongrainii_(R)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('Acanthes_bongrainii_(S)', axis=1,)

CRS_count_raw = CRS_count_raw.drop('Cocconeis_costata_var._antarctica_(R)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('Cocconeis_costata_var._antarctica_(S)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('C._californica_(R)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('C._californica_(S)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('C._californica_var._kerg_(R)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('C._californica_var._kerg_(S)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('C._fasciolata_(R)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('C._fasciolata_(S)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('C._imperatix_(R)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('C._imperatix_(S)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('C._melochorides_(R)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('C._melochorides_(S)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('C._scutellum_(R)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('C._scutellum_(S)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('C._schuttii_(R)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('C._schuttii_(S)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('C._spina_christa_(R)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('C._spina_christa_(S)', axis=1,)

CRS_count_raw = CRS_count_raw.drop('Chaetoceros_bulbosus', axis=1,)
CRS_count_raw = CRS_count_raw.drop('Chaetoceros_dichaeta', axis=1,)
CRS_count_raw = CRS_count_raw.drop('Chaetoceros_criophilius', axis=1,)
CRS_count_raw = CRS_count_raw.drop('Chaetoceros_-_Hyalochaete', axis=1,)
CRS_count_raw = CRS_count_raw.drop('Chaetoceros_-_Phaeoceros', axis=1,)

CRS_count_raw = CRS_count_raw.drop('Dactyliosolen_girdle_bands', axis=1,)

CRS_count_raw = CRS_count_raw.drop('Eucampia_sp._(valve_view)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('Eucampia_var._antarctica_(asymm):_summer', axis=1,)
CRS_count_raw = CRS_count_raw.drop('Eucampia_var._antarctica_(asymm):_winter', axis=1,)
CRS_count_raw = CRS_count_raw.drop('Eucampia_var._antarctica_(asymm):_winter_intercalary', axis=1,)
CRS_count_raw = CRS_count_raw.drop('Eucampia_var._antarctica_(asymm):_winter_terminal', axis=1,)
CRS_count_raw = CRS_count_raw.drop('Eucampia_var._recta_(symm):_summer', axis=1,)
CRS_count_raw = CRS_count_raw.drop('Eucampia_var._recta_(symm):_winter', axis=1,)
CRS_count_raw = CRS_count_raw.drop('Eucampia_var._recta_(symm):_winter_intercalary', axis=1,)
CRS_count_raw = CRS_count_raw.drop('Eucampia_var._recta_(symm):_winter_terminal', axis=1,)

CRS_count_raw = CRS_count_raw.drop('Fragilariopsis_colonies_<10micron', axis=1,)
CRS_count_raw = CRS_count_raw.drop('Fragilariopsis_cylindrus_(>2.4__m_wide)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('Fragilariopsis_kerguelensis_(elliptical)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('Fragilariopsis_kerguelensis_(lanceolate)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('Fragilariopsis_kerguelensis_(intermediate)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('Fragilariopsis_nana_(<2.4__m_wide)', axis=1,)

CRS_count_raw = CRS_count_raw.drop('Navicula_directa', axis=1,)
CRS_count_raw = CRS_count_raw.drop('Navicula_directa_(cn)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('Navicula_glacei', axis=1,)
CRS_count_raw = CRS_count_raw.drop('Navicula_glacei_(cn)', axis=1,)

CRS_count_raw = CRS_count_raw.drop('Proboscia_inermis_(spring)', axis=1,)	
CRS_count_raw = CRS_count_raw.drop('Proboscia_truncata_(spring)', axis=1,)	
CRS_count_raw = CRS_count_raw.drop('Proboscia_spp._(winter)', axis=1,)	

CRS_count_raw = CRS_count_raw.drop('Shionodiscus_gracilis_var._expecta', axis=1,)	
CRS_count_raw = CRS_count_raw.drop('Shionodiscus_gracilis_var._gracilis', axis=1,)

CRS_count_raw = CRS_count_raw.drop('Thalassiosira_antarctica_RS_(T1)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('Thalassiosira_antarctica_RS_(T2)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('Thalassiosira_lentiginosa_(fasc.)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('Thalassiosira_lentiginosa_(intermediate)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('Thalassiosira_lentiginosa_(random_-_no_straight_lines_at_all)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('Thalassiosira_scotia_(primary)', axis=1,)
CRS_count_raw = CRS_count_raw.drop('Thalassiosira_scotia_(secondary)', axis=1,)
##############################################################################
##############################################################################
CRS_free_count_raw = CRS_free_count_raw.drop('Acnanthes_spp._(R)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('Acnanthes_spp._(S)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('Acanthes_bongrainii_(R)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('Acanthes_bongrainii_(S)', axis=1,)

CRS_free_count_raw = CRS_free_count_raw.drop('Cocconeis_costata_var._antarctica_(R)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('Cocconeis_costata_var._antarctica_(S)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('C._californica_(R)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('C._californica_(S)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('C._californica_var._kerg_(R)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('C._californica_var._kerg_(S)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('C._fasciolata_(R)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('C._fasciolata_(S)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('C._imperatix_(R)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('C._imperatix_(S)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('C._melochorides_(R)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('C._melochorides_(S)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('C._scutellum_(R)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('C._scutellum_(S)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('C._schuttii_(R)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('C._schuttii_(S)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('C._spina_christa_(R)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('C._spina_christa_(S)', axis=1,)

CRS_free_count_raw = CRS_free_count_raw.drop('Chaetoceros_bulbosus', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('Chaetoceros_dichaeta', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('Chaetoceros_criophilius', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('Chaetoceros_-_Hyalochaete', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('Chaetoceros_-_Phaeoceros', axis=1,)

CRS_free_count_raw = CRS_free_count_raw.drop('Dactyliosolen_girdle_bands', axis=1,)

CRS_free_count_raw = CRS_free_count_raw.drop('Eucampia_sp._(valve_view)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('Eucampia_var._antarctica_(asymm):_summer', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('Eucampia_var._antarctica_(asymm):_winter', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('Eucampia_var._antarctica_(asymm):_winter_intercalary', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('Eucampia_var._antarctica_(asymm):_winter_terminal', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('Eucampia_var._recta_(symm):_summer', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('Eucampia_var._recta_(symm):_winter', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('Eucampia_var._recta_(symm):_winter_intercalary', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('Eucampia_var._recta_(symm):_winter_terminal', axis=1,)

CRS_free_count_raw = CRS_free_count_raw.drop('Fragilariopsis_colonies_<10micron', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('Fragilariopsis_cylindrus_(>2.4__m_wide)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('Fragilariopsis_kerguelensis_(elliptical)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('Fragilariopsis_kerguelensis_(lanceolate)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('Fragilariopsis_kerguelensis_(intermediate)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('Fragilariopsis_nana_(<2.4__m_wide)', axis=1,)

CRS_free_count_raw = CRS_free_count_raw.drop('Navicula_directa', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('Navicula_directa_(cn)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('Navicula_glacei', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('Navicula_glacei_(cn)', axis=1,)

CRS_free_count_raw = CRS_free_count_raw.drop('Proboscia_inermis_(spring)', axis=1,)	
CRS_free_count_raw = CRS_free_count_raw.drop('Proboscia_truncata_(spring)', axis=1,)	
CRS_free_count_raw = CRS_free_count_raw.drop('Proboscia_spp._(winter)', axis=1,)	

CRS_free_count_raw = CRS_free_count_raw.drop('Shionodiscus_gracilis_var._expecta', axis=1,)	
CRS_free_count_raw = CRS_free_count_raw.drop('Shionodiscus_gracilis_var._gracilis', axis=1,)

CRS_free_count_raw = CRS_free_count_raw.drop('Thalassiosira_antarctica_RS_(T1)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('Thalassiosira_antarctica_RS_(T2)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('Thalassiosira_lentiginosa_(fasc.)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('Thalassiosira_lentiginosa_(intermediate)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('Thalassiosira_lentiginosa_(random_-_no_straight_lines_at_all)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('Thalassiosira_scotia_(primary)', axis=1,)
CRS_free_count_raw = CRS_free_count_raw.drop('Thalassiosira_scotia_(secondary)', axis=1,)
##############################################################################
##############################################################################
print(CRS_count_raw.shape, 'CRS_count_after_editing')
print(CRS_free_count_raw.shape, 'CRS_free_count_after_editing')
##############################################################################
#add data sets together to get total counted
total_count_with_CRS = CRS_count_raw.add(CRS_free_count_raw, axis='columns')
#total_count_with_CRS.to_csv ('TC46GC47_total_count_with_CRS')
print(total_count_with_CRS.shape, 'total_CRS_count')

total_count_CRS_free = total_count_with_CRS.drop('Chaetoceros_resting_spore', axis = 1)
#total_count_CRS_free.to_csv('CRS_free_total_count')
print(total_count_CRS_free.shape, 'total_CRS_free_count_')
#should be -1 difference between 'total_counts_raw' and 'CRS_free_total_raw_counts'

total_count_CRS_free.to_csv ('total_count_CRS_free')

#caclulate the total number of valves counted
#use this for m/vgds as the CRS count
total_valves_counted_CRS = CRS_count_raw.sum(axis = 1)
#use for species m/vgds
total_valves_counted_CRS_free = total_count_CRS_free.sum(axis = 1)
##############################################################################
##############################################################################
# Section 6. Calculate relative abundance i.e. percentage of diatom species
CRS_free_species_rel_abundance = total_count_CRS_free.apply(lambda c: c / c.sum() * 100, axis=1)
totals_check = CRS_free_species_rel_abundance.sum(axis=1) #check that numbers add up 100%
#print(totals_check, 'total_%')

#transpose data - easier to work in rows for this part
transposed_species_abundance = CRS_free_species_rel_abundance.transpose()
#calculate max values of each species % and add column to df
max_values = transposed_species_abundance.apply(max, axis=1)
transposed_species_abundance['max'] = max_values
CRS_free_species_rel_abundance = transposed_species_abundance.transpose()
##############################################################################
# Section 7. Select sub-set based on >=2% abundance 
species_subset = transposed_species_abundance.loc[transposed_species_abundance['max'] >= Threshold]
##drop max row and transpose back to original ready for pca analysis and stats
species_subset = species_subset.drop('max',axis=1,) 
species_subset_relative_abundance = species_subset.transpose()
species_subset_relative_abundance = species_subset_relative_abundance.rename(index = cmbsf_mid)
print(species_subset_relative_abundance.shape, 'species_subset_relative_abundance')
##############################################################################
# Section 8. Select sub-set of raw counts based on >=2% abundance 
headers = list(species_subset_relative_abundance)
species_subset_raw_counts = total_count_CRS_free[headers] #select raw counts with >2% in one sample
print(species_subset_relative_abundance.shape, 'species_subset_raw_counts')
##############################################################################
# Section 9. Plot CRS abundance against depth (cmbsf) and age (cal. yrs BP)
CRS_data_percentages = CRS_count_raw.apply(lambda c: c / c.sum() * 100, axis=1)
CRS_abundance = CRS_data_percentages.iloc[:,34]

#plt.plot(cmbsf_mid, CRS_abundance)
#plt.ylabel('CRS %')
#plt.xlabel('Depth (cmbsf)') 
#plt.savefig('CRSDepth.pdf')
#plt.show()

plt.plot(age_best, CRS_abundance)
plt.ylabel('CRS %')
plt.xlabel('Age (cal. yrs BP)') 
#plt.savefig('CRSage.pdf')
plt.show()

CRS_abundance.to_csv ('CRS_abundance')
##############################################################################
# Section 10. Calculate ADA (Warnock and Scherer, 2015) ADA=(N/(AF))*(B/M)
total_FOV = FOV_Chaet + FOV_Chaet_free
a = FOV_area*FOV_Chaet
b = FOV_area*total_FOV
c = (beaker_area/mass)

#calculate CRS mv/gds
Chaetoceros_resting_spores = CRS_count_raw.iloc[:,34]
CRS_mvgds = Chaetoceros_resting_spores/a
CRS_mvgds = CRS_mvgds*c
CRS_mvgds.to_csv ('CRS_mvgds')

plt.plot(age_best, CRS_mvgds)
plt.ylabel('CRS_absolute_diatom_abundance mv/gds')
plt.xlabel('Age (cal. yrs BP)') 

#calculate total diatom mv/gds
CRS_count_mvgds = total_valves_counted_CRS/a
CRS_count_mvgds = CRS_count_mvgds*c
CRS_count_mvgds = CRS_count_mvgds.rename(index = cmbsf_mid)
CRS_count_mvgds.to_csv ('Total_diatom_mvgds')

#plt.plot(age_best, CRS_mvgds)
#plt.ylabel('CRS_absolute_diatom_abundance mv/gds')
#plt.xlabel('Age (cal. yrs BP)') 

#calculate CRS-free species mv/gds
CRS_free_total_diatoms_counted = total_count_CRS_free.sum(axis=1)
species_mvgds = total_count_CRS_free.divide(b, axis= 0, level=None, fill_value=None )
species_mvgds = species_mvgds.multiply(c, axis= 0, level=None, fill_value=None)

#select >1%
species_subset_mvgds = species_mvgds[headers] #select concentrationswith >2% in one sample
species_subset_mvgds = species_subset_mvgds.rename(index = cmbsf_mid)
##############################################################################
species_subset_mvgds.to_csv ('>1%__mvgds')
print(species_subset_mvgds.shape, 'sub_species_mvgds')

species_subset_relative_abundance.to_csv ('>1%_relative_abundance', index = False)
print(species_subset_relative_abundance.shape, '>1%_relative_abundance')
##############################################################################
# Section 11. Check if any slides need to be counted further
not_enough_counts = total_valves_counted_CRS_free <400
not_enough_counts = not_enough_counts.loc[lambda x : x!=False]
##############################################################################
# Section 12. Calculate species statistical weights
subset_total_counts = total_count_CRS_free[headers]
stat_weight = subset_total_counts
stat_weight = stat_weight.transpose()
stat_weight = stat_weight.apply(sum, axis=1)
stat_weight = stat_weight.sort_values(axis=0, ascending=False)
stat_weight.to_csv('raw_counts_stat.csv')
upper_quantile = stat_weight.quantile(q=0.75, interpolation='linear')
indicator_species = stat_weight.loc[stat_weight >= upper_quantile]
#indicator_species.to_csv ('indicator_species')
print('\nupper quantile\n%s' %upper_quantile)

indicator_species_names = indicator_species.index.tolist()
##############################################################################
indicator_species_rel_abundance = species_subset_relative_abundance[indicator_species_names]
indicator_species_mvgds = species_subset_mvgds[indicator_species_names]

indicator_species_mvgds.to_csv ('Indicator_species_mvgds')
print(indicator_species_mvgds.shape, 'Indicator_species_mvgds')

indicator_species_rel_abundance.to_csv ('Indicator_species_rel_abundance')
print(indicator_species_rel_abundance.shape, 'Indicator_species_rel_abundance')

##############################################################################
#check that threshold pulls out at least 95% of total diatoms
sub_species_relative_abundance_totals = species_subset_relative_abundance.sum(axis = 1) 
print('\nsub_species_relative_abundance_totals\n%s' %sub_species_relative_abundance_totals)
inertia = np.mean(sub_species_relative_abundance_totals)
print('\ninertia\n%s' %inertia)
##############################################################################
species_subset_relative_abundance_dropped = species_subset_relative_abundance[species_subset_relative_abundance.columns.drop(list(species_subset_relative_abundance.filter(regex='Unidentified')))]
species_subset_relative_abundance_dropped = species_subset_relative_abundance_dropped[species_subset_relative_abundance_dropped.columns.drop(list(species_subset_relative_abundance_dropped.filter(regex='colonies_>10micron')))]
species_subset_relative_abundance_dropped = species_subset_relative_abundance_dropped[species_subset_relative_abundance_dropped.columns.drop(list(species_subset_relative_abundance_dropped.filter(regex='spp')))]

print(species_subset_relative_abundance.shape, 'species_subset_raw_counts')
print(species_subset_relative_abundance_dropped.shape, 'species_subset_relative_abundance_dropped')
species_subset_relative_abundance_dropped.to_csv ('species_subset_relative_abundance_dropped')

##############################################################################
##############################################################################
##############################################################################
#Section 13. PCA analysis 
#Choose if the PCA is to be performed on the relative abundance or mvgds
x = species_subset_relative_abundance
#x = species_subset_mvgds
##############################################################################
##############################################################################
#x = x[x.columns.drop(list(x.filter(regex='spp.')))]
x = x[x.columns.drop(list(x.filter(regex='Unidentified')))]
x = x[x.columns.drop(list(x.filter(regex='colonies.')))]
x_headers = list(x)
##############################################################################
#square root transformation 
x = x.transform([np.sqrt])

#data centered and standardized (i.e. correlation matrix) 
x1 = StandardScaler().fit_transform(x)
mean_vec1 = np.mean(x1, axis=0)
correlation_matrix = (x1 - mean_vec1).T.dot((x1 - mean_vec1)) / (x1.shape[0]-1)
#check correlation matrix and matrix are the same
cor_matrix = x.corr()  

#data centered, no standardising (i.e. covariance matrix)
mean_vec = np.mean(x, axis=0)
covariance_matrix = (x - mean_vec).T.dot((x - mean_vec)) / (x.shape[0]-1)
# check covariance matrix and matrix are the same
cov_matrix = x.cov()

# calculate eigen vectors and eigen values 
eig_vals, eig_vecs = np.linalg.eig(cor_matrix)
#print('Eigenvectors \n%s' %eig_vecs)
#print('\nEigenvalues \n%s' %eig_vals)

## calculate the eigen pairs      
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs.sort()
eig_pairs.reverse()
#print('Eigenvalues in descending order:')
#for i in eig_pairs:
#    print(i[0])               
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print('cumulative variance \n%s' %cum_var_exp)

#plt.plot(cum_var_exp)
#plt.ylabel('Total variance explained (%)')
#plt.xlabel('PCA axis') 
###################################################################
###################################################################
##eig_vecs is the equivalent of PCA variable loadings in MVSP!
variable_loadings = eig_vecs
PCA_data = pd.DataFrame(variable_loadings[:,:])
index_info = pd.Series(x_headers)
PCA_data = PCA_data.rename(index=index_info)
PCA_data.to_csv ('PCA_data')

filename_3 = 'PCA_data'
PCA_data_all = pd.read_csv(filename_3,sep = ',', index_col = 0 )
PCA_data_all = PCA_data_all.iloc[:,0:4]
PCA_data_all = PCA_data_all.rename(columns = {'0':'PCA_axis_1','1':'PCA_axis_2','2':'PCA_axis_3','3':'PCA_axis_4'})
PCA_axis_1_inverted = PCA_data_all.loc[:,'PCA_axis_1'] * -1
PCA_data_all.to_csv ('PCA_data_all')

PCA_data_indicator_species = PCA_data_all.loc[indicator_species_names]
PCA_data_indicator_species.to_csv ('PCA_data_indicator_species')
###################################################################
PCA_axis_1 = variable_loadings[:,0]
PCA_axis_1 = PCA_axis_1*-1 
PCA_axis_1_mean = PCA_axis_1.mean()
PCA_axis_1 = PCA_axis_1 - PCA_axis_1_mean
PCA_axis_1_std = PCA_axis_1.std()
#calculate the upper and lower position of the SD box for the PCA axes
Upper1_1Std = PCA_axis_1_mean + PCA_axis_1_std
Lower1_1Std = PCA_axis_1_mean - PCA_axis_1_std
#print('PCA_axis_1_mean\n%s' %PCA_axis_1_mean)
#print('Upper1_1Std 1\n%s' %Upper1_1Std)
#print('Lower1_1Std\n%s' %Lower1_1Std)
###################################################################
PCA_axis_2 = variable_loadings[:,1]
PCA_axis_2_mean = PCA_axis_2.mean()
PCA_axis_2 = PCA_axis_2 - PCA_axis_2_mean
PCA_axis_2_std = PCA_axis_2.std()

Upper2_1Std = PCA_axis_2_mean + PCA_axis_2_std
Lower2_1Std = PCA_axis_2_mean - PCA_axis_2_std
#print('PCA_axis_2_mean\n%s' %PCA_axis_2_mean)
#print('Upper2_1Std\n%s' %Upper2_1Std)
#print('Lower2_1Std\n%s' %Lower2_1Std)

plt.scatter(PCA_axis_1,PCA_axis_2)
plt.show()
###################################################################
PCA_axis_3 = variable_loadings[:,2]
PCA_axis_3 = PCA_axis_3*-1 
PCA_axis_3_mean = PCA_axis_3.mean()
PCA_axis_3 = PCA_axis_3 - PCA_axis_1_mean
PCA_axis_3_std = PCA_axis_3.std()
#calculate the upper and lower position of the SD box for the PCA axes
Upper3_1Std = PCA_axis_3_mean + PCA_axis_3_std
Lower3_1Std = PCA_axis_3_mean - PCA_axis_3_std
#print('PCA_axis_1_mean\n%s' %PCA_axis_3_mean)
#print('Upper3_1Std 1\n%s' %Upper3_1Std)
#print('Lower3_1Std\n%s' %Lower3_1Std)
###################################################################
PCA_axis_4 = variable_loadings[:,3]
PCA_axis_4 = PCA_axis_4*-1 
PCA_axis_4_mean = PCA_axis_4.mean()
PCA_axis_4 = PCA_axis_4 - PCA_axis_4_mean
PCA_axis_4_std = PCA_axis_4.std()
#calculate the upper and lower position of the SD box for the PCA axes
Upper4_1Std = PCA_axis_4_mean + PCA_axis_4_std
Lower4_1Std = PCA_axis_4_mean - PCA_axis_4_std
#print('PCA_axis_1_mean\n%s' %PCA_axis_4_mean)
#print('Upper4_1Std 1\n%s' %Upper4_1Std)
#print('Lower4_1Std\n%s' %Lower4_1Std)

plt.scatter(PCA_axis_3,PCA_axis_4)
plt.show()
###################################################################
pca1 = variable_loadings[:,0] 
pca1_case_scores = x.dot(pca1)   
min_pca1_case_scores = min(pca1_case_scores)
max_pca1_case_scores = max(pca1_case_scores)
pca1_case_scores_minus_min = pca1_case_scores - min_pca1_case_scores
min_minus_max_pca1_case_scores = max_pca1_case_scores - min_pca1_case_scores
norm_pca1_case_scores = 2*(pca1_case_scores_minus_min/min_minus_max_pca1_case_scores)-1
####################################################################
pca2 = variable_loadings[:,1] 
pca2_case_scores = x.dot(pca2)   
min_pca2_case_scores = min(pca2_case_scores)
max_pca2_case_scores = max(pca2_case_scores)
pca2_case_scores_minus_min = pca2_case_scores - min_pca2_case_scores
min_minus_max_pca2_case_scores = max_pca2_case_scores - min_pca2_case_scores
norm_pca2_case_scores = 2*(pca2_case_scores_minus_min/min_minus_max_pca2_case_scores)-1
####################################################################
pca3 = variable_loadings[:,2] 
pca3_case_scores = x.dot(pca3)   
min_pca3_case_scores = min(pca3_case_scores)
max_pca3_case_scores = max(pca3_case_scores)
pca3_case_scores_minus_min = pca3_case_scores - min_pca3_case_scores
min_minus_max_pca3_case_scores = max_pca3_case_scores - min_pca3_case_scores
norm_pca3_case_scores = 2*(pca3_case_scores_minus_min/min_minus_max_pca3_case_scores)-1
####################################################################
pca4 = variable_loadings[:,3] 
pca4_case_scores = x.dot(pca4)   
min_pca4_case_scores = min(pca4_case_scores)
max_pca4_case_scores = max(pca4_case_scores)
pca4_case_scores_minus_min = pca4_case_scores - min_pca4_case_scores
min_minus_max_pca4_case_scores = max_pca4_case_scores - min_pca4_case_scores
norm_pca4_case_scores = 2*(pca4_case_scores_minus_min/min_minus_max_pca4_case_scores)-1

Case_Scores = pd.concat([norm_pca1_case_scores, norm_pca2_case_scores, norm_pca3_case_scores, norm_pca4_case_scores], axis = 1)
Case_Scores = Case_Scores.set_index(age_best)
Case_Scores.to_csv ('Case_Scores')
####################################################################
fig=plt.figure()
ax1=fig.add_subplot(311)
ax2=fig.add_subplot(311)
ax3=fig.add_subplot(312)
ax4=fig.add_subplot(312)
ax1.plot(norm_pca1_case_scores)
ax2.plot(norm_pca2_case_scores)
ax3.plot(norm_pca3_case_scores)
ax4.plot(norm_pca4_case_scores)
plt.show()
####################################################################


