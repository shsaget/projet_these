# coding=utf-8
"""
Script d'analyse par similarité de représentation
Le script prend en entrée :
    -> les images d'activation conditionnelle IRM (Betas)
    -> Un mask sur lequel limiter l'analyse en searchlight

"""

###############################################################################################################
##  IMPORTS  #####################################################################################  IMPORTS  ##
###############################################################################################################
import os
import time

import glob
import pandas as pd
import matplotlib.pyplot as plt

# pip install git+https://github.com/shsaget/nilearn@master
from nilearn import image
from nilearn.decoding import searchlight, rsanalysis

#Time onset
tic = time.time()

# import LocalUtilities as LU

###############################################################################################################
##  LOAD FILES AND DATA  #############################################################  LOAD FILES AND DATA  ##
###############################################################################################################
## Définition des chemins
cwd = os.getcwd()
betas_path = cwd + '/betas/'
masks_path = cwd + '/masks/'
## Chargement du mask des STGs
mask_img = image.load_img(masks_path + 'ROI_mask_STG_left_and_right.nii')

## Take the liste of subjects, their number, and the name of the betas
subs, nsubs, betas = rsanalysis.BetaNamesConstruction(betas_path, verbose = 0)

## Create the labels list
labels = rsanalysis.GucluLabels(nsubs, mode = 'detailled')


###############################################################################################################
##  EXTRACT ROI'S VALUES  ###########################################################  EXTRACT ROI'S VALUES  ##
###############################################################################################################
# # nsubs = 1 #just for testing
# masked_imgs = rsanalysis.roi_values_extraction(mask_img, subs, nsubs, betas, betas_path,  verbose = 1)
# save_path = cwd + '/extracted_values_from_roi'
# rsanalysis.save_roi_values(masked_imgs, save_path, subs, mask_name="test")




###############################################################################################################
##  EXTRACT SEARCHLIGHT VALUES  ###############################################  EXTRACT SEARCHLIGHT VALUES  ##
###############################################################################################################
# # nsubs = 1 #for testing
# radius = 10
# save_path = cwd + '/extracted_spheres_values_10mm'
#
# # masked_values = rsanalysis.searchlight_values_extraction(mask_img, radius, subs, nsubs, betas, betas_path)
# rsanalysis.searchlight_values_direct_saving(mask_img, radius, subs, nsubs, betas, betas_path, save_path, chunk_size=500)


###############################################################################################################
##  LOADING HDF5 VALUES FILE  ###################################################  LOADING HDF5 VALUES FILE  ##
###############################################################################################################
hdf_path = cwd + '/extracted_spheres_values_5mm'
hdf_files = sorted(glob.glob(hdf_path + '/*.h5'))
# get infos from the first H5 file assuming all the files have the same caracteristics
nbr_chunks, nbr_spheres, norm_chunk_length = rsanalysis.get_h5_info(hdf_files[0])
print(norm_chunk_length)

###############################################################################################################
##  COMPUTE and SAVE DSMs  #########################################################  COMPUTE and SAVE DSMs  ##
###############################################################################################################
#
# dsms = rsanalysis.get_dsm_from_searchlight(
#     hdf_files[0], nbr_chunks, nbr_spheres,
#     nbr_runs=8, metric = 'spearmanr')
# dsms = rsanalysis.get_dsm_from_searchlight_opti(
#     hdf_files[0], nbr_chunks, nbr_spheres,
#     norm_chunk_length, nbr_runs=8, metric = 'spearmanr')
#


for ind_sub in range(12):


	print('sujet en cours : {}' .format(subs[ind_sub]))

	dsms = rsanalysis.get_dsm_parallel(
	    hdf_files[ind_sub], nbr_chunks, nbr_spheres,
	    norm_chunk_length, nbr_runs=8, metric = 'euclidean',
	    n_jobs = 64)
	# print(type(dsms))
	# print(dsms.shape)
	# print(type(dsms[0][0]))
	# print(dsms[0][0].shape)
	# print(type(dsms[0][499]))
	save_path = cwd + '/DSMs_spheres'
	save_name =  str(subs[ind_sub]) + '_dsms_euclid_5mm_vect.h5'
	# print(save_name)
	rsanalysis.save_dsms(dsms,save_path,save_name)

#######################################
##  LOAD DSMs #########################
# ###################################"##
# #
# dsm_path = cwd + '/DSMs_spheres'
# dsm_files = sorted(glob.glob(dsm_path + '/*.h5'))
# print(dsm_files)
# #
# dsms_vect = rsanalysis.load_dsms(path_name = dsm_files[0])
# # print(dsms_vect.shape)
#
# for i in [100, 1000, 5000, 10000, 12630]:
#     dsm = rsanalysis.triu2full(rsanalysis.vect2triu(dsms_vect[0][i]))
#     rsanalysis.show_dsm(dsm)
#
# plt.show()
#

#Time end onset and programme duration
toc = time.time()
print('Durée du programme : {:.2f} sec.' .format(toc-tic) )
