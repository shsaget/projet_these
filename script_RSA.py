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
import numpy as np

# pip install git+https://github.com/shsaget/nilearn@master
import nilearn
from nilearn import image, masking, plotting
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
# nsubs = 0 #just for testing
# masked_imgs = rsanalysis.roi_values_extraction(mask_img, subs, nsubs, betas, betas_path,  verbose = 1)
# print(type(masked_imgs))

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
# hdf_path = cwd + '/extracted_spheres_values_5mm'
# hdf_files = sorted(glob.glob(hdf_path + '/*.h5'))
# # get infos from the first H5 file assuming all the files have the same caracteristics
# nbr_chunks, nbr_spheres, norm_chunk_length = rsanalysis.get_h5_info(hdf_files[0])
# print(norm_chunk_length)

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

#
# for ind_sub in range(12):
#
#
# 	print('sujet en cours : {}' .format(subs[ind_sub]))
#'/assignments_arrays'
# 	dsms = rsanalysis.get_dsm_parallel(
# 	    hdf_files[ind_sub], nbr_chunks, nbr_spheres,
# 	    norm_chunk_length, nbr_runs=8, metric = 'euclidean',
# 	    n_jobs = 64)
# 	# print(type(dsms))
# 	# print(dsms.shape)
# 	# print(type(dsms[0][0]))
# 	# print(dsms[0][0].shape)
# 	# print(type(dsms[0][499]))
# 	save_path = cwd + '/DSMs_spheres'
# 	save_name =  str(subs[ind_sub]) + '_dsms_euclid_5mm_vect.h5'
# 	# print(save_name)
# 	rsanalysis.save_dsms(dsms,save_path,save_name)

###############################################################################################################
##  LOAD DSMs  #################################################################################  LOAD DSMs  ##
###############################################################################################################

# ### BRAIN DSMs ###
#
sub_ind = 0
#
brain_dsm_path = cwd + '/DSMs_spheres/Spearman_5mm'
brain_dsm_files = sorted(glob.glob(brain_dsm_path + '/*.h5'))
print(brain_dsm_files[sub_ind])
#
brain_dsms_vect = rsanalysis.load_dsms(path_name = brain_dsm_files[sub_ind])

print(brain_dsms_vect[0][0].shape)
# rsanalysis.show_dsm(rsanalysis.vect2triu(brain_dsms_vect[0]))
# plt.show()

#
# ### MODEL DSMs ###
# model_dsm_path = cwd + '/../../DEEP/DSM_H5/spearman_avg'
# model_dsm_files = sorted(glob.glob(model_dsm_path + '/*'))
# model_dsms_vect = np.empty((1, len(model_dsm_files)), dtype = 'object')


#
# # The model_dsm are added to the same array (same structure as the brain_dsms_vect one)
# for m in range(model_dsms_vect.shape[1]):
#     model_dsms_vect[0, m] = rsanalysis.full2vect(rsanalysis.load_dsms(path_name = model_dsm_files[m]))



###############################################################################################################
##  COMPARE DSMs  ###########################################################################  COMPARE DSMs  ##
###############################################################################################################

### Compare 2 DSM and get spearman score ###
# score = rsanalysis.compare_dsms(brain_dsms_vect[0][0] , model_dsms, verbose = 1)
# print(score)
# score = rsanalysis.compare_dsms(brain_dsms_vect[0][0] , rsanalysis.full2vect(model_dsms), verbose = 1)
# print(score)
#
# for i in [0, 1, 2, 3, 4, 5, 6]:
#     dsm = rsanalysis.triu2full(rsanalysis.vect2triu(brain_dsms_vect[0][i]))
#     rsanalysis.show_dsm(dsm)
#     score = rsanalysis.compare_dsms(brain_dsms_vect[0][i] , model_dsms, verbose = 0)
#     print(score)
#
# rsanalysis.show_dsm(model_dsms)
# plt.show()

# Compare candidates DSMs and reference DSMs ###
# #
# for s in np.arange(nsubs):
#     brain_dsms_vect = rsanalysis.load_dsms(path_name = brain_dsm_files[s])
#     assign_array = rsanalysis.cand_dsms_to_ref_dsms(model_dsms_vect , brain_dsms_vect , attribution_mode = 'auto')
#     to_save = pd.DataFrame(assign_array)
#     to_save.to_hdf('assignments_arrays/layer_assigment_from_brain_euclid_5mm_and_deep_temp_spearman_avg_' + subs[s] + '.h5', key = 'assignment')

###############################################################################################################
##  LOAD ASSIGMENTS AND RETURN TO MNI  #################################  LOAD ASSIGMENTS AND RETURN TO MNI  ##
###############################################################################################################
# assig_path = cwd + '/assignments_arrays'
# assig_files = sorted(glob.glob(assig_path + '/*.h5'))
# print(assig_files)
#
# for s in np.arange(nsubs):
#
#     assign_array = np.asarray(pd.read_hdf(assig_files[s]))
#     print(type(assign_array[1][0]))
#
#     layer_assign = assign_array[1].astype(float)'/assignments_arrays'
#     reconst = masking.unmask(layer_assign, mask_img)
#     print(type(reconst))
#     plotting.plot_img(reconst)
#     plotting.show()
#     reconst.to_filename('assignments_imgs/' + subs[s] + '_assignment_euclid_5mm_spearman_deep.nii')


###############################################################################################################
##  VISUALIZE ASSIGMENTS MAPS  #################################################  VISUALIZE ASSIGMENTS MAPS  ##
###############################################################################################################

assig_map_path = cwd + '/assignments_imgs'
assig_map_files = sorted(glob.glob(assig_map_path + '/*/*03*'))

ind_img = 2
assig_img = image.load_img(assig_map_files[ind_img])

split = os.path.split(os.path.dirname(assig_map_files[ind_img]))
name = split[1]
print("Image shown : ", name)


#### Slices ####
# plt.subplots(3, 1)
# plotting.plot_stat_map(assig_img, display_mode='z', cut_coords=20, title="", cmap = 'jet')
# plotting.plot_stat_map(assig_img, display_mode='z', cut_coords=20, title="", cmap = 'jet')
# plotting.plot_stat_map(assig_img, display_mode='z', cut_coords=20, title="", cmap = 'jet')
#
# plt.show()

#### interactive ####
view = plotting.view_stat_map(assig_img, bg_img='MNI152', vmax=None, cmap='jet', symmetric_cmap=False)
view.open_in_browser()

####   3D   ####
# 'cold_white_hot'
# view = plotting.html_surface.view_img_on_surf_sym(assig_img, surf_mesh='fsaverage', cmap = 'jet' , black_bg = False , symmetric_cmap=False )
# view.open_in_browser()
# view.save_as_html(assig_map_path + '/' + name + "_lowres.html")


#Time end onset and programme duration
toc = time.time()
print('Durée du programme : {:.3f} sec.' .format(toc-tic) )
