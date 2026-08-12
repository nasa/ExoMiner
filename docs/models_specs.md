---
license: apache-2.0
language:
- en
tags:
- nasa
- exoplanets
- astronomy
- time-series
- keras
- photometry
- exominer
datasets:
- tess
library_name: keras
---

# ExoMiner Pipeline Model Specifications

**Date:** June 2026  
**Pipeline Version:** `v2.1`  
**Base Architecture:** `ExoMiner++ (ExoMinerPlusPlusTemp)`

This document serves as the primary bookkeeping record for the models currently shipped with the ExoMiner pipeline, detailing the trained tasks, input features, hyperparameters, and the compiled Keras network topology.

---

## 1. Shipped Models & Tasks

The pipeline currently ships with two primary models designed for distinct stages of exoplanet candidate processing.

### Task 1: Planet Validation (Planet vs. Not-Planet)
* **Model ID:** `cv_tess-spoc-tces_2min-s1-s98_10-folds-matched-distribution-2min-paper-by-weighted-loss_exominerpp_validation_no-secondaries_20260526_151727`
* **Classes:** Planet, Not-Planet
* **Training Objective:** Weighted binary cross-entropy loss. Designed to match the sub-class distribution of the TESS 2-min paper for KPs, CPs, FPs, EBs, BDs, and NTPs.
* **Dataset:** `cv_tfrecords_tess-spoc-2min_tces_s1-s98_5-6-2026_1552_no-secondaries10`
  * 10-fold cross-validation.
  * **Class Rates:** Planets (~13%), Not-planets (~87% -> AFP: ~29%, NTP: ~58%).

### Task 2: Photometry Vetting
* **Model ID:** `cv_tess-spoc-tces_2min-s1-s98_10-folds_added-ntps_20260615_212500`
* **Classes:** PC (Planet Candidate), AFP (Astrophysical False Positive), NTP (Non-Transiting Phenomena)
* **Training Objective:** Non-weighted binary cross-entropy.
* **Dataset:**
  * 10-fold cross-validation.
  * Supplemented with NTPs from TEC results past Sector 41 (shared by Katharine, up to S98).
  * **Class Rates:** PC (~20%), AFP (~15%), NTP (~65%).

---

## 2. Input Features Definition

Below is the expected feature set ingested by the pipeline. Each feature specifies its required tensor dimension `[dim]` and datatype (`float`).

### 1D & 2D Views
* **Unfolded Local Flux:** `unfolded_local_flux_view_fluxnorm` [20, 31]
* **Global Flux:** `global_flux_view_fluxnorm` & `var` [301, 1]
* **Local Flux:** `local_flux_view_fluxnorm` & `var` [31, 1]
* **Local Flux (Odd/Even):** `local_flux_odd_view_fluxnorm` & `var` [31, 1], `local_flux_even_view_fluxnorm` & `var` [31, 1]
* **Local Centroid:** `local_centr_view_std_noclip` & `var` [31, 1]
* **Local Weak Secondary:** `local_weak_secondary_view_selfnorm` & `var` [31, 1]
* **Momentum Dump:** `local_momentum_dump_view` & `var` [31, 1]
* **Flux Trend:** `flux_trend_global_norm` & `var` [301, 1]
* **Periodogram:** `pgram_smooth_norm` [674, 1], `pgram_tpm_smooth_norm` [674, 1]

### 3D Views (Image Tensors)
* **Difference Images:** `diff_imgs_tc_hybrid_norm` [55, 55, 5]
* **Out-of-Transit (OOT) Images:** `oot_imgs_tc_hybrid_norm` [55, 55, 5]
* **SNR Images:** `snr_imgs_tc_hybrid_norm` [55, 55, 5]
* **Quality Map:** `quality` [5, 1]

### Scalar Metrics `[1,]`
* **Transit Stats:** `tce_num_transits_norm`, `tce_num_transits_obs_norm`, `flux_global_stat_abs_min_norm`, `flux_local_stat_abs_min`, `flux_even_local_stat_abs_min_norm`, `flux_odd_local_stat_abs_min_norm`, `tce_maxmes_norm`, `tce_albedo_stat_norm`, `tce_ptemp_stat_norm`, `flux_weak_secondary_local_stat_abs_min_norm`, `flux_trend_global_stat_min_norm`, `flux_trend_global_stat_max_norm`, `pgram_smooth_max_power_norm`, `pgram_tpm_smooth_max_power_norm`
* **DV/TCE Fit:** `boot_fap_norm`, `tce_cap_stat_norm`, `tce_hap_stat_norm`, `tce_period_norm`, `tce_max_mult_ev_norm`, `tce_max_sngle_ev_norm`, `tce_robstat_norm`, `tce_model_chisq_norm`, `tce_prad_norm`
* **Stellar/Astrometric:** `tce_sdens_norm`, `tce_steff_norm`, `tce_smet_norm`, `tce_slogg_norm`, `tce_smass_norm`, `tce_sradius_norm`, `ruwe_norm`, `mag_shift_norm`, `tce_dikco_msky_norm`, `tce_dikco_msky_err_norm`

---

## 3. Network Configuration & Hyperparameters

### Model Topology Setup
The architecture relies on multiple specialized convolutional branches concatenated with dense scalar branches.
* **Conv Branches:** `local_unfolded_flux`, `global_flux`, `local_flux`, `local_weak_secondary`, `local_centroid`, `local_odd_even`, `momentum_dump`, `flux_trend`, `flux_periodogram`, `diff_img_branch`
* **Scalar Branches:** `stellar`, `dv_tce_fit`

### Hyperparameter Dictionary
* **Global Conv:** 2 blocks (3 LS/block), Init Filters: 3, Kernel: 5, Pool: 8
* **Local Flux Conv:** 2 blocks (3 LS/block), Init Filters: 3, Kernel: 6, Pool: 4
* **Centroid/Trend/Pgram Conv:** 2 blocks (3 LS/block), Init Filters: 3, Kernel: 5, Pool: 4
* **Difference Image Conv:** 3 blocks (3 LS/block), Init Filters: 2, Kernel: 3, Pool: 2, FC Units: 3
* **Unfolded Flux Conv:** 2 blocks (3 LS/block), Init Filters: 3, Kernel: 6, Pool: 4, Unfolded Stats Kernel: 1 (Filters: 4)
* **Shared Conv Config:** FC Conv Units: 3, Kernel Stride: 1, Pool Stride: 1, Dropout: `0.1211`
* **Classification Head:** 4 FC layers, Init Neurons: 512, Dropout: `0.02149`, L2-Decay: `null`

### Optimization Engine
* **Optimizer:** Adam
* **Learning Rate:** `4.176e-05`
* **Loss:** `crossentropy`
* **Activation:** `prelu`

---

## 4. Keras Architecture Summary

Below is the compiled structural output of the active graph.

**Parameter Summary:**
* **Total params:** 1,424,003 (5.43 MB)
* **Trainable params:** 1,424,003 (5.43 MB)
* **Non-trainable params:** 0 (0.00 Byte)

```text
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 diff_imgs_tc_hybrid_norm (  [(None, 55, 55, 5)]          0         []                            
 InputLayer)                                                                                      
 oot_imgs_tc_hybrid_norm (I  [(None, 55, 55, 5)]          0         []                            
 nputLayer)                                                                                       
 snr_imgs_tc_hybrid_norm (I  [(None, 55, 55, 5)]          0         []                            
 nputLayer)                                                                                       
 expanding_diff_imgs_tc_hyb  (None, 55, 55, 5, 1)         0         ['diff_imgs_tc_hybrid_norm[0][0]']                          
 expanding_oot_imgs_tc_hybr  (None, 55, 55, 5, 1)         0         ['oot_imgs_tc_hybrid_norm[0][0]']                           
 expanding_snr_imgs_tc_hybr  (None, 55, 55, 5, 1)         0         ['snr_imgs_tc_hybrid_norm[0][0]']                           
 input_diff_img_concat (Con  (None, 55, 55, 5, 3)         0         ['expanding_diff_imgs_tc_hybrid_norm_dims[0][0]',           
 catenate)                                                           'expanding_oot_imgs_tc_hybrid_norm_dims[0][0]',            
                                                                     'expanding_snr_imgs_tc_hybrid_norm_dims[0][0]']            
 convdiff_img_0_0 (Conv3D)   (None, 55, 55, 5, 4)         112       ['input_diff_img_concat[0][0]']                             
 preludiff_img_0_0 (PReLU)   (None, 55, 55, 5, 4)         20        ['convdiff_img_0_0[0][0]']    
 maxpooling_diff_img_0_0 (M  (None, 54, 54, 5, 4)         0         ['preludiff_img_0_0[0][0]']   
 axPooling3D)                                                                                     
 convdiff_img_0_1 (Conv3D)   (None, 54, 54, 5, 4)         148       ['maxpooling_diff_img_0_0[0][0]']                           
 preludiff_img_0_1 (PReLU)   (None, 54, 54, 5, 4)         20        ['convdiff_img_0_1[0][0]']    
 maxpooling_diff_img_0_1 (M  (None, 53, 53, 5, 4)         0         ['preludiff_img_0_1[0][0]']   
 axPooling3D)                                                                                     
 convdiff_img_0_2 (Conv3D)   (None, 53, 53, 5, 4)         148       ['maxpooling_diff_img_0_1[0][0]']                           
 preludiff_img_0_2 (PReLU)   (None, 53, 53, 5, 4)         20        ['convdiff_img_0_2[0][0]']    
 maxpooling_diff_img_0_2 (M  (None, 52, 52, 5, 4)         0         ['preludiff_img_0_2[0][0]']   
 axPooling3D)                                                                                     
 convdiff_img_1_0 (Conv3D)   (None, 52, 52, 5, 8)         296       ['maxpooling_diff_img_0_2[0][0]']                           
 local_flux_odd_view_fluxno  [(None, 31, 1)]              0         []                            
 rm (InputLayer)                                                                                  
 local_flux_odd_view_fluxno  [(None, 31, 1)]              0         []                            
 rm_var (InputLayer)                                                                              
 local_flux_even_view_fluxn  [(None, 31, 1)]              0         []                            
 orm (InputLayer)                                                                                 
 local_flux_even_view_fluxn  [(None, 31, 1)]              0         []                            
 orm_var (InputLayer)                                                                             
 local_flux_view_fluxnorm (  [(None, 31, 1)]              0         []                            
 InputLayer)                                                                                      
 local_flux_view_fluxnorm_v  [(None, 31, 1)]              0         []                            
 ar (InputLayer)                                                                                  
 local_weak_secondary_view_  [(None, 31, 1)]              0         []                            
 selfnorm (InputLayer)                                                                            
 local_weak_secondary_view_  [(None, 31, 1)]              0         []                            
 selfnorm_var (InputLayer)                                                                        
 preludiff_img_1_0 (PReLU)   (None, 52, 52, 5, 8)         40        ['convdiff_img_1_0[0][0]']    
 expanding_local_flux_odd_v  (None, 1, 31, 1)             0         ['local_flux_odd_view_fluxnorm[0][0]']                      
 iew_fluxnorm_dim (Reshape)                                                                       
 expanding_local_flux_odd_v  (None, 1, 31, 1)             0         ['local_flux_odd_view_fluxnorm_var[0][0]']                  
 iew_fluxnorm_var_dim (Resh                                                                       
 ape)                                                                                             
 expanding_local_flux_even_  (None, 1, 31, 1)             0         ['local_flux_even_view_fluxnorm[0][0]']                     
 view_fluxnorm_dim (Reshape                                                                       
 )                                                                                                
 expanding_local_flux_even_  (None, 1, 31, 1)             0         ['local_flux_even_view_fluxnorm_var[0][0]']                 
 view_fluxnorm_var_dim (Res                                                                       
 hape)                                                                                            
 expanding_local_flux_view_  (None, 1, 31, 1)             0         ['local_flux_view_fluxnorm[0][0]']                          
 fluxnorm_dim (Reshape)                                                                           
 expanding_local_flux_view_  (None, 1, 31, 1)             0         ['local_flux_view_fluxnorm_var[0][0]']                      
 fluxnorm_var_dim (Reshape)                                                                       
 expanding_local_weak_secon  (None, 1, 31, 1)             0         ['local_weak_secondary_view_selfnorm[0][0]']                
 dary_view_selfnorm_dim (Re                                                                       
 shape)                                                                                           
 expanding_local_weak_secon  (None, 1, 31, 1)             0         ['local_weak_secondary_view_selfnorm_var[0][0]']            
 dary_view_selfnorm_var_dim                                                                       
  (Reshape)                                                                                       
 unfolded_local_flux_view_f  [(None, 20, 31)]             0         []                            
 luxnorm (InputLayer)                                                                             
 maxpooling_diff_img_1_0 (M  (None, 51, 51, 5, 8)         0         ['preludiff_img_1_0[0][0]']   
 axPooling3D)                                                                                     
 local_flux_concat_local_fl  (None, 1, 31, 2)             0         ['expanding_local_flux_odd_view_fluxnorm_dim[0][0]',        
 ux_odd_view_fluxnorm_with_                                          'expanding_local_flux_odd_view_fluxnorm_var_dim[0][0]']    
 var (Concatenate)                                                                                
 local_flux_concat_local_fl  (None, 1, 31, 2)             0         ['expanding_local_flux_even_view_fluxnorm_dim[0][0]',       
 ux_even_view_fluxnorm_with                                          'expanding_local_flux_even_view_fluxnorm_var_dim[0][0]']   
 _var (Concatenate)                                                                               
 local_flux_concat_local_fl  (None, 1, 31, 2)             0         ['expanding_local_flux_view_fluxnorm_dim[0][0]',            
 ux_view_fluxnorm_with_var                                           'expanding_local_flux_view_fluxnorm_var_dim[0][0]']        
 (Concatenate)                                                                                    
 local_flux_concat_local_we  (None, 1, 31, 2)             0         ['expanding_local_weak_secondary_view_selfnorm_dim[0][0]',  
 ak_secondary_view_selfnorm                                          'expanding_local_weak_secondary_view_selfnorm_var_dim[0][0]']
 _with_var (Concatenate)                                                                          
 expanding_unfolded_flux_di  (None, 20, 31, 1)            0         ['unfolded_local_flux_view_fluxnorm[0][0]']                 
 m (Reshape)                                                                                      
 convdiff_img_1_1 (Conv3D)   (None, 51, 51, 5, 8)         584       ['maxpooling_diff_img_1_0[0][0]']                           
 local_flux_concat_local_vi  (None, 4, 31, 2)             0         ['local_flux_concat_local_flux_odd_view_fluxnorm_with_var[0][0]',                         
 ews (Concatenate)                                                   'local_flux_concat_local_flux_even_view_fluxnorm_with_var[0][0]',                        
                                                                     'local_flux_concat_local_flux_view_fluxnorm_with_var[0][0]', 
                                                                     'local_flux_concat_local_weak_secondary_view_selfnorm_with_var[0][0]']                  
 unfolded_flux_convlocal_un  (None, 20, 31, 8)            56        ['expanding_unfolded_flux_dim[0][0]']                       
 folded_flux_0_0 (Conv2D)                                                                         
 preludiff_img_1_1 (PReLU)   (None, 51, 51, 5, 8)         40        ['convdiff_img_1_1[0][0]']    
 local_flux_conv0_0 (Conv2D  (None, 4, 31, 8)             104       ['local_flux_concat_local_views[0][0]']                     
 )                                                                                                
 unfolded_flux_prelulocal_u  (None, 20, 31, 8)            8         ['unfolded_flux_convlocal_unfolded_flux_0_0[0][0]']         
 nfolded_flux_0_0 (PReLU)                                                                         
 maxpooling_diff_img_1_1 (M  (None, 50, 50, 5, 8)         0         ['preludiff_img_1_1[0][0]']   
 axPooling3D)                                                                                     
 local_flux_prelu_0_0 (PReL  (None, 4, 31, 8)             8         ['local_flux_conv0_0[0][0]']  
 U)                                                                                               
 unfolded_flux_convlocal_un  (None, 20, 31, 8)            392       ['unfolded_flux_prelulocal_unfolded_flux_0_0[0][0]']        
 folded_flux_0_1 (Conv2D)                                                                         
 convdiff_img_1_2 (Conv3D)   (None, 50, 50, 5, 8)         584       ['maxpooling_diff_img_1_1[0][0]']                           
 global_flux_view_fluxnorm   [(None, 301, 1)]             0         []                            
 (InputLayer)                                                                                     
 global_flux_view_fluxnorm_  [(None, 301, 1)]             0         []                            
 var (InputLayer)                                                                                 
 local_centr_view_std_nocli  [(None, 31, 1)]              0         []                            
 p (InputLayer)                                                                                   
 local_centr_view_std_nocli  [(None, 31, 1)]              0         []                            
 p_var (InputLayer)                                                                               
 flux_trend_global_norm (In  [(None, 301, 1)]             0         []                            
 putLayer)                                                                                        
 flux_trend_global_norm_var  [(None, 301, 1)]             0         []                            
  (InputLayer)                                                                                    
 pgram_smooth_norm (InputLa  [(None, 674, 1)]             0         []                            
 yer)                                                                                             
 pgram_tpm_smooth_norm (Inp  [(None, 674, 1)]             0         []                            
 utLayer)                                                                                         
 local_flux_conv0_1 (Conv2D  (None, 4, 31, 8)             392       ['local_flux_prelu_0_0[0][0]']
 )                                                                                                
 unfolded_flux_prelulocal_u  (None, 20, 31, 8)            8         ['unfolded_flux_convlocal_unfolded_flux_0_1[0][0]']         
 nfolded_flux_0_1 (PReLU)                                                                         
 preludiff_img_1_2 (PReLU)   (None, 50, 50, 5, 8)         40        ['convdiff_img_1_2[0][0]']    
 input_global_flux (Concate  (None, 301, 2)               0         ['global_flux_view_fluxnorm[0][0]',                         
 nate)                                                               'global_flux_view_fluxnorm_var[0][0]']                     
 input_local_centroid (Conc  (None, 31, 2)                0         ['local_centr_view_std_noclip[0][0]',                       
 atenate)                                                            'local_centr_view_std_noclip_var[0][0]']                   
 local_momentum_dump_view (  [(None, 31, 1)]              0         []                            
 InputLayer)                                                                                      
 local_momentum_dump_view_v  [(None, 31, 1)]              0         []                            
 ar (InputLayer)                                                                                  
 input_flux_trend (Concaten  (None, 301, 2)               0         ['flux_trend_global_norm[0][0]',                            
 ate)                                                                'flux_trend_global_norm_var[0][0]']                        
 input_flux_periodogram (Co  (None, 674, 2)               0         ['pgram_smooth_norm[0][0]',   
 ncatenate)                                                          'pgram_tpm_smooth_norm[0][0]']                             
 local_flux_prelu_0_1 (PReL  (None, 4, 31, 8)             8         ['local_flux_conv0_1[0][0]']  
 U)                                                                                               
 unfolded_flux_convlocal_un  (None, 20, 31, 8)            392       ['unfolded_flux_prelulocal_unfolded_flux_0_1[0][0]']        
 folded_flux_0_2 (Conv2D)                                                                         
 maxpooling_diff_img_1_2 (M  (None, 49, 49, 5, 8)         0         ['preludiff_img_1_2[0][0]']   
 axPooling3D)                                                                                     
 convglobal_flux_0_0 (Conv1  (None, 301, 8)               88        ['input_global_flux[0][0]']   
 D)                                                                                               
 convlocal_centroid_0_0 (Co  (None, 31, 8)                88        ['input_local_centroid[0][0]']
 nv1D)                                                                                            
 input_momentum_dump (Conca  (None, 31, 2)                0         ['local_momentum_dump_view[0][0]',                          
 tenate)                                                             'local_momentum_dump_view_var[0][0]']                      
 convflux_trend_0_0 (Conv1D  (None, 301, 8)               88        ['input_flux_trend[0][0]']    
 )                                                                                                
 convflux_periodogram_0_0 (  (None, 674, 8)               88        ['input_flux_periodogram[0][0]']                            
 Conv1D)                                                                                          
 local_flux_conv0_2 (Conv2D  (None, 4, 31, 8)             392       ['local_flux_prelu_0_1[0][0]']
 )                                                                                                
 unfolded_flux_prelulocal_u  (None, 20, 31, 8)            8         ['unfolded_flux_convlocal_unfolded_flux_0_2[0][0]']         
 nfolded_flux_0_2 (PReLU)                                                                         
 convdiff_img_2_0 (Conv3D)   (None, 49, 49, 5, 16)        1168      ['maxpooling_diff_img_1_2[0][0]']                           
 preluglobal_flux_0_0 (PReL  (None, 301, 8)               1         ['convglobal_flux_0_0[0][0]'] 
 U)                                                                                               
 prelulocal_centroid_0_0 (P  (None, 31, 8)                1         ['convlocal_centroid_0_0[0][0]']                            
 ReLU)                                                                                            
 convmomentum_dump_0_0 (Con  (None, 31, 8)                88        ['input_momentum_dump[0][0]'] 
 v1D)                                                                                             
 preluflux_trend_0_0 (PReLU  (None, 301, 8)               1         ['convflux_trend_0_0[0][0]']  
 )                                                                                                
 preluflux_periodogram_0_0   (None, 674, 8)               1         ['convflux_periodogram_0_0[0][0]']                          
 (PReLU)                                                                                          
 local_flux_prelu_0_2 (PReL  (None, 4, 31, 8)             8         ['local_flux_conv0_2[0][0]']  
 U)                                                                                               
 unfolded_flux_maxpooling_l  (None, 20, 28, 8)            0         ['unfolded_flux_prelulocal_unfolded_flux_0_2[0][0]']        
 ocal_unfolded_flux_0 (MaxP                                                                       
 ooling2D)                                                                                        
 preludiff_img_2_0 (PReLU)   (None, 49, 49, 5, 16)        80        ['convdiff_img_2_0[0][0]']    
 convglobal_flux_0_1 (Conv1  (None, 301, 8)               328       ['preluglobal_flux_0_0[0][0]']
 D)                                                                                               
 convlocal_centroid_0_1 (Co  (None, 31, 8)                328       ['prelulocal_centroid_0_0[0][0]']                           
 nv1D)                                                                                            
 prelumomentum_dump_0_0 (PR  (None, 31, 8)                1         ['convmomentum_dump_0_0[0][0]']                             
 eLU)                                                                                             
 convflux_trend_0_1 (Conv1D  (None, 301, 8)               328       ['preluflux_trend_0_0[0][0]'] 
 )                                                                                                
 convflux_periodogram_0_1 (  (None, 674, 8)               328       ['preluflux_periodogram_0_0[0][0]']                         
 Conv1D)                                                                                          
 local_flux_maxpooling_0 (M  (None, 4, 28, 8)             0         ['local_flux_prelu_0_2[0][0]']
 axPooling2D)                                                                                     
 unfolded_flux_convlocal_un  (None, 20, 28, 16)           784       ['unfolded_flux_maxpooling_local_unfolded_flux_0[0][0]']    
 folded_flux_1_0 (Conv2D)                                                                         
 maxpooling_diff_img_2_0 (M  (None, 48, 48, 5, 16)        0         ['preludiff_img_2_0[0][0]']   
 axPooling3D)                                                                                     
 preluglobal_flux_0_1 (PReL  (None, 301, 8)               1         ['convglobal_flux_0_1[0][0]'] 
 U)                                                                                               
 prelulocal_centroid_0_1 (P  (None, 31, 8)                1         ['convlocal_centroid_0_1[0][0]']                            
 ReLU)                                                                                            
 convmomentum_dump_0_1 (Con  (None, 31, 8)                328       ['prelumomentum_dump_0_0[0][0]']                            
 v1D)                                                                                             
 preluflux_trend_0_1 (PReLU  (None, 301, 8)               1         ['convflux_trend_0_1[0][0]']  
 )                                                                                                
 preluflux_periodogram_0_1   (None, 674, 8)               1         ['convflux_periodogram_0_1[0][0]']                          
 (PReLU)                                                                                          
 local_flux_conv1_0 (Conv2D  (None, 4, 28, 16)            784       ['local_flux_maxpooling_0[0][0]']                           
 )                                                                                                
 unfolded_flux_prelulocal_u  (None, 20, 28, 16)           16        ['unfolded_flux_convlocal_unfolded_flux_1_0[0][0]']         
 nfolded_flux_1_0 (PReLU)                                                                         
 convdiff_img_2_1 (Conv3D)   (None, 48, 48, 5, 16)        2320      ['maxpooling_diff_img_2_0[0][0]']                           
 convglobal_flux_0_2 (Conv1  (None, 301, 8)               328       ['preluglobal_flux_0_1[0][0]']
 D)                                                                                               
 convlocal_centroid_0_2 (Co  (None, 31, 8)                328       ['prelulocal_centroid_0_1[0][0]']                           
 nv1D)                                                                                            
 prelumomentum_dump_0_1 (PR  (None, 31, 8)                1         ['convmomentum_dump_0_1[0][0]']                             
 eLU)                                                                                             
 convflux_trend_0_2 (Conv1D  (None, 301, 8)               328       ['preluflux_trend_0_1[0][0]'] 
 )                                                                                                
 convflux_periodogram_0_2 (  (None, 674, 8)               328       ['preluflux_periodogram_0_1[0][0]']                         
 Conv1D)                                                                                          
 local_flux_prelu_1_0 (PReL  (None, 4, 28, 16)            16        ['local_flux_conv1_0[0][0]']  
 U)                                                                                               
 unfolded_flux_convlocal_un  (None, 20, 28, 16)           1552      ['unfolded_flux_prelulocal_unfolded_flux_1_0[0][0]']        
 folded_flux_1_1 (Conv2D)                                                                         
 preludiff_img_2_1 (PReLU)   (None, 48, 48, 5, 16)        80        ['convdiff_img_2_1[0][0]']    
 preluglobal_flux_0_2 (PReL  (None, 301, 8)               1         ['convglobal_flux_0_2[0][0]'] 
 U)                                                                                               
 prelulocal_centroid_0_2 (P  (None, 31, 8)                1         ['convlocal_centroid_0_2[0][0]']                            
 ReLU)                                                                                            
 convmomentum_dump_0_2 (Con  (None, 31, 8)                328       ['prelumomentum_dump_0_1[0][0]']                            
 v1D)                                                                                             
 preluflux_trend_0_2 (PReLU  (None, 301, 8)               1         ['convflux_trend_0_2[0][0]']  
 )                                                                                                
 preluflux_periodogram_0_2   (None, 674, 8)               1         ['convflux_periodogram_0_2[0][0]']                          
 (PReLU)                                                                                          
 local_flux_conv1_1 (Conv2D  (None, 4, 28, 16)            1552      ['local_flux_prelu_1_0[0][0]']
 )                                                                                                
 unfolded_flux_prelulocal_u  (None, 20, 28, 16)           16        ['unfolded_flux_convlocal_unfolded_flux_1_1[0][0]']         
 nfolded_flux_1_1 (PReLU)                                                                         
 maxpooling_diff_img_2_1 (M  (None, 47, 47, 5, 16)        0         ['preludiff_img_2_1[0][0]']   
 axPooling3D)                                                                                     
 maxpooling_global_flux_0 (  (None, 294, 8)               0         ['preluglobal_flux_0_2[0][0]']
 MaxPooling1D)                                                                                    
 maxpooling_local_centroid_  (None, 28, 8)                0         ['prelulocal_centroid_0_2[0][0]']                           
 0 (MaxPooling1D)                                                                                 
 prelumomentum_dump_0_2 (PR  (None, 31, 8)                1         ['convmomentum_dump_0_2[0][0]']                             
 eLU)                                                                                             
 maxpooling_flux_trend_0 (M  (None, 298, 8)               0         ['preluflux_trend_0_2[0][0]'] 
 axPooling1D)                                                                                     
 maxpooling_flux_periodogra  (None, 671, 8)               0         ['preluflux_periodogram_0_2[0][0]']                         
 m_0 (MaxPooling1D)                                                                               
 local_flux_prelu_1_1 (PReL  (None, 4, 28, 16)            16        ['local_flux_conv1_1[0][0]']  
 U)                                                                                               
 unfolded_flux_convlocal_un  (None, 20, 28, 16)           1552      ['unfolded_flux_prelulocal_unfolded_flux_1_1[0][0]']        
 folded_flux_1_2 (Conv2D)                                                                         
 convdiff_img_2_2 (Conv3D)   (None, 47, 47, 5, 16)        2320      ['maxpooling_diff_img_2_1[0][0]']                           
 convglobal_flux_1_0 (Conv1  (None, 294, 16)              656       ['maxpooling_global_flux_0[0][0]']                          
 D)                                                                                               
 convlocal_centroid_1_0 (Co  (None, 28, 16)               656       ['maxpooling_local_centroid_0[0][0]']                       
 nv1D)                                                                                            
 maxpooling_momentum_dump_0  (None, 28, 8)                0         ['prelumomentum_dump_0_2[0][0]']                            
  (MaxPooling1D)                                                                                  
 convflux_trend_1_0 (Conv1D  (None, 298, 16)              656       ['maxpooling_flux_trend_0[0][0]']                           
 )                                                                                                
 convflux_periodogram_1_0 (  (None, 671, 16)              656       ['maxpooling_flux_periodogram_0[0][0]']                     
 Conv1D)                                                                                          
 local_flux_conv1_2 (Conv2D  (None, 4, 28, 16)            1552      ['local_flux_prelu_1_1[0][0]']
 )                                                                                                
 unfolded_flux_prelulocal_u  (None, 20, 28, 16)           16        ['unfolded_flux_convlocal_unfolded_flux_1_2[0][0]']         
 nfolded_flux_1_2 (PReLU)                                                                         
 preludiff_img_2_2 (PReLU)   (None, 47, 47, 5, 16)        80        ['convdiff_img_2_2[0][0]']    
 preluglobal_flux_1_0 (PReL  (None, 294, 16)              1         ['convglobal_flux_1_0[0][0]'] 
 U)                                                                                               
 prelulocal_centroid_1_0 (P  (None, 28, 16)               1         ['convlocal_centroid_1_0[0][0]']                            
 ReLU)                                                                                            
 convmomentum_dump_1_0 (Con  (None, 28, 16)               656       ['maxpooling_momentum_dump_0[0][0]']                        
 v1D)                                                                                             
 preluflux_trend_1_0 (PReLU  (None, 298, 16)              1         ['convflux_trend_1_0[0][0]']  
 )                                                                                                
 preluflux_periodogram_1_0   (None, 671, 16)              1         ['convflux_periodogram_1_0[0][0]']                          
 (PReLU)                                                                                          
 local_flux_prelu_1_2 (PReL  (None, 4, 28, 16)            16        ['local_flux_conv1_2[0][0]']  
 U)                                                                                               
 unfolded_flux_maxpooling_l  (None, 20, 25, 16)           0         ['unfolded_flux_prelulocal_unfolded_flux_1_2[0][0]']        
 ocal_unfolded_flux_1 (MaxP                                                                       
 ooling2D)                                                                                        
 maxpooling_diff_img_2_2 (M  (None, 46, 46, 5, 16)        0         ['preludiff_img_2_2[0][0]']   
 axPooling3D)                                                                                     
 convglobal_flux_1_1 (Conv1  (None, 294, 16)              1296      ['preluglobal_flux_1_0[0][0]']
 D)                                                                                               
 convlocal_centroid_1_1 (Co  (None, 28, 16)               1296      ['prelulocal_centroid_1_0[0][0]']                           
 nv1D)                                                                                            
 prelumomentum_dump_1_0 (PR  (None, 28, 16)               1         ['convmomentum_dump_1_0[0][0]']                             
 eLU)                                                                                             
 convflux_trend_1_1 (Conv1D  (None, 298, 16)              1296      ['preluflux_trend_1_0[0][0]'] 
 )                                                                                                
 convflux_periodogram_1_1 (  (None, 671, 16)              1296      ['preluflux_periodogram_1_0[0][0]']                         
 Conv1D)                                                                                          
 local_flux_maxpooling_1 (M  (None, 4, 25, 16)            0         ['local_flux_prelu_1_2[0][0]']
 axPooling2D)                                                                                     
 unfolded_flux_split_input   [(None, 1, 25, 16),          0         ['unfolded_flux_maxpooling_local_unfolded_flux_1[0][0]']    
 (SplitLayer)                 ... (repeats 20 times) ]                                                                               
 permute_diff_imgs (Permute  (None, 46, 46, 16, 5)        0         ['maxpooling_diff_img_2_2[0][0]']                           
 )                                                                                                
 quality (InputLayer)        [(None, 5, 1)]               0         []                            
 preluglobal_flux_1_1 (PReL  (None, 294, 16)              1         ['convglobal_flux_1_1[0][0]'] 
 U)                                                                                               
 prelulocal_centroid_1_1 (P  (None, 28, 16)               1         ['convlocal_centroid_1_1[0][0]']                            
 ReLU)                                                                                            
 convmomentum_dump_1_1 (Con  (None, 28, 16)               1296      ['prelumomentum_dump_1_0[0][0]']                            
 v1D)                                                                                             
 preluflux_trend_1_1 (PReLU  (None, 298, 16)              1         ['convflux_trend_1_1[0][0]']  
 )                                                                                                
 preluflux_periodogram_1_1   (None, 671, 16)              1         ['convflux_periodogram_1_1[0][0]']                          
 (PReLU)                                                                                          
 local_flux_split_merge (Sp  [(None, 1, 25, 16) x 4]      0         ['local_flux_maxpooling_1[0][0]']                           
 litLayer)                                                                                        
 unfolded_flux_min (Minimum  (None, 1, 25, 16)            0         ['unfolded_flux_split_input[0][0]...20']                        
 )                                                                                                
 unfolded_flux_max (Maximum  (None, 1, 25, 16)            0         ['unfolded_flux_split_input[0][0]...20']                        
 )                                                                                                
 unfolded_flux_avg (Average  (None, 1, 25, 16)            0         ['unfolded_flux_split_input[0][0]...20']                        
 )                                                                                                
 flatten_diff_imgs (Reshape  (None, 33856, 5)             0         ['permute_diff_imgs[0][0]']   
 )                                                                                                
 permute_quality (Permute)   (None, 1, 5)                 0         ['quality[0][0]']             
 convglobal_flux_1_2 (Conv1  (None, 294, 16)              1296      ['preluglobal_flux_1_1[0][0]']
 D)                                                                                               
 convlocal_centroid_1_2 (Co  (None, 28, 16)               1296      ['prelulocal_centroid_1_1[0][0]']                           
 nv1D)                                                                                            
 prelumomentum_dump_1_1 (PR  (None, 28, 16)               1         ['convmomentum_dump_1_1[0][0]']                             
 eLU)                                                                                             
 convflux_trend_1_2 (Conv1D  (None, 298, 16)              1296      ['preluflux_trend_1_1[0][0]'] 
 )                                                                                                
 convflux_periodogram_1_2 (  (None, 671, 16)              1296      ['preluflux_periodogram_1_1[0][0]']                         
 Conv1D)                                                                                          
 local_flux_transit_merge_l  (None, 2, 25, 16)            0         ['local_flux_split_merge[0][0]',                            
 ocal_odd_even (Concatenate                                          'local_flux_split_merge[0][1]']                            
 )                                                                                                
 unfolded_flux_merge (Conca  (None, 3, 25, 16)            0         ['unfolded_flux_min[0][0]',   
 tenate)                                                             'unfolded_flux_max[0][0]',   
                                                                     'unfolded_flux_avg[0][0]']   
 flatten_wscalar_diff_img_i  (None, 33857, 5)             0         ['flatten_diff_imgs[0][0]',   
 mgsscalars (Concatenate)                                            'permute_quality[0][0]']     
 preluglobal_flux_1_2 (PReL  (None, 294, 16)              1         ['convglobal_flux_1_2[0][0]'] 
 U)                                                                                               
 prelulocal_centroid_1_2 (P  (None, 28, 16)               1         ['convlocal_centroid_1_2[0][0]']                            
 ReLU)                                                                                            
 convmomentum_dump_1_2 (Con  (None, 28, 16)               1296      ['prelumomentum_dump_1_1[0][0]']                            
 v1D)                                                                                             
 preluflux_trend_1_2 (PReLU  (None, 298, 16)              1         ['convflux_trend_1_2[0][0]']  
 )                                                                                                
 preluflux_periodogram_1_2   (None, 671, 16)              1         ['convflux_periodogram_1_2[0][0]']                          
 (PReLU)                                                                                          
 local_flux_split_oe (Split  [(None, 1, 25, 16),          0         ['local_flux_transit_merge_local_odd_even[0][0]']           
 Layer)                       (None, 1, 25, 16)]                                                                                    
 unfolded_flux_permute_merg  (None, 25, 16, 3)            0         ['unfolded_flux_merge[0][0]'] 
 e (Permute)                                                                                      
 convfc_diff_img (Conv1D)    (None, 1, 3)                 507858    ['flatten_wscalar_diff_img_imgsscalars[0][0]']              
 maxpooling_global_flux_1 (  (None, 287, 16)              0         ['preluglobal_flux_1_2[0][0]']
 MaxPooling1D)                                                                                    
 maxpooling_local_centroid_  (None, 25, 16)               0         ['prelulocal_centroid_1_2[0][0]']                           
 1 (MaxPooling1D)                                                                                 
 tce_dikco_msky_norm (Input  [(None, 1)]                  0         []                            
 Layer)                                                                                           
 tce_dikco_msky_err_norm (I  [(None, 1)]                  0         []                            
 nputLayer)                                                                                       
 ruwe_norm (InputLayer)      [(None, 1)]                  0         []                            
 mag_shift_norm (InputLayer  [(None, 1)]                  0         []                            
 )                                                                                                
 prelumomentum_dump_1_2 (PR  (None, 28, 16)               1         ['convmomentum_dump_1_2[0][0]']                             
 eLU)                                                                                             
 maxpooling_flux_trend_1 (M  (None, 295, 16)              0         ['preluflux_trend_1_2[0][0]'] 
 axPooling1D)                                                                                     
 flux_trend_global_stat_max  [(None, 1)]                  0         []                            
 _norm (InputLayer)                                                                               
 flux_trend_global_stat_min  [(None, 1)]                  0         []                            
 _norm (InputLayer)                                                                               
 maxpooling_flux_periodogra  (None, 668, 16)              0         ['preluflux_periodogram_1_2[0][0]']                         
 m_1 (MaxPooling1D)                                                                               
 pgram_smooth_max_power_nor  [(None, 1)]                  0         []                            
 m (InputLayer)                                                                                   
 pgram_tpm_smooth_max_power  [(None, 1)]                  0         []                            
 _norm (InputLayer)                                                                               
 subtract_oe (Subtract)      (None, 1, 25, 16)            0         ['local_flux_split_oe[0][0]', 
                                                                     'local_flux_split_oe[0][1]'] 
 flux_odd_local_stat_abs_mi  [(None, 1)]                  0         []                            
 n_norm (InputLayer)                                                                              
 flux_even_local_stat_abs_m  [(None, 1)]                  0         []                            
 in_norm (InputLayer)                                                                             
 tce_ptemp_stat_norm (Input  [(None, 1)]                  0         []                            
 Layer)                                                                                           
 tce_albedo_stat_norm (Inpu  [(None, 1)]                  0         []                            
 tLayer)                                                                                          
 tce_maxmes_norm (InputLaye  [(None, 1)]                  0         []                            
 r)                                                                                               
 flux_weak_secondary_local_  [(None, 1)]                  0         []                            
 stat_abs_min_norm (InputLa                                                                       
 yer)                                                                                             
 unfolded_flux_2convlocal_u  (None, 25, 16, 4)            16        ['unfolded_flux_permute_merge[0][0]']                       
 nfolded_flux_1 (Conv2D)                                                                          
 tce_num_transits_obs_norm   [(None, 1)]                  0         []                            
 (InputLayer)                                                                                     
 tce_num_transits_norm (Inp  [(None, 1)]                  0         []                            
 utLayer)                                                                                         
 convfc_prelu_diff_img (PRe  (None, 1, 3)                 3         ['convfc_diff_img[0][0]']     
 LU)                                                                                              
 flatten_global_flux (Flatt  (None, 4592)                 0         ['maxpooling_global_flux_1[0][0]']                          
 en)                                                                                              
 flux_global_stat_abs_min_n  [(None, 1)]                  0         []                            
 orm (InputLayer)                                                                                 
 flatten_local_centroid (Fl  (None, 400)                  0         ['maxpooling_local_centroid_1[0][0]']                       
 atten)                                                                                           
 local_centroid_scalar_inpu  (None, 4)                    0         ['tce_dikco_msky_norm[0][0]', 
 t (Concatenate)                                                     'tce_dikco_msky_err_norm[0][0]',                           
                                                                     'ruwe_norm[0][0]',           
                                                                     'mag_shift_norm[0][0]']      
 maxpooling_momentum_dump_1  (None, 25, 16)               0         ['prelumomentum_dump_1_2[0][0]']                            
  (MaxPooling1D)                                                                                  
 flatten_flux_trend (Flatte  (None, 4720)                 0         ['maxpooling_flux_trend_1[0][0]']                           
 n)                                                                                               
 flux_trend_scalar_input (C  (None, 2)                    0         ['flux_trend_global_stat_max_norm[0][0]',                   
 oncatenate)                                                         'flux_trend_global_stat_min_norm[0][0]']                   
 flatten_flux_periodogram (  (None, 10688)                0         ['maxpooling_flux_periodogram_1[0][0]']                     
 Flatten)                                                                                         
 flux_periodogram_scalar_in  (None, 2)                    0         ['pgram_smooth_max_power_norm[0][0]',                       
 put (Concatenate)                                                   'pgram_tpm_smooth_max_power_norm[0][0]']                   
 local_flux_flatten_local_o  (None, 400)                  0         ['subtract_oe[0][0]']         
 dd_even (Flatten)                                                                                
 local_flux_local_odd_even_  (None, 2)                    0         ['flux_odd_local_stat_abs_min_norm[0][0]',                  
 scalar_input (Concatenate)                                          'flux_even_local_stat_abs_min_norm[0][0]']                 
 local_flux_flatten_local_f  (None, 400)                  0         ['local_flux_split_merge[0][2]']                            
 lux (Flatten)                                                                                    
 flux_local_stat_abs_min (I  [(None, 1)]                  0         []                            
 nputLayer)                                                                                       
 local_flux_flatten_local_w  (None, 400)                  0         ['local_flux_split_merge[0][3]']                            
 eak_secondary (Flatten)                                                                          
 local_flux_local_weak_seco  (None, 4)                    0         ['tce_ptemp_stat_norm[0][0]', 
 ndary_scalar_input (Concat                                          'tce_albedo_stat_norm[0][0]',
 enate)                                                              'tce_maxmes_norm[0][0]',     
                                                                     'flux_weak_secondary_local_stat_abs_min_norm[0][0]']       
 unfolded_flux_flatten_loca  (None, 1600)                 0         ['unfolded_flux_2convlocal_unfolded_flux_1[0][0]']          
 l_unfolded_flux (Flatten)                                                                        
 local_unfolded_flux_scalar  (None, 2)                    0         ['tce_num_transits_obs_norm[0][0]',                         
 _input (Concatenate)                                                'tce_num_transits_norm[0][0]']                             
 flatten_convfc_diff_img (F  (None, 3)                    0         ['convfc_prelu_diff_img[0][0]']                             
 latten)                                                                                          
 tce_sdens_norm (InputLayer  [(None, 1)]                  0         []                            
 )                                                                                                
 tce_steff_norm (InputLayer  [(None, 1)]                  0         []                            
 )                                                                                                
 tce_smet_norm (InputLayer)  [(None, 1)]                  0         []                            
 tce_slogg_norm (InputLayer  [(None, 1)]                  0         []                            
 )                                                                                                
 tce_smass_norm (InputLayer  [(None, 1)]                  0         []                            
 )                                                                                                
 tce_sradius_norm (InputLay  [(None, 1)]                  0         []                            
 er)                                                                                              
 boot_fap_norm (InputLayer)  [(None, 1)]                  0         []                            
 tce_cap_stat_norm (InputLa  [(None, 1)]                  0         []                            
 yer)                                                                                             
 tce_hap_stat_norm (InputLa  [(None, 1)]                  0         []                            
 yer)                                                                                             
 tce_period_norm (InputLaye  [(None, 1)]                  0         []                            
 r)                                                                                               
 tce_max_mult_ev_norm (Inpu  [(None, 1)]                  0         []                            
 tLayer)                                                                                          
 tce_max_sngle_ev_norm (Inp  [(None, 1)]                  0         []                            
 utLayer)                                                                                         
 tce_robstat_norm (InputLay  [(None, 1)]                  0         []                            
 er)                                                                                              
 tce_model_chisq_norm (Inpu  [(None, 1)]                  0         []                            
 tLayer)                                                                                          
 tce_prad_norm (InputLayer)  [(None, 1)]                  0         []                            
 flatten_wscalar_global_flu  (None, 4593)                 0         ['flatten_global_flux[0][0]', 
 x (Concatenate)                                                     'flux_global_stat_abs_min_norm[0][0]']                     
 flatten_wscalar_local_cent  (None, 404)                  0         ['flatten_local_centroid[0][0]',                            
 roid (Concatenate)                                                  'local_centroid_scalar_input[0][0]']                       
 flatten_momentum_dump (Fla  (None, 400)                  0         ['maxpooling_momentum_dump_1[0][0]']                        
 tten)                                                                                            
 flatten_wscalar_flux_trend  (None, 4722)                 0         ['flatten_flux_trend[0][0]',  
  (Concatenate)                                                      'flux_trend_scalar_input[0][0]']                           
 flatten_wscalar_flux_perio  (None, 10690)                0         ['flatten_flux_periodogram[0][0]',                          
 dogram (Concatenate)                                                'flux_periodogram_scalar_input[0][0]']                     
 local_flux_flatten_wscalar  (None, 402)                  0         ['local_flux_flatten_local_odd_even[0][0]',                 
 _local_odd_even (Concatena                                          'local_flux_local_odd_even_scalar_input[0][0]']            
 te)                                                                                              
 local_flux_flatten_wscalar  (None, 401)                  0         ['local_flux_flatten_local_flux[0][0]',                     
 _local_flux (Concatenate)                                           'flux_local_stat_abs_min[0][0]']                           
 local_flux_flatten_wscalar  (None, 404)                  0         ['local_flux_flatten_local_weak_secondary[0][0]',           
 _local_weak_secondary (Con                                          'local_flux_local_weak_secondary_scalar_input[0][0]']      
 catenate)                                                                                        
 flatten_wscalar_local_unfo  (None, 1602)                 0         ['unfolded_flux_flatten_local_unfolded_flux[0][0]',         
 lded_flux (Concatenate)                                             'local_unfolded_flux_scalar_input[0][0]']                  
 flatten_wscalar_diff_img_s  (None, 4)                    0         ['flatten_convfc_diff_img[0][0]',                           
 calars (Concatenate)                                                'mag_shift_norm[0][0]']      
 stellar_scalar_input (Conc  (None, 6)                    0         ['tce_sdens_norm[0][0]',      
 atenate)                                                            'tce_steff_norm[0][0]',      
                                                                     'tce_smet_norm[0][0]',       
                                                                     'tce_slogg_norm[0][0]',      
                                                                     'tce_smass_norm[0][0]',      
                                                                     'tce_sradius_norm[0][0]']    
 dv_tce_fit_scalar_input (C  (None, 9)                    0         ['boot_fap_norm[0][0]',       
 oncatenate)                                                         'tce_cap_stat_norm[0][0]',   
                                                                     'tce_hap_stat_norm[0][0]',   
                                                                     'tce_period_norm[0][0]',     
                                                                     'tce_max_mult_ev_norm[0][0]',
                                                                     'tce_max_sngle_ev_norm[0][0]', 
                                                                     'tce_robstat_norm[0][0]',   
                                                                     'tce_model_chisq_norm[0][0]',
                                                                     'tce_prad_norm[0][0]']       
 fc_global_flux (Dense)      (None, 3)                    13782     ['flatten_wscalar_global_flux[0][0]']                       
 fc_local_centroid (Dense)   (None, 3)                    1215      ['flatten_wscalar_local_centroid[0][0]']                    
 fc_momentum_dump (Dense)    (None, 3)                    1203      ['flatten_momentum_dump[0][0]']                             
 fc_flux_trend (Dense)       (None, 3)                    14169     ['flatten_wscalar_flux_trend[0][0]']                        
 fc_flux_periodogram (Dense  (None, 3)                    32073     ['flatten_wscalar_flux_periodogram[0][0]']                  
 )                                                                                                
 local_flux_fc_local_odd_ev  (None, 3)                    1209      ['local_flux_flatten_wscalar_local_odd_even[0][0]']         
 en (Dense)                                                                                       
 local_flux_fc_local_flux (  (None, 3)                    1206      ['local_flux_flatten_wscalar_local_flux[0][0]']             
 Dense)                                                                                           
 local_flux_fc_local_weak_s  (None, 3)                    1215      ['local_flux_flatten_wscalar_local_weak_secondary[0][0]']   
 econdary (Dense)                                                                                 
 fc_local_unfolded_flux (De  (None, 3)                    4809      ['flatten_wscalar_local_unfolded_flux[0][0]']               
 nse)                                                                                             
 fc_diff_img (Dense)         (None, 3)                    15        ['flatten_wscalar_diff_img_scalars[0][0]']                  
 fc_stellar_scalar (Dense)   (None, 3)                    21        ['stellar_scalar_input[0][0]']
 fc_dv_tce_fit_scalar (Dens  (None, 3)                    30        ['dv_tce_fit_scalar_input[0][0]']                           
 e)                                                                                               
 fc_prelu_global_flux (PReL  (None, 3)                    1         ['fc_global_flux[0][0]']      
 U)                                                                                               
 fc_prelu_local_centroid (P  (None, 3)                    1         ['fc_local_centroid[0][0]']   
 ReLU)                                                                                            
 fc_prelu_momentum_dump (PR  (None, 3)                    1         ['fc_momentum_dump[0][0]']    
 eLU)                                                                                             
 fc_prelu_flux_trend (PReLU  (None, 3)                    1         ['fc_flux_trend[0][0]']       
 )                                                                                                
 fc_prelu_flux_periodogram   (None, 3)                    1         ['fc_flux_periodogram[0][0]'] 
 (PReLU)                                                                                          
 local_flux_fc_prelu_local_  (None, 3)                    1         ['local_flux_fc_local_odd_even[0][0]']                      
 odd_even (PReLU)                                                                                 
 local_flux_fc_prelu_local_  (None, 3)                    1         ['local_flux_fc_local_flux[0][0]']                          
 flux (PReLU)                                                                                     
 local_flux_fc_prelu_local_  (None, 3)                    1         ['local_flux_fc_local_weak_secondary[0][0]']                
 weak_secondary (PReLU)                                                                           
 fc_prelu_local_unfolded_fl  (None, 3)                    1         ['fc_local_unfolded_flux[0][0]']                            
 ux (PReLU)                                                                                       
 fc_prelu_diff_img (PReLU)   (None, 3)                    1         ['fc_diff_img[0][0]']         
 fc_prelu_stellar_scalar (P  (None, 3)                    1         ['fc_stellar_scalar[0][0]']   
 ReLU)                                                                                            
 fc_prelu_dv_tce_fit_scalar  (None, 3)                    1         ['fc_dv_tce_fit_scalar[0][0]']
  (PReLU)                                                                                         
 dropout_fc_conv_global_flu  (None, 3)                    0         ['fc_prelu_global_flux[0][0]']
 x (Dropout)                                                                                      
 dropout_fc_conv_local_cent  (None, 3)                    0         ['fc_prelu_local_centroid[0][0]']                           
 roid (Dropout)                                                                                   
 dropout_fc_conv_momentum_d  (None, 3)                    0         ['fc_prelu_momentum_dump[0][0]']                            
 ump (Dropout)                                                                                    
 dropout_fc_conv_flux_trend  (None, 3)                    0         ['fc_prelu_flux_trend[0][0]'] 
  (Dropout)                                                                                       
 dropout_fc_conv_flux_perio  (None, 3)                    0         ['fc_prelu_flux_periodogram[0][0]']                         
 dogram (Dropout)                                                                                 
 local_flux_dropout_fc_conv  (None, 3)                    0         ['local_flux_fc_prelu_local_odd_even[0][0]']                
 _local_odd_even (Dropout)                                                                        
 local_flux_dropout_fc_conv  (None, 3)                    0         ['local_flux_fc_prelu_local_flux[0][0]']                    
 _local_flux (Dropout)                                                                            
 local_flux_dropout_fc_conv  (None, 3)                    0         ['local_flux_fc_prelu_local_weak_secondary[0][0]']          
 _local_weak_secondary (Dro                                                                       
 pout)                                                                                            
 dropout_fc_conv_local_unfo  (None, 3)                    0         ['fc_prelu_local_unfolded_flux[0][0]']                      
 lded_flux (Dropout)                                                                              
 dropout_fc_diff_img (Dropo  (None, 3)                    0         ['fc_prelu_diff_img[0][0]']   
 ut)                                                                                              
 convbranch_wscalar_concat   (None, 36)                   0         ['fc_prelu_stellar_scalar[0][0]',                           
 (Concatenate)                                                       'fc_prelu_dv_tce_fit_scalar[0][0]',                        
                                                                     'dropout_fc_conv_global_flux[0][0]',                       
                                                                     'dropout_fc_conv_local_centroid[0][0]',                    
                                                                     'dropout_fc_conv_momentum_dump[0][0]',                     
                                                                     'dropout_fc_conv_flux_trend[0][0]',                        
                                                                     'dropout_fc_conv_flux_periodogram[0][0]',                  
                                                                     'local_flux_dropout_fc_conv_local_odd_even[0][0]',         
                                                                     'local_flux_dropout_fc_conv_local_flux[0][0]',             
                                                                     'local_flux_dropout_fc_conv_local_weak_secondary[0][0]',   
                                                                     'dropout_fc_conv_local_unfolded_flux[0][0]',               
                                                                     'dropout_fc_diff_img[0][0]'] 
 fc0 (Dense)                 (None, 512)                  18944     ['convbranch_wscalar_concat[0][0]']                         
 fc_prelu0 (PReLU)           (None, 512)                  1         ['fc0[0][0]']                 
 dropout_fc0 (Dropout)       (None, 512)                  0         ['fc_prelu0[0][0]']           
 fc1 (Dense)                 (None, 512)                  262656    ['dropout_fc0[0][0]']         
 fc_prelu1 (PReLU)           (None, 512)                  1         ['fc1[0][0]']                 
 dropout_fc1 (Dropout)       (None, 512)                  0         ['fc_prelu1[0][0]']           
 fc2 (Dense)                 (None, 512)                  262656    ['dropout_fc1[0][0]']         
 fc_prelu2 (PReLU)           (None, 512)                  1         ['fc2[0][0]']                 
 dropout_fc2 (Dropout)       (None, 512)                  0         ['fc_prelu2[0][0]']           
 fc3 (Dense)                 (None, 512)                  262656    ['dropout_fc2[0][0]']         
 fc_prelu3 (PReLU)           (None, 512)                  1         ['fc3[0][0]']                 
 dropout_fc3 (Dropout)       (None, 512)                  0         ['fc_prelu3[0][0]']           
 logits (Dense)              (None, 1)                    513       ['dropout_fc3[0][0]']         
 main (Activation)           (None, 1)                    0         ['logits[0][0]']              
==================================================================================================