import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import os

directory_csv = "C:\\Users\\wzhong\\Downloads\\diff_img_quality_metric"



# Important note: This code only adds data for samples with valid difference images. Samples without valid images should have quality metrics set to 0 and the other features be filled with np.nan



allcsvs = []
for file in os.listdir(directory_csv):
    filepath = directory_csv + "/" + file
    print(file)
    currdict = pd.read_csv(filepath)
    allcsvs.append(currdict)

quality_metrics = pd.concat(allcsvs)

directory = "C:\\Users\\wzhong\\Downloads\\data"

alldicts = []
files_got = []
ev = 0
for file in os.listdir(directory):
    #if ev%2 == 0:``
    #    ev += 1
    #    continue
    filepath = directory + "/" + file
    print(file)
    files_got.append(file)
    currdict = np.load(filepath, allow_pickle=True).item()
    alldicts.append(currdict)
    ev += 1

dall = {}
for d in alldicts:
    dall.update(d)

saturated = pd.read_csv("C:\\Users\\wzhong\\Downloads\\tess_tces_dv_s1-s55_10-05-2022_1338_ticstellar_ruwe_tec_tsoebs_ourmatch_preproc.csv")

saturated['target_id'] = saturated['target_id'].astype(str)
saturated['tce_plnt_num'] = saturated['tce_plnt_num'].astype(str)
saturated['sector_run'] = saturated['sector_run'].astype(str)
saturated['name'] = saturated.target_id.str.cat(saturated.tce_plnt_num, sep='-')
saturated['name'] = saturated['name'].astype(str) + "-S"
saturated['name'] = saturated.name.str.cat(saturated.sector_run, sep='')
saturatedids = saturated[saturated.mag < 7]['name'].tolist() # 7 instead of 12 for TESS

# replace all negative values with nan for option 4
for tce_uid, tce_data in dall.items():
    for i in range(len(tce_data['image_data'])):
        curr_img = tce_data['image_data'][i][:,:,1,0] # oot check negatives
        curr_img[curr_img < 0] = np.nan # set to nan
        tce_data['image_data'][i][:,:,1,0] = curr_img

bad_c = 0

missing_diff = []

sample_count = 0

# split into smaller functions for each preprocessing step when converted into python script

number_of_quarters_to_sample = 5
pad_images_number_of_pixels = 20
cropped_dims = 11

cnt = 0
for tce_uid, tce_data in dall.items(): # tce_uid is id, tce_data is data
    print(tce_uid)
    # make a new dictionary key for cropped images
    #quarter numbers will correspond with cropped_imgs so we can access each cropped image's quarter number if needed
    tce_id = tce_uid
    # get the list of valid quarters
    #tce_id = '123233041-1-S40'
    #tce_id = '237906138-1-S1-36'
    sector_run = tce_id.split('S')[-1]
    filepath = directory_csv + "/" + sector_run + ".csv"
    quality_metrics = pd.read_csv(filepath)

    all_sectors = []
    all_sectors_num = []

    valid_sectors = []
    valid_sectors_num = []
    for col in list(quality_metrics.columns):
        if col.count('valid') == 0:
            continue
        # gets all sectors
        if quality_metrics.loc[quality_metrics['uid'] == tce_id][col].item() == True or quality_metrics.loc[quality_metrics['uid'] == tce_id][col].item() == False:
            all_sectors.append(col)
            all_sectors_num.append(re.findall(r'\d+', col))


        # gets the valid sectors only
        if quality_metrics.loc[quality_metrics['uid'] == tce_id][col].item() == True:
            valid_sectors.append(col)
            valid_sectors_num.append(re.findall(r'\d+', col))

    print(f'all sectors: {all_sectors_num}')
    print(f'valid sectors: {valid_sectors_num}')
    valid_quarters = []

    if sector_run.count('-') > 0: # multiple
        start_index = int(sector_run.split('-')[0])
        end_index = int(sector_run.split('-')[-1])
        curr_index = 0
        # goes through all diff imgs for this sample
        for target_ref_q in dall[tce_id]['target_ref_centroid']:
            #key of quality metric, could be true or false, we want the true ones only
            key_validity = all_sectors[curr_index]
            #print(target_ref_q['col']['uncertainty'])
            #print(quality_metrics.loc[quality_metrics['uid'] == tce_id][key_validity].item())

            if target_ref_q['col']['uncertainty'] != -1 and (quality_metrics.loc[quality_metrics['uid'] == tce_id][key_validity].item() == True):
                valid_quarters.append(curr_index)
            curr_index = curr_index + 1
    else:
        curr_index = 0
        for target_ref_q in dall[tce_id]['target_ref_centroid']:
            #key of quality metric, could be true or false, we want the true ones only
            key_validity = all_sectors[curr_index]
            #print(target_ref_q['col']['uncertainty'])
            #print(quality_metrics.loc[quality_metrics['uid'] == tce_id][key_validity].item())

            if target_ref_q['col']['uncertainty'] != -1 and (quality_metrics.loc[quality_metrics['uid'] == tce_id][key_validity].item() == True):
                valid_quarters.append(curr_index)
            curr_index = curr_index + 1    
    print(f'ids: {valid_quarters}')
    
    # if saturated then skip
    
    # if no valid quarters
    if len(valid_quarters) == 0:
        # difference image add nans
        missing_diff.append(tce_uid)
        #print('remove')
        del(quality_metrics)
        continue
        
    dall[tce_uid].update({'cropped_diff_images' : {
        'cropped_imgs' : {'img' : [], 'x' : [], 'y' : [], 'sub_x' : [], 'sub_y' : [], 'quality' : []}
      , 'quarter_numbers' : []
    }}
    )
    
    dall[tce_uid].update({'out_of_transit_images' : {
        'cropped_imgs' : {'img' : [], 'x' : [], 'y' : [], 'sub_x' : [], 'sub_y' : [], 'quality' : []}
      , 'quarter_numbers' : []
    }}
    )
    
    # randomly sample valid quarters
    random_valid_quarters_weights = []
    
    random_sample = []
    if len(valid_quarters) < number_of_quarters_to_sample:
        random_sample = [random.choice(valid_quarters) for _ in range(number_of_quarters_to_sample)]
    else:
        random_sample = random.sample(valid_quarters,number_of_quarters_to_sample)
    print(f'random sample of sectors: {[all_sectors_num[quarter][0] for quarter in random_sample]}')
    
    #want to use index to find sector number for that index which finds quality metric
    
    curr_tce_df = quality_metrics.loc[quality_metrics['uid'] == tce_id]
    cc = 0
    #print(valid_sectors_num)
    
    for quarter in random_sample:
        key_value = 's' + all_sectors_num[quarter][0] + '_value'
        random_valid_quarters_weights.append(curr_tce_df[key_value].item())
    
    print(f'random sample: {random_valid_quarters_weights}')
    
    # sort list
    #random_sample.sort()
    #print(f'sorted random sample: {random_sample}')
    
    
    # go through list
    images = tce_data['image_data']
    
    #diff img
    counter = 0
    for quarter in random_sample:
        # apply algorithm
        curr_img = images[quarter][:,:,2,0] # only diff img
        #plt.imshow(curr_img)
        padded = np.pad(curr_img, pad_images_number_of_pixels, mode='constant', constant_values=np.nan) # pad with nan
        #s = np.isnan(padded)
        #padded[s]=0
        
        x = tce_data['target_ref_centroid'][quarter]['col']['value']
        y = tce_data['target_ref_centroid'][quarter]['row']['value']
        target_x_rounded = round(x)
        target_y_rounded = round(y)
        center_x = round(x)
        center_y = round(y)
        # center is shifted 20 on both axis because it's padded by 20
        padded_pos_x = center_x + pad_images_number_of_pixels
        padded_pos_y = center_y + pad_images_number_of_pixels
        
        # create a new image matrix
        # crop image to be centered around target pixel
        side_pixel_cnt = cropped_dims//2
        cropped_img = padded[padded_pos_y-side_pixel_cnt:padded_pos_y+side_pixel_cnt+1,
                             padded_pos_x-side_pixel_cnt:padded_pos_x+side_pixel_cnt+1]
        # we know the target should be in center pixel (5,5)
        # this uses the subpixel coordinates of the target in the target pixel
        # to locate the correct subpixel position in the new center (5,5)
        final_x = side_pixel_cnt - (target_x_rounded - x)
        final_y = side_pixel_cnt - (target_y_rounded - y)
        #final_x = x - target_x_rounded
        #final_y = y - target_y_rounded                
        # add to dictionary
        dall[tce_uid]['cropped_diff_images']['cropped_imgs']['img'].append(cropped_img)
        
        dall[tce_uid]['cropped_diff_images']['cropped_imgs']['x'].append(final_x)
        dall[tce_uid]['cropped_diff_images']['cropped_imgs']['y'].append(final_y)
        
        #subpixel coordinates
        dall[tce_uid]['cropped_diff_images']['cropped_imgs']['sub_x'].append((x - target_x_rounded))
        dall[tce_uid]['cropped_diff_images']['cropped_imgs']['sub_y'].append((y - target_y_rounded))
        
        #key_value = 's' + str(quarter+start_index) + '_value'
        
        #quality
        dall[tce_uid]['cropped_diff_images']['cropped_imgs']['quality'].append(random_valid_quarters_weights[counter])
        counter += 1
        # add current quarter number to dictionary
        dall[tce_uid]['cropped_diff_images']['quarter_numbers'].append(quarter)
    
    #out of transit img
    counter = 0
    for quarter in random_sample:
        # apply algorithm
        curr_img = images[quarter][:,:,1,0] # only out of transit img
        #plt.imshow(curr_img)
        padded = np.pad(curr_img, pad_images_number_of_pixels, mode='constant', constant_values=np.nan) # pad with nan
        
        #s = np.isnan(padded)
        #padded[s]=-1 # should be -1 if nan
        
        x = tce_data['target_ref_centroid'][quarter]['col']['value']
        y = tce_data['target_ref_centroid'][quarter]['row']['value']
        target_x_rounded = round(x)
        target_y_rounded = round(y)
        center_x = round(x)
        center_y = round(y)
        # center is shifted 20 on both axis because it's padded by 20
        padded_pos_x = center_x + pad_images_number_of_pixels
        padded_pos_y = center_y + pad_images_number_of_pixels
        
        # create a new image matrix
        # crop image to be centered around target pixel
        side_pixel_cnt = cropped_dims//2
        cropped_img = padded[padded_pos_y-side_pixel_cnt:padded_pos_y+side_pixel_cnt+1,
                             padded_pos_x-side_pixel_cnt:padded_pos_x+side_pixel_cnt+1]
        # we know the target should be in center pixel (5,5)
        # this uses the subpixel coordinates of the target in the target pixel
        # to locate the correct subpixel position in the new center (5,5)
        final_x = side_pixel_cnt - (target_x_rounded - x)
        final_y = side_pixel_cnt - (target_y_rounded - y)
        
        # add to dictionary
        dall[tce_uid]['out_of_transit_images']['cropped_imgs']['img'].append(cropped_img)
        
        dall[tce_uid]['out_of_transit_images']['cropped_imgs']['x'].append(final_x)
        dall[tce_uid]['out_of_transit_images']['cropped_imgs']['y'].append(final_y)
        
        #subpixel coordinates
        dall[tce_uid]['out_of_transit_images']['cropped_imgs']['sub_x'].append((x - target_x_rounded))
        dall[tce_uid]['out_of_transit_images']['cropped_imgs']['sub_y'].append((y - target_y_rounded))
        
        #key_value = 's' + str(quarter+start_index) + '_value'
        
        #quality
        dall[tce_uid]['out_of_transit_images']['cropped_imgs']['quality'].append(random_valid_quarters_weights[counter])
        counter += 1
        # add current quarter number to dictionary
        dall[tce_uid]['out_of_transit_images']['quarter_numbers'].append(quarter)
    cnt = cnt + 1
    del(quality_metrics)
    del(curr_tce_df)
    print(cnt)

# adjust saturated tces
for sat in saturatedids:
    print(sat)
    if sat not in dall.keys():
        continue
    # set diff imgs to 0
    if 'cropped_diff_images' not in list(dall[sat].keys()):
        print("bad")
        continue
        
    print(sat)
    for i in range(len(dall[sat]['cropped_diff_images']['cropped_imgs']['img'])):
        img = dall[sat]['cropped_diff_images']['cropped_imgs']['img'][i]
        img[:][:] = np.nan # nan instead of 0
        dall[sat]['cropped_diff_images']['cropped_imgs']['img'][i] = img
    
    # set all target locations to 0
    #for i in range(len(dall[sat]['cropped_diff_images']['cropped_imgs']['x'])):
    #    dall[sat]['cropped_diff_images']['cropped_imgs']['x'][i] = 0
    #for i in range(len(dall[sat]['cropped_diff_images']['cropped_imgs']['y'])):
    #    dall[sat]['cropped_diff_images']['cropped_imgs']['y'][i] = 0
    for i in range(len(dall[sat]['cropped_diff_images']['cropped_imgs']['sub_x'])):
        dall[sat]['cropped_diff_images']['cropped_imgs']['sub_x'][i] = 0
    for i in range(len(dall[sat]['cropped_diff_images']['cropped_imgs']['sub_y'])):
        dall[sat]['cropped_diff_images']['cropped_imgs']['sub_y'][i] = 0
        
    # set out of transit to -1
    for i in range(len(dall[sat]['out_of_transit_images']['cropped_imgs']['img'])):
        img = dall[sat]['out_of_transit_images']['cropped_imgs']['img'][i]
        img[:][:] = np.nan # nan instead of -1
        dall[sat]['out_of_transit_images']['cropped_imgs']['img'][i] = img
    
    # set all target locations to 0
    #for i in range(len(dall[sat]['out_of_transit_images']['cropped_imgs']['x'])):
    #    dall[sat]['out_of_transit_images']['cropped_imgs']['x'][i] = 0
    #for i in range(len(dall[sat]['out_of_transit_images']['cropped_imgs']['y'])):
    #    dall[sat]['out_of_transit_images']['cropped_imgs']['y'][i] = 0
    for i in range(len(dall[sat]['out_of_transit_images']['cropped_imgs']['sub_x'])):
        dall[sat]['out_of_transit_images']['cropped_imgs']['sub_x'][i] = 0
    for i in range(len(dall[sat]['out_of_transit_images']['cropped_imgs']['sub_y'])):
        dall[sat]['out_of_transit_images']['cropped_imgs']['sub_y'][i] = 0

np.save("corrected_only_valid_option_2_4_8_9.npy", dall)