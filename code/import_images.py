#import libraries
import os
import numpy as np
import pandas as pd
from PIL import Image


## import images
images_path = '/home/jacks.local/cpatten/ContrastiveLearning_NBIDE/NBIDE/x3p_dataframe/'
im_names = [f for f in os.listdir(images_path) if f.endswith('.csv')]
images_dict = {}

for im in im_names:
    image = pd.read_csv(os.path.join(images_path, im), encoding="latin_1")
    images_dict[im.replace('.csv','')] = image.pivot(index='y', columns='x', values='value')
    
    
## import info   
info_path = '/home/jacks.local/cpatten/ContrastiveLearning_NBIDE/NBIDE/info.csv'
info = pd.read_csv(info_path, encoding="latin_1")


## get min and max_range values images
min = 1
range = 0
for val in images_dict.values():
    temp_min = val.min(axis=None, skipna=True)
    temp_range = val.max(axis=None, skipna=True) - temp_min
    if temp_min < min:
        min = temp_min
    if temp_range > range:
        range = temp_range


## get shifting and scalign values for images
scale = 255/range - 1
shift = 1 - min*scale


## get images into a uniform size, normalized around 0 with std 1
rng = np.random.default_rng()
ecdf_224 = []
ecdf_512 = []

for spec in info['Specimen']:
    image = images_dict[spec] #grab the image with name 'spec'
    image_numpy = image.to_numpy() #convert to numpy
    mask = np.isnan(image_numpy) #create mask over nan values

    masked_numpy = image_numpy.copy() #copy this array
    masked_numpy[mask] = rng.choice(image_numpy[~mask].flatten(), size=mask.sum()) #replace 

    transform_numpy = (masked_numpy*scale)+shift #shift the values to valid pixels
    pillow_224 = Image.fromarray(transform_numpy).resize((224,224)) #convert to 224x224 image
    pillow_512 = Image.fromarray(transform_numpy).resize((512,512)) #convert to 512x512 image
    
    numpy_224 = np.asarray(pillow_224) #convert to numpy
    numpy_512 = np.asarray(pillow_512) #convert to numpy

    normalized_224 = (numpy_224-np.mean(numpy_224))/np.std(numpy_224) #normalize
    normalized_512 = (numpy_512-np.mean(numpy_512))/np.std(numpy_512) #normalize

    ecdf_224.append(np.asarray(normalized_224)) #add to list of images
    ecdf_512.append(np.asarray(normalized_512)) #add to list of images 

ecdf_224 = np.asarray(ecdf_224)
ecdf_512 = np.asarray(ecdf_512)


## save images
np.save('/home/jacks.local/cpatten/ContrastiveLearning_NBIDE/images/ecdf_224.npy', ecdf_224)
np.save('/home/jacks.local/cpatten/ContrastiveLearning_NBIDE/images/ecdf_512.npy', ecdf_512)