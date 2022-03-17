# %% <initialize imageJ gateway>
import imagej
ij = imagej.init("/Applications/Fiji.app")
ij.getVersion() #should print '2.3.0/1.53f'

# %% <import modules>
import skimage as sk
import numpy as np
import matplotlib.pyplot as plt
from scyjava import jimport
import os
import scipy
import cv2
import pandas as pd
import math

#%% <add imageplus>
IJ=jimport("ij.IJ")
WindowManager =jimport("ij.WindowManager")
ImagePlus = jimport("ij.ImagePlus")

#%% <functions>
def open_image(img):
    jimage=ij.io().open(img) #as imageJ Dataset
    imp = ij.convert().convert(jimage, ImagePlus)
    return jimage, imp
def z_stack_projection(img):
    ZProjector = jimport("ij.plugin.ZProjector") #make sure its a ImagePlus Composite image
    projection_type= "max"
    z_proj = ZProjector.run (img, projection_type) #run z projection on max intensity
    return z_proj
def split_channels (img):
    split=img.splitChannels(True)
    chan_1=ij.py.from_java(split[0])
    chan_2=ij.py.from_java(split[1])
    return chan_1, chan_2
def denoise(img):
    denoised= sk.restoration.denoise_wavelet(img)
    return denoised
def filtering(img):
    rchannelGauss= sk.filters.gaussian(img, channel_axis=False)
    yen_thresh = rchannelGauss>=sk.filters.threshold_yen(rchannelGauss)
    filled= scipy.ndimage.binary_fill_holes(yen_thresh)
    fill_eroded = sk.morphology.binary_erosion(yen_thresh)
    return filled
def watersheding (img):
    distance = scipy.ndimage.distance_transform_edt(img)
    local_max_coords = sk.feature.peak_local_max(distance, min_distance=1,labels = scipy.ndimage.label(img)[0])
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True
    markers = sk.measure.label(local_max_mask)
    watersheded = sk.segmentation.watershed(-distance, markers, mask=img,watershed_line=True)
    return watersheded
def particle_analyzer (img):
    contours= sk.measure.find_contours(img, .8)
    labelarray, particle_count = scipy.ndimage.label(img)
    regions = sk.measure.regionprops(labelarray)
    props = sk.measure.regionprops_table(labelarray, properties=('area','extent','orientation','axis_major_length','axis_minor_length', 'coords', 'equivalent_diameter_area', 'perimeter'))
    props_table=pd.DataFrame(props)
    props_table['pixels_squared']= props_table['area']**2
    #filtered_df=props_table[props_table['area']>= 66095] #250 um , each pixel is 264.38 microns
    filtered_df=props_table[props_table['pixels_squared']>= 66095] #250 um , each pixel is 264.38 microns
    return filtered_df

#%% <define path and create list with file names>
#dir_path="/Users/romina/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/internship_1/test"
dir_path="/Users/romina/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/internship_1/python_images/5_months_mpfc"
all_files=[]
#type(all_files)

for filename in os.listdir(dir_path):
    if filename.endswith(".lsm"):
        all_files.append(filename)
#print(all_files)

#%% <create empty dataframe>
df= pd.DataFrame(columns= ['area','extent','orientation','axis_major_length','axis_minor_length', 'coords', 'equivalent_diameter_area', 'perimeter'])

#%% <analyze and create table>
for filename in all_files:
    try:
        image_path=dir_path + "/" + filename
        jimage, imp= open_image(image_path)
        z_proj= z_stack_projection (imp)
        red_chan, green_chan= split_channels(z_proj)
        img_denoised= denoise(red_chan)
        img_filtered= filtering(img_denoised)
        #img_watersheded= watersheding(img_filtered)
        particle_analysis_table = particle_analyzer(img_filtered)
        particle_analysis_table["image_id"]= str(filename)
        try:
            jimage.close()
            imp.close()
        except:
            print("not closed")
        df=df.append(particle_analysis_table)
    except:
        print(filename + " not readable")

df.to_csv("/Users/romina/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/internship_1/python_images/5_months_analyzed.csv")

 
#%% <open one image>
file=all_files[6]
image_path=dir_path + "/" + file
jimage, imp= open_image(image_path)
z_proj= z_stack_projection (imp)
red_chan, green_chan= split_channels(z_proj)
img_denoised= denoise(red_chan)
img_filtered= filtering(img_denoised)
print(image_path)
ij.py.show(img_filtered, cmap="gray")
