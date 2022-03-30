# %% <initialize imageJ gateway>
import imagej
ij = imagej.init("/Applications/Fiji.app")
ij.getVersion() #should print '2.3.0/1.53f'

# %% <import modules>
import skimage as sk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scyjava import jimport
import os
import scipy
import cv2
import pandas as pd
import math
from scipy import ndimage as ndi

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
    thresh = rchannelGauss>=sk.filters.threshold_yen(rchannelGauss)
    mask= ndi.median_filter(thresh, 10) >0
    filled= scipy.ndimage.binary_fill_holes(mask)
    return filled
def watersheding (img):
    distance = scipy.ndimage.distance_transform_edt(img)
    local_max_coords = sk.feature.peak_local_max(distance, min_distance=1,labels = scipy.ndimage.label(img)[0])
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True
    markers = sk.measure.label(local_max_mask)
    watersheded = sk.segmentation.watershed(-distance, markers, mask=img,watershed_line=True)
    return watersheded
def find_edges(img):
    weights= [[1, 1, 1], [0,0,0], [-1,-1,-1]]
    edges=ndi.convolve(img, weights)
    return edges
def particle_analyzer (img):
    contours= sk.measure.find_contours(img, .8)
    label_image= sk.measure.label(img)
    regions = sk.measure.regionprops(label_image)
    props = sk.measure.regionprops_table(label_image, properties=('area', 'area_bbox', 'area_convex','area_filled','axis_major_length','axis_minor_length', 'equivalent_diameter_area', 'extent','perimeter'))
    props_table=pd.DataFrame(props)
    props_table['area_um']= props_table['area_filled']/(1.7067**2)*1.1377 #by scaling factor (image is scaled by 0.5)
    filtered_df=props_table[props_table['area_um']>= 250] #250 um
    return filtered_df
def show_labels(img, img_original):
    label_image= sk.measure.label(img)
    image_label_overlay = sk.color.label2rgb(label_image, image=img, bg_label=0)
    f, (ax1, ax2)=plt.subplots(1,2)
    ax1.imshow(img_original, cmap="gray")
    ax2.imshow(image_label_overlay)

    for region in sk.measure.regionprops(label_image):
        # take regions with large enough areas
        if region.area >= (250*2.19*1.1377):
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax2.add_patch(rect)

    ax1.set_axis_off()
    ax2.set_axis_off()
    plt.tight_layout()
    plt.show()

def save_labels(img, img_original, save_path, name):
    label_image= sk.measure.label(img)
    image_label_overlay = sk.color.label2rgb(label_image, image=img, bg_label=0)
    f, (ax1, ax2)=plt.subplots(1,2)
    ax1.imshow(img_original, cmap="gray")
    ax2.imshow(image_label_overlay)

    for region in sk.measure.regionprops(label_image):
        # take regions with large enough areas
        if region.area >= (250*2.19*1.1377):
            # draw rectangle around segmented plaques
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax2.add_patch(rect)

    ax1.set_axis_off()
    ax2.set_axis_off()
    plt.tight_layout()
    plt.savefig(save_path+"/"+str(name)+".png")

#%% <define path and create list with file names>
dir_path="/Users/python_images/5_months_mpfc"
all_files=[]

for filename in os.listdir(dir_path):
    if filename.endswith(".lsm"):
        all_files.append(filename)


#%% <create empty dataframe>
df= pd.DataFrame(columns= ['area', 'area_bbox', 'area_convex','area_filled','axis_major_length','axis_minor_length', 'equivalent_diameter_area', 'extent','perimeter'])

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
            imp.close()
        except:
            print("not closed")
        df=df.append(particle_analysis_table)
    except:
        print(filename + " not readable")

df.to_csv("/Users/python_images/5_months_analyzed_scaled.csv")

#%% <open one image>

#%%<save plots>
save_path="/Users/python_images/filt_images"

for filename in all_files:
    try:
        name=filename.strip(".lsm")
        image_path=dir_path + "/" + filename
        jimage, imp= open_image(image_path)
        z_proj= z_stack_projection (imp)
        red_chan, green_chan= split_channels(z_proj)
        img_denoised= denoise(red_chan)
        img_filtered= filtering(img_denoised)
        save_labels(img_filtered, red_chan, save_path, name)
        try:
            imp.close()
        except:
            print("not closed")
    except:
        print(filename + " not readable")
