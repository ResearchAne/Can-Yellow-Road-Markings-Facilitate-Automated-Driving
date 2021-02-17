#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This is an example of code used to produce different color space representations
# and histograms for the paper "Camera-Based Lane Detection - Can Yellow Road Marking Facilitate Automated Driving in Snow?"
# Authors: Ane Dalsnes Storsæter, Kelly Pitera and Edward McCormack, submitted to the Journal of Field Robotics in D
# December 2020. The aim of the paper was to investigate if yellow road marking could be beneficial for camera-based
# lane detection. Code by Ane Dalsnes Storsæter.

# Import libraries
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math


# In[2]:


# Setting title to make it easier to find and sort the images that are produced. Change this title when changing images 
# or if you don't want to ovewrite previously produced images and plots.
title= "Case_1a_test_"

# Reading image from local folder, replace with own location.
img = mpimg.imread('testImages/1a_croppedROI.jpg')

# Addressing the three color channels to be used to show and plot the respective color channels.
R = img[:,:,0]
G = img[:,:,1]
B = img[:,:,2]

# Finding the shape of the image, used to do simple slicing.
h, w, c = img.shape
print('width:  ', w)
print('height: ', h)
print('channel:', c)

# Removing axis on image, showing image and saving image. Set the save location to a folder that exists and update the 
# savefig-location.
plt.axis('off')
plt.imshow(img) 
plt.savefig('C:/Cases/'+title+'RGB'+'.jpg', dpi=300, bbox_inches='tight', pad_inches=0)


# In[3]:


# For Case 1a, the image is not sliced in half, instead the pixel row to separate white from yellow markings 
# is set manually. Creating upper half of image (yellow marking).
sub_image = img[0:430,:]

# Showing upper part of image
plt.axis('off')
plt.imshow(sub_image)


# In[4]:


# The lower part of the image is also manually set in this case. 
sub_image2 = img[500:1344,:]

# Showing lower part of image (white marking)
plt.axis('off')
plt.imshow(sub_image2)


# In[5]:


# In Case 1a the upper and lower part of the image have different color marking.
# As the difference between these two markings, in terms of visibility in color spaces and histograms,
# is the aim of the paper the upper and lower parts of the image are plotted separately to look at differences.

# Histograms are found by using the np.sum function and plotted for upper (yellow) and lower (white) parts
# of the image for the three different color spaces of the RGB color space.

Rlow = np.sum(R[500:1344,:],axis=0)
Rhigh = np.sum(R[0:430,:],axis=0)
Glow = np.sum(G[500:1344,:],axis=0)
Ghigh = np.sum(G[0:430,:],axis=0)
Blow = np.sum(B[500:1344,:],axis=0)
Bhigh = np.sum(B[0:430,:],axis=0)
plt.plot(Rhigh, color = 'red', label ='RGB-R_yellow')
plt.plot(Ghigh, color = 'green', label ='RGB-G_yellow')
plt.plot(Bhigh, color = 'blue', label ='RGB-B_yellow')
plt.plot(Rlow, color = 'orange', label ='RGB-R_white')
plt.plot(Glow, color = 'lightgreen', label ='RGB-G_white')
plt.plot(Blow, color = 'lightblue', label ='RGB-B_white')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('C:/Cases/'+ title+'RGBhisthighlow'+'.jpg', dpi=300, bbox_inches='tight', pad_inches=0)


# In[6]:


# Plotting only the upper part of the image, i.e. yellow marking.

plt.plot(Rhigh, color = 'red', label ='RGB-R_yellow')
plt.plot(Ghigh, color = 'green', label ='RGB-G_yellow')
plt.plot(Bhigh, color = 'blue', label ='RGB-B_yellow')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('C:/Cases/'+ title+'RGBhisthigh'+'.jpg', dpi=300, bbox_inches='tight', pad_inches=0)


# In[7]:


# Plotting only the lower part of the image, i.e. white marking.

plt.plot(Rlow, color = 'red', label ='RGB-R')
plt.plot(Glow, color = 'green', label ='RGB-G')
plt.plot(Blow, color = 'blue', label ='RGB-B')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('C:/Cases/'+title+'RGBhistlow'+'.jpg', dpi=300, bbox_inches='tight', pad_inches=0)


# In[8]:


# Highlighting the channels that give the most distinct peaks at the position of the road markings.

plt.plot(Rhigh, color = 'red', label ='RGB-R yellow')
#plt.plot(Ghigh, color = 'green', label ='RGB-G_yellow')
#plt.plot(Bhigh, color = 'blue', label ='RGB-B_yellow')
plt.plot(Rlow, color = 'orange', label ='RGB-R white')
#plt.plot(Glow, color = 'green', label ='RGB-G')
#plt.plot(Blow, color = 'blue', label ='RGB-B')
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.legend()
plt.savefig('C:/Cases/'+title+'Rhist_highlow'+'.jpg', dpi=300, bbox_inches='tight', pad_inches=0)


# In[9]:


#plt.plot(Rhigh, color = 'red', label ='RGB-R yellow')
plt.plot(Ghigh, color = 'green', label ='RGB-G yellow')
#plt.plot(Bhigh, color = 'blue', label ='RGB-B_yellow')
#plt.plot(Rlow, color = 'orange', label ='RGB-R white')
plt.plot(Glow, color = 'lightgreen', label ='RGB-G white')
#plt.plot(Blow, color = 'blue', label ='RGB-B')
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.legend()
plt.savefig('C:/Cases/'+title+'Ghist_highlow'+'.jpg', dpi=300, bbox_inches='tight', pad_inches=0)


# In[10]:


#plt.plot(Rhigh, color = 'red', label ='RGB-R yellow')
#plt.plot(Ghigh, color = 'green', label ='RGB-G_yellow')
plt.plot(Bhigh, color = 'blue', label ='RGB-B yellow')
#plt.plot(Rlow, color = 'orange', label ='RGB-R white')
#plt.plot(Glow, color = 'green', label ='RGB-G')
plt.plot(Blow, color = 'lightblue', label ='RGB-B white')
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.legend()
plt.savefig('C:/Cases/'+title+'Bhist_highlow'+'.jpg', dpi=300, bbox_inches='tight', pad_inches=0)


# In[11]:


# Converting the RGB image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Showing image and saving image
plt.axis('off')
plt.imshow(gray, cmap='gray')
print(gray.shape)
plt.savefig('testImages/gray.jpg')
plt.savefig('C:/Cases/'+title+'gray'+'.jpg', dpi=300, bbox_inches='tight', pad_inches=0)


# In[12]:


# Making histogram plots for the upper and lower part of the image for the grayscale representation
histogramgraylow = np.sum(gray[0:430,:],axis=0)
histogramgrayhigh = np.sum(gray[500:1344,:],axis=0)
plt.plot(histogramgraylow, color = 'gray', label = 'Gray white')
plt.plot(histogramgrayhigh, color = 'black', label = 'Gray yellow')

plt.legend()
plt.savefig('C:/LD/Cases/'+title+'grayhist_highlow'+'.jpg', dpi=300, bbox_inches='tight', pad_inches=0)


# In[13]:


# Converting the RGB image to HSL color space, OpenCV uses the order HLS.
hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

# Naming the three color channels of the HLS representation
H = hls[:,:,0]
L = hls[:,:,1]
S = hls[:,:,2]

# Finding the shape of the image
he, wi, ch = hls.shape
print('width:  ', wi)
print('height: ', he)
print('channel:', ch)

# Showing and saving the HLS representation
plt.imshow(hls) 
plt.axis('off')
plt.savefig('C:/Cases/'+title+'HLS'+'.jpg', dpi=300, bbox_inches='tight', pad_inches=0)


# In[14]:


# Separating the upper and lower part of the image to produce histogram plots for the lower and upper part 
# of the image.

Hlow = np.sum(H[500:1344,:],axis=0)
Hhigh = np.sum(H[0:430,:],axis=0)
Llow = np.sum(L[500:1344,:],axis=0)
Lhigh = np.sum(L[0:430,:],axis=0)
Slow = np.sum(S[500:1344,:],axis=0)
Shigh = np.sum(S[0:430,:],axis=0)

# Plotting the histograms for the lower part of the image (white)
plt.plot(Hlow, color = 'blue', label ='HSL-H')
plt.plot(Llow, color = 'green', label ='HSL-L')
plt.plot(Slow, color = 'red', label ='HSL-S')
#plt.legend()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('C:/Cases/'+title+'HLShistlow'+'.jpg', dpi=300, bbox_inches='tight', pad_inches=0)


# In[15]:


# Plotting the histograms for the L-channel from the HSL image, comparing the white marking (lower part of image)
# to the yellow marking (upper part of image).

plt.plot(Llow, color = 'lightgreen', label ='HSL-L white')
plt.plot(Lhigh, color = 'green', label ='HSL-L yellow')

plt.legend()
plt.savefig('C:/Cases/'+title+'HSL-Lhist_highlow'+'.jpg', dpi=300, bbox_inches='tight', pad_inches=0)


# In[16]:


# Plotting the histograms for the H-channel from the HSL image, comparing the white marking (lower part of image)
# to the yellow marking (upper part of image).

plt.plot(Hlow, color = 'lightblue', label ='HSL-H white')
plt.plot(Hhigh, color = 'blue', label ='HSL-H yellow')

plt.legend()
plt.savefig('C:/Cases/'+title+'HSL-H_histhighlow'+'.jpg', dpi=300, bbox_inches='tight', pad_inches=0)


# In[17]:


# Plotting the histograms for the S-channel from the HSL image, comparing the white marking (lower part of image)
# to the yellow marking (upper part of image).

plt.plot(Slow, color = 'orange', label ='HSL-S white')
plt.plot(Shigh, color = 'red', label ='HSL-S yellow')

plt.legend()
plt.savefig('C:/Cases/'+title+'HSL-Shistlow'+'.jpg', dpi=300, bbox_inches='tight', pad_inches=0)


# In[18]:


# Converting the RGB image to HSV color space and naming the three color channels

hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
H2 = hsv[:,:,0]
S2 = hsv[:,:,1]
V = hsv[:,:,2]

# Showing and saving the HLS representation
plt.imshow(hsv) 
plt.axis('off')
plt.savefig('C:/Cases/'+title+'HSV'+'.jpg', dpi=300, bbox_inches='tight', pad_inches=0)


# In[19]:


# Separating the white (low) from the yellow (high) in the histogram plots

H2low = np.sum(H2[500:1344,:],axis=0)
H2high = np.sum(H2[0:430,:],axis=0)
Vlow = np.sum(V[500:1344,:],axis=0)
Vhigh = np.sum(V[0:430,:],axis=0)
S2low = np.sum(S2[500:1344,:],axis=0)
S2high = np.sum(S2[0:430,:],axis=0)

plt.plot(H2low, color = 'blue', label ='HSV-H')
plt.plot(Vlow, color = 'green', label ='HSV-V')
plt.plot(S2low, color = 'red', label ='HSV-S')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('C:/Cases/'+title+'HSVhistlow'+'.jpg', dpi=300, bbox_inches='tight', pad_inches=0)


# In[20]:


# Plotting the histograms for the H-channel from the HSV image, comparing the white marking (lower part of image)
# to the yellow marking (upper part of image).

plt.plot(H2low, color = 'lightblue', label ='HSV-H white')
plt.plot(H2high ,color='blue', label ='HSV-H yellow')

plt.legend()
plt.savefig('C:/Cases/'+title+'HSV-Hhist_highlow'+'.jpg', dpi=300, bbox_inches='tight', pad_inches=0)


# In[21]:


# Plotting the histograms for the S-channel from the HSV image, comparing the white marking (lower part of image)
# to the yellow marking (upper part of image).

plt.plot(S2low, color = 'orange', label ='HSV-S white')
plt.plot(S2high ,color='red', label ='HSV-S yellow')

plt.legend()
plt.savefig('C:/Cases/'+title+'HSV-Shist_highlow'+'.jpg', dpi=300, bbox_inches='tight', pad_inches=0)


# In[22]:


# Plotting the histograms for the V-channel from the HSV image, comparing the white marking (lower part of image)
# to the yellow marking (upper part of image).

plt.plot(Vlow, color = 'lightgreen', label ='HSV-V white')
plt.plot(Vhigh ,color='green', label ='HSV-V yellow')

plt.legend()
plt.savefig('C:/Cases/'+title+'HSV-Vhist_highlow'+'.jpg', dpi=300, bbox_inches='tight', pad_inches=0)


# In[23]:


# Converting the RGB image to YUV color space and naming the three color channels

yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

Y = yuv[:,:,0]
U = yuv[:,:,1]
V2 = yuv[:,:,2]

# Showing and saving the image
plt.axis('off')
plt.imshow(yuv)
plt.savefig('C:/Cases/'+title+'YUV'+'.jpg', dpi=300, bbox_inches='tight', pad_inches=0)


# In[24]:


# Separating the white (low) from the yellow (high) in the histogram plots

Ylow = np.sum(Y[500:1344,:],axis=0)
Yhigh = np.sum(Y[0:430,:],axis=0)
V2low = np.sum(V2[500:1344,:],axis=0)
V2high = np.sum(V2[0:430,:],axis=0)
Ulow = np.sum(U[500:1344,:],axis=0)
Uhigh = np.sum(U[0:430,:],axis=0)

plt.plot(Ylow, color = 'green', label ='YUV-Y')
plt.plot(Ulow, color = 'blue', label ='YUV-U')
plt.plot(V2low, color = 'red', label ='YUV-V')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('C:/Cases/'+title+'YUV_hist'+'.jpg', dpi=300, bbox_inches='tight', pad_inches=0)


# In[25]:


# Plotting the histograms for the Y-channel from the YUV image, comparing the white marking (lower part of image)
# to the yellow marking (upper part of image).

plt.plot(Ylow, color = 'lightgreen', label ='YUV-Y white')
plt.plot(Yhigh, color = 'green', label ='YUV-Y yellow')

plt.legend()
plt.savefig('C:/Cases/'+title+'YUV-Yhist_highlow'+'.jpg', dpi=300, bbox_inches='tight', pad_inches=0)


# In[26]:


# Plotting the histograms for the U-channel from the YUV image, comparing the white marking (lower part of image)
# to the yellow marking (upper part of image).

plt.plot(V2low, color = 'orange', label ='YUV-V white')
plt.plot(V2high, color = 'red', label ='YUV-V yellow')

plt.legend()
plt.savefig('C:/Cases/'+title+'YUV-Vhist_highlow'+'.jpg', dpi=300, bbox_inches='tight', pad_inches=0)


# In[27]:


# Plotting the histograms for the V-channel from the YUV image, comparing the white marking (lower part of image)
# to the yellow marking (upper part of image).

plt.plot(Ulow, color = 'lightblue', label ='YUV-U white')
plt.plot(Uhigh, color = 'blue', label ='YUV-U yellow')

plt.legend()
plt.savefig('C:/Cases/'+title+'YUV-Uhist_highlow'+'.jpg', dpi=300, bbox_inches='tight', pad_inches=0)


# In[28]:


# Plotting the histograms for the U- and V- channels from the YUV image, only for the yellow marking.

plt.plot(Uhigh, color = 'blue', label ='YUV-U yellow')
plt.plot(V2high, color = 'red', label ='YUV-V yellow')

plt.legend()
plt.savefig('C:/Cases/'+title+'YUV-UVhist_highlow'+'.jpg', dpi=300, bbox_inches='tight', pad_inches=0)

