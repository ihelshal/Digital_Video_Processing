# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 11:56:46 2019
@author: Ibrahim El-Shal

Assignment 1:
- Convert a RAW rgb image to a YUV format then Reconstruct the RGB channels. 
- Compute the PSNR between the source rgb image and the Reconstructed RGB using
  4:4:4, 4:2:2 and 4:1:1 format.
- You may select to use the average or the median, you shall show the different resultant images. 
- Remember to normalize the pixel values by 128.
- You may use any three of the RAW images provided under the class Moodle account.
- You can use any language, however, a low level language is recommended. 
Report is due on October 18, 2019.
"""
# In[1]: Import Packages

## Importing OpenCV(cv2) module 
import cv2
import math
import numpy as np
from scipy import misc

# In[2]: 

def Read_Image(Image_Path):
    ## Read RGB image 
    img = cv2.imread(Image_Path)
    return(img)
    
def Display_Image(Name,Image):
    ## Output img with window name as 'image' 
    cv2.imshow(Name, Image)    
    ## Save Image
    cv2.imwrite(Name+'.png',Image)
    ## Maintain output window utill, user presses a key  
    cv2.waitKey(2000)           
    ## Destroying present windows on screen 
    cv2.destroyAllWindows()  
    return(1)

# In[2-1]: Set Height and Width

W = 512
H = 512

# In[2-2]: Read Image and Reshape

img = np.fromfile("Lena Gray Raw Image.txt", dtype='uint8', sep="")
img = np.reshape(img, (W, H))

'''Please remove # below to proceed with barbara_gray and comment at the previous two lines'''
#img = misc.imread('barbara_gray.bmp', flatten= 1)

Display_Image("Original Image", img)
Image = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)   #Convert to 3D Image

# In[3]:    

def Normalize(Image):
    return (Image / 128)

def RGB_To_YUV(RGB_Image):
    
    RGB_Image = Normalize(RGB_Image)
        
    Coeff = np.array([[0.299, -0.14713, 0.615],
                      [0.587, -0.28886, -0.51499],
                      [0.114, 0.436, -0.10001]])
    
    YUV_Image = RGB_Image.dot(Coeff)
    YUV_Image = YUV_Image*128
    
    YUV_Image = YUV_Image.astype(np.uint8)
    
    YUV_Image[:,:,1] += 128  #b1
    YUV_Image[:,:,2] += 128  #b2
    
    return(YUV_Image)
        
# In[3-1]: 4:4:4 means no downsampling of the chroma channels.
    
yuv_444 = RGB_To_YUV(Image)
Display_Image("YUV Image 444", yuv_444)

# In[4]: 

def YUV_To_RGB(YUV_Image): 

    Coeff = np.array([[1, 1, 1],
                      [0, -0.39465, 2.03211],
                      [1.13983, -0.58060, 0]])
    
    RGB_Image = YUV_Image.dot(Coeff)
    RGB_Image = RGB_Image.astype(np.uint8)
    
    RGB_Image[:,:,0] -= 128  #b0
    RGB_Image[:,:,1] -= 128  #b1
    
    return(RGB_Image)

# In[4-1]: 

Recover_rgb_444 = YUV_To_RGB(yuv_444)
Display_Image("Recover RGB Image 444", Recover_rgb_444)

# In[5]: 

def Compute_MSE(OriginalImage,RecoveriedImage):
    
    err  = 0.0
    w = np.shape(OriginalImage)[0]  #Width
    h = np.shape(OriginalImage)[1]  #Hight
    
    err = np.sum((OriginalImage - RecoveriedImage)** 2)
    err /= (w * h)
    return(err) 

def Compute_PSNR(Computed_MSE):
    
    if (Computed_MSE == 0):
        return 100
    
    else:
        PIXEL_MAX = 255
        PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(Computed_MSE))
        return(PSNR)

# In[5-1]: 
        
mse_1 = Compute_MSE(Image,Recover_rgb_444)
psnr_1 = Compute_PSNR(mse_1)

# In[6]: 

def Filter(Array2D,SamplingType):
    
    W = np.shape(Array2D)[0]
    H = np.shape(Array2D)[1]
    
    for Row in range(0,W,2):
        for Col in range(0,H,2):
            
            Temp = Array2D[Row:Row+2, Col:Col+2]
            
            if(SamplingType == '422'):
                Temp[:,1] = Temp[:,0]
            elif(SamplingType == '411'):
                Temp[:,:] = Temp[0,0] 
            elif(SamplingType == 'Mean'):
                Temp = np.mean(Temp.ravel())
            else:
                return(0)
                
            Array2D[Row:Row+2, Col:Col+2] = Temp
    return(Array2D)
    
# In[7]: 
## 4:2:2 means 2:1 horizontal downsampling, with no vertical downsampling. 
## Every scan line contains four Y samples for every two U or V samples.
   
def Get_422_Partitioning(YUVImage):
    
    Y = YUVImage[:,:,0]
    U = Filter(YUVImage[:,:,1],'422')
    V = Filter(YUVImage[:,:,2],'422')
    New_Image = cv2.merge((Y,U,V))

    return (New_Image)

# In[7-1]: 
    
yuv_422 = Get_422_Partitioning(yuv_444)
rgb_422 = YUV_To_RGB(yuv_422)

Display_Image("Recover RGB Image 422", rgb_422)

mse_2 = Compute_MSE(Image,rgb_422)
psnr_2 = Compute_PSNR(mse_2)

# In[8]:
## 4:1:1 means 4:1 horizontal downsampling, with no vertical downsampling.
## Every scan line contains four Y samples for each U and V sample.

def Get_411_Partitioning(YUVImage):
    
    Y = YUVImage[:,:,0]
    U = Filter(YUVImage[:,:,1],'411')
    V = Filter(YUVImage[:,:,2],'411')
    New_Image = cv2.merge((Y,U,V))

    return (New_Image)
 
# In[8-1]: 
    
yuv_411 = Get_411_Partitioning(yuv_444)
rgb_411 = YUV_To_RGB(yuv_411)

Display_Image("Recover RGB Image 411", rgb_411)

mse_3 = Compute_MSE(Image,rgb_411)
psnr_3 = Compute_PSNR(mse_3)

# In[9]:

def Get_Mean_Partitioning(YUVImage):
    
    Y = YUVImage[:,:,0]
    U = Filter(YUVImage[:,:,1],'Mean')
    V = Filter(YUVImage[:,:,2],'Mean')
    New_Image = cv2.merge((Y,U,V))

    return (New_Image)

# In[9-1]:
    
yuv_Mean = Get_Mean_Partitioning(yuv_444)
rgb_Mean = YUV_To_RGB(yuv_Mean)

Display_Image("Recover RGB Image Mean", rgb_Mean)

mse_4 = Compute_MSE(Image,rgb_Mean)
psnr_4 = Compute_PSNR(mse_4)

# In[10]: 

print("\n")
print("At 444 Format,the MSE= ",mse_1,"and PSNR= ",psnr_1)
print("At 422 Format,the MSE= ",mse_2,"and PSNR= ",psnr_2)
print("At 411 Format,the MSE= ",mse_3,"and PSNR= ",psnr_3)
print("At Mean Format,the MSE= ",mse_4,"and PSNR= ",psnr_4)

# In[10]:  

def CV2_Library(Image_Path):
    img_yuv = cv2.cvtColor(Image_Path, cv2.COLOR_BGR2YUV)
    Display_Image("BGR2YUV",img_yuv)
 
    img_bgr =cv2.cvtColor(img_yuv,cv2.COLOR_YUV2BGR)
    Display_Image("YUV2BGR",img_bgr)
    return(0)

#CV2_Library(Image)
