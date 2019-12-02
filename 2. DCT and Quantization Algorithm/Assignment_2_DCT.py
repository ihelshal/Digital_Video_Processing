# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 19:53:08 2019

@author: Ibrahim El-Shal

Assignment 2:
- Develop a DCT algorithm to encode images in YUV format.  
- Use 8x8 block and computer the DCT coefficients.
- Use a quantization of your choice.  
- Reconstruct the images using IDCT.
- Compute the PSNR between the Source Image and the Reconstructed Images.
"""
# In[1]: Import Packages

## Importing OpenCV(cv2) module 
import cv2
import math
import itertools 
import numpy as np
from scipy import misc

# In[2]: Display The Image and Save it

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
    
# In[3]: Split & Merge channels
    
def Split_Channels(img):
    return (cv2.split((img)))    

def Merge_Channels(ch1,ch2,ch3):
    return (cv2.merge((ch1,ch2,ch3)))  

# In[4]: Convert to 3D Image
    
def Image3D(img):
    return (cv2.cvtColor(img,cv2.COLOR_GRAY2RGB))

# In[5]: Image Normalization     

def Normalize(Image):
    return (Image / 128)

# In[6]: Convert from RGB to YUV Image   

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
    print("--------- Image Converted to YUV ---------")
    print("\n")
    return(YUV_Image)
 
# In[7]: Convert from RGB to YUV Image
    
def YUV_To_RGB(YUV_Image): 

    Coeff = np.array([[1, 1, 1],
                      [0, -0.39465, 2.03211],
                      [1.13983, -0.58060, 0]])
    
    RGB_Image = YUV_Image.dot(Coeff)
    RGB_Image = RGB_Image.astype(np.uint8)
    
    RGB_Image[:,:,0] -= 128  #b0
    RGB_Image[:,:,1] -= 128  #b1
    
    return(RGB_Image)

# In[8]: Convert RGB to Gray Scale
    
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
        
# In[9]: Get Needed blocked 8x8
   
def Get_Blocks(Channel):
    
    # prevent against multiple-channel images
    if len(Channel.shape) != 2:
        raise ValueError('Input image must be a single channel 2D array')
    
    # shape of image
    height, width = Channel.shape
    
    # No of needed blocks 
    # new block height
    n_height = np.int32(math.ceil(height / 8)) * 8
    # new block width
    n_width = np.int32(math.ceil(width / 8)) * 8
    
    # create a numpy zero matrix with size of H,W
    padded_img = np.zeros((n_height, n_width))
    padded_img[0:n_height, 0:n_width] = Channel
    
    # split into blocks
    img_blocks = [padded_img[j:j + 8, i:i + 8]
                  for (j, i) in itertools.product(range(0, n_height, 8),
                                                  range(0, n_width, 8))] 
    print("--------- Get The Needed Blockes ---------")
    return(img_blocks)

# In[10]: Get the alfa values

def Check_Alfa_Value(Spatial_Frequency):
    
    if(Spatial_Frequency == 0):
        #Normalizing Scale Factor
        NSF = 1 / np.sqrt(2)  
    else:
        NSF = 1
    return(NSF)

# In[11]: DCT / IDCT Formula 
    
def DCT_Formula(blockimg,u,v):
    
    DCT = 0
    # shape of image
    M, N = blockimg.shape
    alfa_u = Check_Alfa_Value(u)
    alfa_v = Check_Alfa_Value(v)
        
    for x in range(0,M):
        a0 = (2*x+1)*u*math.pi
        for y in range(0,N):    
            a1 = (2*y+1)*v*math.pi
            
            result = blockimg[x][y] * math.cos(a0/(2*M)) * math.cos(a1/(2*N)) * alfa_u * alfa_v
            DCT += result
    
    DCT = DCT/4
    return(DCT)

# In[12]: DCT / IDCT transform functions   

def Compute_DCT(imgblock,BlockSize= 8):
    #DCT transform every block
    dct_img = np.zeros((BlockSize, BlockSize))
    
    for i  in range(0,BlockSize):
        for j in range(0,BlockSize):
            dct_img[i][j] = DCT_Formula(imgblock,i,j)   
            
    return(dct_img)

# In[13 & 14]: Quantize and Inv. all the DCT coefficients using the quant. matrix
'''
- The compression the floating point data, will be not effective.
- Convert the weights-matrix back to values in the space of [0,255].
- Do this by finding the min/max value for the matrix and dividing each number 
  in this range to give us a value between [0,1] to which we multiply by 255 to get final value.
'''
Quant = np.array([[16,11,10,16, 24,40, 51, 61],
                  [12,12,14,19, 26,48, 60, 55],
                  [14,13,16,24, 40,57, 69, 56],
                  [14,17,22,29, 51,87, 80, 62],
                  [18,22,37,56, 68,109,103,77],
                  [24,35,55,64, 81,104,113,92],
                  [49,64,78,87,103,121,120,101],
                  [72,92,95,98,112,100,103,99]])

def Quantize(dct_blocks):
    Quantized_Blocks = [(dct_block/Quant)for dct_block in dct_blocks]
    print("--------- Quantization Done ---------")
    print("\n")
    return(Quantized_Blocks)

def Inv_Quantize(qnt_blocks):
    Inv_Quantized_Blocks = []
    for i in range(0,len(qnt_blocks)):
        Inv_Quantized_Blocks.append(qnt_blocks[i]*Quant)
        
    print("--------- Inverse Quantization Done ---------")
    return(Inv_Quantized_Blocks)
    
# In[15]: Reshape the image blocks to 512x512

def Chunk(l,n):
    return [l[i:i+n]  for i in range(0, len(l), n)]

def Reshape_Image(ImgBlocks,BlockSize= 8,ImageSize= 512):
    rec_img = []
    for chunk_row_blocks in Chunk(ImgBlocks, ImageSize // BlockSize):
        for row_block_num in range(BlockSize):
            for block in chunk_row_blocks:
                rec_img.extend(block[row_block_num])
            
    rec_img = np.array(rec_img).reshape(ImageSize, ImageSize)
    print("--------- Image Reshaped ---------") 
    print("\n")
    return(rec_img)  

# In[16]: Compute MSE
    
def Compute_MSE(OriginalImage,RecoveriedImage):
    return np.sqrt(((OriginalImage-RecoveriedImage)**2).mean())

# In[17]: Compute PSNR
    
def Compute_PSNR(Computed_MSE):
    
    if (Computed_MSE == 0):
        return 100
    
    else:
        PIXEL_MAX = 255
        PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(Computed_MSE))
        return(PSNR)

# In[18]: 
 
def Encode(source):
    
    print("--------- Encoding Process ---------") 
    # Init block 8x8
    blocks = Get_Blocks(source)
    
    # DCT every block
    dct_blocks = []
    for blk in range(0,len(blocks)):
        dct_blocks.append(Compute_DCT(blocks[blk]))
    print("--------- DCT Done ---------")
    
    # Quantize every block
    QuantBlocks = Quantize(dct_blocks)
    
    return(blocks, QuantBlocks)

# In[19]: 
    
def Decode(blocks,QuantBlocks):
    
    print("--------- Decoding Process ---------") 
    # Inverse Quantization for every block
    InvQuantBlocks = Inv_Quantize(QuantBlocks)
    
    # IDCT of every block
    idct_blocks = []
    for blk in range(0,len(blocks)):
        idct_blocks.append(Compute_DCT(InvQuantBlocks[blk]))
    print("--------- Inverse DCT Done ---------")

    rec_img = Reshape_Image(idct_blocks)
    
    return(rec_img)

# In[20]: 

def main(Image= 'img_1'):
    
    if(Image == 'img_1'):
        #Read Image and Reshape
        img = np.fromfile("Lena Gray Raw Image.txt", dtype='uint8', sep="")
        #Set Height and Width
        img = np.reshape(img, (512, 512)) 
        
    elif(Image == 'img_2'):
        
        img = misc.imread('barbara_gray.bmp', flatten= 1)
    else:
        raise AttributeError("Enter the Image Number please")

    Display_Image("Original Image", img)
    
    #Convert to 3D Image
    img = Image3D(img) 
    #Convert to YUV Image
    yuv = RGB_To_YUV(img)
    Display_Image("YUV Image", yuv)
    
    Y, U, V = Split_Channels(yuv)
    
    Blocks_Y, Encoded_Y = Encode(Y)
    Blocks_U, Encoded_U = Encode(U)
    Blocks_V, Encoded_V = Encode(V)
    
    Recovered_Y = Decode(Blocks_Y, Encoded_Y)
    Recovered_U = Decode(Blocks_U, Encoded_U)
    Recovered_V = Decode(Blocks_V, Encoded_V)
    
    Recovered_Img = Merge_Channels(Recovered_Y, Recovered_U, Recovered_V)
    Display_Image("Reconstructed YUV Image", Recovered_Img)
    
    RGB = YUV_To_RGB(Recovered_Img)
    Display_Image("Reconstructed Original Image", RGB)
    
    mse = np.round(Compute_MSE(Recovered_Img, RGB),2)
    psnr = np.round(Compute_PSNR(mse), 2)
    print('mse={} and psnr={}dB'.format(mse, psnr)) 
    
    Gray = rgb2gray(RGB)   
    Display_Image("gray", Gray)
    return(1)
    
# In[21]: 

## call the main function
if __name__ == '__main__':
    Recovered_Img = main()
