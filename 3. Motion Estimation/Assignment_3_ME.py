# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 00:32:16 2019

@author: Ibrahim El-Shal
Assignment 3:
- Develop a Motion Estimation algorithm using Window Search to encode two images  in YUV format.  
- Use 8x8 block.
- Reconstruct the images using Motion Compensation. 
- Compute PSNR between the Source Image and the Reconstructed Images.
- Compare between the two Algorithms
- Bonus : you may choose more than one matching criteria
- Bonus : you may choose more than these two algorithms 
"""
# In[1]: Import Packages

import os
import sys
import cv2
import math
import time
import numpy as np

# In[1-2]:
 
GRID_SIZE = 8
OVERLAPPED_WIDTH    = 10
OVERLAPPED_HEIGHT   = 10

# In[2]: Functions of Image

def ReadFrames(FrameNumber):
    return(cv2.imread("frames/frame%d.jpg"%FrameNumber))
    
def RGB2YUV(RGBImage):
    return(cv2.cvtColor(RGBImage, cv2.COLOR_BGR2YUV)) 
    
def YUV2RGB(YUVImage):
    return(cv2.cvtColor(YUVImage,cv2.COLOR_YUV2BGR)) 
    
def Split_Channels(img):
    return (cv2.split((img)))   
    
def Create_Dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def Save_Image(Name,Image):
    return(cv2.imwrite(Image, Name))
    
def Get_PSNR(arr):
    mse = (arr ** 2).mean()
    psnr = 10 * math.log10((255 ** 2) / mse)
    return psnr

def psnr(orignal_picture,compressed_picture):
#     peak signal-to-noise ratio
    mse =0
    #mean squared error
    for i in range(len(orignal_picture)):
        for j in range(len(orignal_picture[i])):
            mse=mse+(orignal_picture[i][j]-compressed_picture[i][j])*(orignal_picture[i][j]-compressed_picture[i][j])
    mse=mse/(len(orignal_picture)*len(orignal_picture[i]))
    mx_value=0
    for lst in orignal_picture:
        value=max(lst)
        if value > mx_value:
            mx_value=value
    psnr_=10*math.log( mx_value*mx_value/ mse, 10)
    return psnr_
    
# In[3]: Convert Video Into frames
    
def Video2Frames(VideoName):
    
    cap = cv2.VideoCapture(VideoName)
    Create_Dir('./frames/') 
    frame_counter = 0
    
    if not cap.isOpened():
        print('{} not opened'.format(VideoName))
        sys.exit(1)

    while(1):
        return_flag, frame = cap.read()
        if not return_flag:
            print('Video Reach End')
            break
        #Start
        cv2.imwrite('./frames/' + 'frame%d.jpg' % frame_counter, frame)
        frame_counter += 1
        #End
    cap.release()
    return(1)

# In[4]: Get the needed search area 
    
def Search_Range(w_min, h_min, w_max, h_max, curr_w, curr_h, w_size, h_size):
    
    start_w = curr_w - w_size if curr_w - w_size > w_min else w_min
    end_w = curr_w + w_size if curr_w + w_size < w_max else curr_w
    start_h = curr_h - h_size if curr_h - h_size > h_min else h_min
    end_h = curr_h + h_size if curr_h + h_size < h_max else curr_h
    return (start_w, start_h, end_w, end_h)

# In[5]: Get Needed blocked 8x8
   
def Needed_Blocks(Frame):
    
    img_blocks = []
    img_blocks_idx = []    

    # shape of image
    height, width = Frame.shape

    for h_idx in range(0, height, GRID_SIZE):
        micro_block_per_row = []
        micro_block_idx_per_row = []
        for w_idx in range(0, width, GRID_SIZE):
            micro_block_per_row.append(Frame[h_idx: h_idx + GRID_SIZE, w_idx: w_idx + GRID_SIZE])
            micro_block_idx_per_row.append((w_idx, h_idx))
        img_blocks_idx.append(micro_block_idx_per_row)
        img_blocks.append(micro_block_per_row)

    return(img_blocks_idx, img_blocks)
    
# In[6]:Get the Movtion Vector of each picked Block to comparison with others 
   
def MotionVector(Current_Block, Next_Frame, x_micro, y_micro, search_area):
    
    mv = (0, 0)
    min_value = np.inf
    
    start_w, start_h, end_w, end_h = search_area
    
    for y in range(start_h, end_h + 1):
        for x in range(start_w, end_w + 1):
            # search range 
            window_block = Next_Frame[y:y + GRID_SIZE, x:x + GRID_SIZE]
            value = np.sum(np.abs(Current_Block - window_block))

            if value < min_value:
                mv = (x - x_micro, y - y_micro)
                min_value = value         
    return(mv)
    
# In[7]: 

def Block_Matching(curr_frame, next_frame):

    height, width = curr_frame.shape
    block_idx_list, block_list = Needed_Blocks(curr_frame)

    frame_motion_vector = [[0 for j in range(len(block_idx_list[0]))] for i in range(len(block_list))]
    
    for h in range(len(block_idx_list)):
        for w in range(len(block_list[0])):
            # search range 
            micro_x, micro_y = block_idx_list[h][w]
            Grid_Block = block_list[h][w]

            search_range = Search_Range(0, 0, width, height, micro_x, micro_y, GRID_SIZE, GRID_SIZE)
        
            frame_motion_vector[h][w] = MotionVector(Grid_Block,next_frame,
                                                     micro_x, micro_y, search_range)

    return frame_motion_vector

# In[8]: 
    
def TSS_Block_Matching(curr_frame, next_frame):

    TSS_GRID_SIZE = GRID_SIZE
    height, width = curr_frame.shape
    block_idx_list, block_list = Needed_Blocks(curr_frame)

    frame_motion_vector = [[(0,0) for j in range(len(block_idx_list[0]))] for i in range(len(block_list))]
    
    for h in range(len(block_idx_list)-1): 
        for w in range(len(block_list[0])-1): 
            # search range 
            micro_x, micro_y = block_idx_list[h][w]
            Grid_Block = block_list[h][w]            
            TSS_GRID_SIZE = GRID_SIZE
            
            for i in range(3):
                TSS_GRID_SIZE = TSS_GRID_SIZE // 2
                search_range = Search_Range(0, 0, width, height, micro_x, micro_y,
                                            TSS_GRID_SIZE, TSS_GRID_SIZE)
                
                frame_motion_vector[h][w] = MotionVector(Grid_Block,next_frame,
                                                         micro_x, micro_y, search_range)
                
                micro_x, micro_y = frame_motion_vector[h][w]
                
    return frame_motion_vector

# In[8]:

def Overlapped_Motion_Vector(Current_frame, motion_vector):
    
    height, width = Current_frame.shape
    Current_frame = Current_frame.astype(np.uint32)
    
    overlapped_range = [[[] for j in range(len(motion_vector[i]))]  for i in range(len(motion_vector))]
    overlapped_width = int((OVERLAPPED_WIDTH - GRID_SIZE) / 2)
    overlapped_height = int((OVERLAPPED_HEIGHT - GRID_SIZE) / 2)

    overlapped_motion_vector = [[[] for j in range(width)] for i in range(height)]

    for h in range(0, int(height / GRID_SIZE)):
        for w in range(0, int(width / GRID_SIZE)):
            temp_w = w * GRID_SIZE
            temp_h = h * GRID_SIZE
            s_x = temp_w - overlapped_width if temp_w - overlapped_width >= 0 else temp_w
            s_y = temp_h - overlapped_height if temp_h - overlapped_height >= 0 else temp_h
            e_x = (w + 1) * GRID_SIZE
            e_x = e_x + overlapped_width if e_x + overlapped_width < width else e_x
            e_y = (h + 1) * GRID_SIZE
            e_y = e_y + overlapped_height if e_y + overlapped_height < height else e_y
            overlapped_range[h][w] = (motion_vector[h][w], [[s_x, s_y], [e_x, e_y]])
            for y in range(s_y, e_y):
                for x in range(s_x, e_x):
                    overlapped_motion_vector[y][x].append(motion_vector[h][w])
    
    return(overlapped_motion_vector)
    
# In[9]:
    
#Function to reconstruct a frame from a reference frame given the motion vectors in a macroblock 
#Inputs: Reference Frame, Macroblocks containing motion vectors
#Outputs:reconstructed_frame 

def Create_Compressed_Image(Curr_frame, Post_frame, overlapped_MV):
    
    height, width = Curr_frame.shape
    Post_frame = Post_frame.astype(np.uint32)
    interpolated_frame = [[0 for j in range(width)] for i in range(height)]

    for y in range(height):
        for x in range(width):
            sum = 0
            for mv in overlapped_MV[y][x]:
                
                prev_y = y + mv[1]
                if prev_y >= height or prev_y < 0:
                    prev_y = 0 if prev_y < 0 else height - 1

                prev_x = x + mv[0]
                if prev_x >= width or prev_x < 0:
                    prev_x = 0 if prev_x < 0 else width - 1

                next_y = y - mv[1]
                if next_y >= height or next_y < 0:
                    next_y = 0 if next_y < 0 else height - 1

                next_x = x - mv[0]
                if next_x >= width or next_x < 0:
                    next_x = 0 if next_x < 0 else width - 1

                sum += Curr_frame[prev_y][prev_x] + Post_frame[next_y, next_x]

            l = len(overlapped_MV[y][x]) * 2
            res = sum / l
            res = np.array(res).T
            
            interpolated_frame[y][x] = res.astype(np.uint8)

    Final_Image = np.array(interpolated_frame)
    return(Final_Image)

# In[10]:

def Window_Full_Search():    

    current_frame = ReadFrames(0)
    next_frame = ReadFrames(1)

    #Convert to YUV Image
    current_yuv = RGB2YUV(current_frame)
    next_yuv = RGB2YUV(next_frame)

    ###Get Channels
    curr_Y, curr_U, curr_V = Split_Channels(current_yuv)
    next_Y, next_U, next_V = Split_Channels(next_yuv)

    Mv = Block_Matching(curr_Y,next_Y)
    Overlapped_Mv = Overlapped_Motion_Vector(curr_Y, Mv)
    Img = Create_Compressed_Image(curr_Y, next_Y, Overlapped_Mv)  
       
    return(Img)


def TSS_Search():    

    current_frame = ReadFrames(0)
    next_frame = ReadFrames(1)

    #Convert to YUV Image
    current_yuv = RGB2YUV(current_frame)
    next_yuv = RGB2YUV(next_frame)

    ###Get Channels
    curr_Y, curr_U, curr_V = Split_Channels(current_yuv)
    next_Y, next_U, next_V = Split_Channels(next_yuv)

    Save_Image(curr_Y,"Original Img.jpg")
    Mv = TSS_Block_Matching(curr_Y,next_Y)
    Overlapped_Mv = Overlapped_Motion_Vector(curr_Y, Mv)
    Img = Create_Compressed_Image(curr_Y, next_Y, Overlapped_Mv)  
       
    return(Img)

# In[11]:
 
def main():
    
    #Video2Frames('./video.mp4')
    
    start = time.time()
    WinImg = Window_Full_Search()
    end = time.time()
    res_psnr = Get_PSNR(WinImg)
    print('PSNR at Window Matching:',res_psnr)
    print('Window Matching Running Time:',(end - start)) 

    start = time.time()
    TssImg = TSS_Search()
    end = time.time()
    res_psnr = Get_PSNR(WinImg)
    print('\nPSNR at TSS:',res_psnr)
    print('TSS Running Time:',(end - start))  
    
    return(WinImg,TssImg)

# In[11]:

## call the main function
if __name__ == '__main__':
    WinImg,TssImg = main()

Save_Image(WinImg,"Img of Window.jpg")
Save_Image(TssImg,"Img of Thee Step.jpg")