# Digital_Video_Processing

Real-time video compression is computationally expensive due to the requirement of high compression efficiency.
Video Processing  is an exciting, emerging area of the security and surveillance industry that offers solutions to the current problem of non-responsive surveillance. Complex algorithms look for specified changes in the video feed and then alert security personnel to threats so they can immediately intervene. The analytic process can be used for various purposes such as:  

1) motion detection of specific objects and movements such as a person running when everyone else is walking,  
2) object detection such as a bag left in a specific public area, 
3) piggybacking or tailgating in which an individual follows another person into a secure area, 
4) perimeter protection which sends an alert when a specific boundary is crossed. 
            
Video analytics offers the security market several benefits including: 

1) less dependence on human vigilance 
2) fewer personnel to monitor video feeds 
3) ability to respond to immediate threats 
4) ability to quickly search and retrieve specified security events.
            
 The repository contains three folders which each on has an implementation for Video decoding/encoding blocks
 
# 1. RGB2YUV

My pipeline consisted of 5 steps. 
1) Convert a RAW rgb image to a YUV format.
2) Reconstruct the RGB channels. 
3) Compute the PSNR between the source rgb image and the Reconstructed RGB.
4) Recompute PSNR using 4:4:4, 4:2:2 and 4:1:1 format.
5) Normalize the pixel values by 128.

<figure>
 <img src="1. RGB2YUV/Lena Gray Output/Recover RGB Image 444.png" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
    <p style="text-align: center;"> Reconstructed RGB Image with 4:4:4 Format </p> 
 </figcaption>
</figure>
 
# 2. DCT and Quantization Algorithm

Develop a DCT algorithm to encode images in YUV format by useing 8x8 block and compute the DCT coefficients then quantize them.
Code also provided with IDCT function in order to compute the PSNR between the Source Image and the Reconstructed Images.

<figure>
 <img src="2. DCT and Quantization Algorithm/Lena Gray Output/Original Image.png" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
    <p style="text-align: center;"> Original RGB Image</p> 
 </figcaption>
</figure>

<figure>
 <img src="2. DCT and Quantization Algorithm/Lena Gray Output/Reconstructed Original Image.png" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
    <p style="text-align: center;"> Reconstructed RGB Image</p> 
 </figcaption>
</figure>

# 3. Motion Estimation and Compensation 

1) Develop a Motion Estimation algorithm using Window Search and Three Step Search to encode two images in YUV format.  
2) Use 8x8 block.
3) Reconstruct the images using Motion Compensation. 
4) Compute PSNR between the Source Image and the Reconstructed Images.

<figure>
 <img src="3. Motion Estimation/Img of Window.jpg" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
    <p style="text-align: center;"> Image of Full Window Search</p> 
 </figcaption>
</figure>

<figure>
 <img src="3. Motion Estimation/Img of Thee Step.jpg" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
    <p style="text-align: center;"> Image of Three Step Search</p> 
 </figcaption>
</figure>
