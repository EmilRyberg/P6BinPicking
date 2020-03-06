import os, PIL
import numpy as np
from PIL import Image

def depth_average(img1, img2, img3):
    imlist=[img1, img2, img3]
    print(img1)

    w = 1280 # Assuming all images are the same size
    h = 720
    n = len(imlist)
    avg=np.zeros((h,w),np.float)

    # Build up average pixel intensities, casting each image as an array of floats
    for img in imlist:
        imarr=np.array(img)
        avg=avg+imarr/n

    # Round values in array and cast as 8-bit integer
    avg=np.array(np.round(avg),dtype=np.uint8)

    # Generate, save and preview final image
    out=Image.fromarray(avg,mode="L")
    out.save("Average.png")
    #out.show()
    return out


if __name__ == "__main__":
    img1 = Image.open("depth1.png")
    img2 = Image.open("depth2.png")
    img3 = Image.open("depth3.png")
    #img1.show()
    #img2.show()
    #img3.show()
    depth_average(img1, img2, img3)
