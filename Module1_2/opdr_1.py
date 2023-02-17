import numpy as np
import matplotlib.pyplot as plt
import PyQt5
import sklearn
import skimage as ski
from skimage import io, color, data

#isolate color based on HSV

def isolateColorHSV(image , low, high, array = np.array([])):
    # itterate through rows
    for i, row in enumerate(image):
        #itterate through pixels
        for j, pixel in enumerate(row):
            if low <= 30/360:
                # print("red")
                if  pixel[0] >= low and pixel[0] <= 330/360:

                    image[i][j][1] = 0
            # check if hue values are between the color(high low)
            else:
                # print("yeet")
                if  pixel[0] < low or pixel[0] > high:
                    
                    image[i][j][1] = 0
            # check if not a white pixel
            if(pixel[1] != 0.0 ):
                # add to array for histogram
                array = np.append(array, pixel[0])
    return image, array



#isolate color based on RGB

def isolateColor(image):
    # itterate through rows
    for i, row in enumerate([image]):
        #itterate through pixels
        for j, pixel in enumerate(row):
            # check if color is red
            # if  (pixel[0]/1.5  > pixel[1]) and (pixel[0]/1.5  > pixel[2]):
            if  (pixel[1]/1.15  > pixel[0]) and (pixel[1]/1.15  > pixel[2]):
            # if  (pixel[2]/1.1 > pixel[1]) and (pixel[2]/1.1  > pixel[0]):
                pixel = pixel
            else:
                image[i][j] = color.rgb2gray(image[i][j])
    return image

def sethistogramvalues(image):
    ravelimg = image.ravel()
    red = image[:, :, 0].ravel()
    green = image[:, :, 1].ravel()
    blue = image[:, :, 2].ravel()
    return ravelimg, red, green, blue



def setGradient(pltX, pltY, image):
    n, bins, patches = ax[pltX,pltY].hist(image[:,:,0].flat, bins=100)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    cm = plt.cm.get_cmap('hsv')
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))

def setGradientHue(pltX, pltY, image):
    n, bins, patches = ax[pltX,pltY].hist(image.flat, bins=100)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    cm = plt.cm.get_cmap('hsv')
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))



image = ski.img_as_float(io.imread("London.jpg"))
# imgcopy = isolateColor(image.copy())

grayscaled = color.rgb2gray(image)
hsvimage = color.rgb2hsv(image)
isolated = hsvimage.copy()

# yellow
# low = [40,20,0]
# high = [255,255,105]

#reds
low = 20/360
high = 330/360
#blues
# low = 120/360
# high = 260/360


cm = plt.cm.get_cmap('hsv')
fig, ax = plt.subplots(2, 3, figsize=(8,4)) 



#diagrams/pictures for rgb


# ravelimg, r,g,b = sethistogramvalues(image)
# ax[0,0].imshow(image)

# ax[1,0].hist(ravelimg, bins= 256, color = "pink")
# ax[1,0].hist(r, bins= 256, color = "red", alpha = 0.5)
# ax[1,0].hist(g, bins= 256, color = "green", alpha = 0.5)
# ax[1,0].hist(b, bins= 256, color = "blue", alpha = 0.5)
# ax[1,0].set_xlabel('value')
# ax[1,0].set_ylabel('pixels')
# ax[1,0].legend(['total', 'red', 'green', 'blue'])


# ravelimg, r,g,b = sethistogramvalues(imgcopy)
# ax[0,1].imshow(imgcopy)
# ax[1,1].hist(ravelimg, bins= 256, color = "pink")
# ax[1,1].hist(r, bins= 256, color = "red", alpha = 0.5)
# ax[1,1].hist(g, bins= 256, color = "green", alpha = 0.5)
# ax[1,1].hist(b, bins= 256, color = "blue", alpha = 0.5)
# ax[1,1].set_xlabel('value')
# ax[1,1].set_ylabel('pixels')
# ax[1,1].legend(['total', 'red', 'green', 'blue'])


#diagrams/pictures for HSV

# print(result)
ax[0,0].imshow(image)
ax[0,0].set_title("isolated")


isolatedpic, array = isolateColorHSV(isolated, low, high)
isolatedpic = color.hsv2rgb(isolatedpic)
ax[0,1].imshow(isolatedpic)
ax[0,1].set_title("isolated")



setGradient(1,0,hsvimage)

setGradientHue(1,1,array)
ax[1,1].set_title("hue histogram of isolated")



ax[0,2].imshow(grayscaled, cmap="gray")
ax[0,2].set_title("grayscaled")

ax[1,2].hist(grayscaled[:,:].flat, color='gray', bins=160, rwidth=1)

# # print(array)

fig.tight_layout()
plt.show()

