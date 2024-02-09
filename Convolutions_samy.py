import numpy as np
import cv2

from matplotlib import pyplot as plt

#Lecture image en niveau de gris et conversion en float64
img=np.float64(cv2.imread('../Image_Pairs/FlowerGarden2.png',0))
(h,w) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes")

#Méthode directe
t1 = cv2.getTickCount()
img2_x = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
img2_y = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
grad_norm = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
for y in range(1,h-1):
  for x in range(1,w-1):
    val_x = - 3*img[y-1, x-1] + 3 * img[y+1,x-1]- 3*img[y-1, x+1] + 3 * img[y+1,x+1] -10*img[y-1,x] + 10*img[y+1,x] #composante horizontale du gradient
    val_y = - 3*img[y-1, x-1] - 3 * img[y+1,x-1] + 3*img[y-1, x+1] + 3 * img[y+1,x+1] -10*img[y,x-1] + 10*img[y,x+1] #composante verticale du gradient
    img2_x[y,x] = min(max(val_x,0),255)
    img2_y[y,x] = min(max(val_y,0),255)
    grad_norm[y,x] = np.sqrt(img2_x[y,x]**2 + img2_y[y,x]**2) #norme du gradient
   
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode directe :",time,"s")

cv2.imshow('Avec boucle python',img2_y.astype(np.uint8))
#Convention OpenCV : une image de type entier est interprétée dans {0,...,255}
cv2.waitKey(0)

cv2.imshow('grad',grad_norm.astype(np.uint8))
cv2.waitKey(0)

plt.subplot(133)
plt.imshow(grad_norm , cmap= 'gray')
plt.title('norme du gradient')

plt.subplot(131)
plt.imshow(img2_y,cmap = 'gray')
plt.title('Convolution - Méthode Directe')

#Méthode filter2D
t1 = cv2.getTickCount()
img3 = cv2.Sobel(img,-1,1,0,ksize=-1) #3x3 Scharr filter (Sobel)
# kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
# img3 = cv2.filter2D(img,-1,kernel)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode filter2D :",time,"s")

cv2.imshow('Avec filter2D',img3/255.0)
#Convention OpenCV : une image de type flottant est interprétée dans [0,1]
cv2.waitKey(0)

plt.subplot(132)
plt.imshow(img3,cmap = 'gray',vmin = 0.0,vmax = 255.0)
#Convention Matplotlib : par défaut, normalise l'histogramme !
plt.title('Convolution - filter2D')

plt.show()
