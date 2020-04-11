import numpy as np
import cv2
import os, os.path

imgs = []
names=[]
path = "C:/Users/tsarcevic/PycharmProjects/fingerprinting-toolbox/robustness_analysis/categorical_neighbourhood/plots/nursery/images/"
#valid_images = [".jpg",".gif",".png",".tga", ".PNG"]
# = [".PNG"]
for f in os.listdir(path):
    names.append(f)
    imgs.append(os.path.join(path,f))
print(names)
print("Images: " + str(imgs))
for (name,items) in zip(names,imgs):
        print(items)
        img = cv2.imread(items) # Read in the image and convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = 255*(gray < 128).astype(np.uint8) # To invert the text to white
        coords = cv2.findNonZero(gray) # Find all non-zero points (text)
        x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
        rect = img[y:y+h, x:x+w] # Crop the image - note we do this on the original image
        #cv2.imshow("Cropped", rect) # Show it
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        dest = os.path.join(path,name)
        cv2.imwrite(dest, rect) # Save the image