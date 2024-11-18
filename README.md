# EX 12: project
## DATE:
## Aim
To write a python program using OpenCV to do the following image manipulations.
i) Extract ROI from  an image.

ii) Perform handwritting detection in an image.

iii) Perform object detection with label in an image.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
## I) Perform ROI from an image

Step1:Import necessary packages 

Step2:Read the image and convert the image into RGB

Step3:Display the image

Step4:Set the pixels to display the ROI 

Step5:Perform bit wise conjunction of the two arrays  using bitwise_and 

Step6:Display the segmented ROI from an image.

## II) Perform handwritting detection in an image

Step1:Import necessary packages 

Step2:Define a function to read the image,Convert the image to grayscale,Apply Gaussian blur to reduce noise and improve edge detection,Use Canny edge detector to find edges in the image,Find contours in the edged image,Filter contours based on area to keep only potential text regions,Draw bounding boxes around potential text regions.

Step3:Display the results.

## III) Perform object detection with label in an image

Step1:Import necessary packages 

Step2:Set and add the config_file,weights to ur folder.

Step3:Use a pretrained Dnn model (MobileNet-SSD v3)

Step4:Create a classLabel and print the same

Step5:Display the image using imshow()

Step6:Set the model and Threshold to 0.5

Step7:Flatten the index,confidence.

Step8: Display the result.

## PROGRAM:

### NAME : Jeevanesh S
### REG.N0 : 212222243002

## I) Perform ROI from an image

```
#Import necessary packages 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image and convert the image into RGB
image_path = "sunrise.jpeg"
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')
plt.show()

# Set the pixels to display the ROI 
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_yellow = np.array([22, 93, 0])#choose the RGB values accordingly to display specific color
upper_yellow = np.array([45, 255, 255])
mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

# Perform bit wise conjunction of the two arrays  using bitwise_and 
segmented_image = cv2.bitwise_and(img, img, mask=mask)

# Convert the image from BGR2RGB
segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

# Display the segmented ROI from an image.
plt.imshow(segmented_image_rgb)
plt.title('Segmented Image (Yellow)')
plt.axis('off')
plt.show()

```

## II) Perform handwritting detection in an image
```
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_handwriting(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detector to find edges in the image
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to keep only potential text regions
    min_area = 100
    text_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Draw bounding boxes around potential text regions
    img_copy = img.copy()
    for contour in text_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title('Handwriting Detection')
    plt.axis('off')
    plt.show()
    
    
    # Path to the image containing handwriting
      image_path = "handwritten.jpg"

    # Perform handwriting detection
      detect_handwriting(image_path)

```
## III) Perform object detection with label in an image
```
# Import necessary packages 
import cv2
import matplotlib.pyplot as plt

# Set and add the config_file,weights to ur folder.
config_file='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model='frozen_inference_graph.pb'

# Use a pretrained Dnn model (MobileNet-SSD v3)
model=cv2.dnn_DetectionModel(frozen_model,config_file)

# Create a classLabel and print the same
classLabels = []
file_name='Labels.txt'
with open(file_name,'rt')as fpt:
    classLabels=fpt.read().rstrip('\n').split('\n')

# Print the classLabels
print(classLabels)
print(len(classLabels))

# Display the image using imshow()
img=cv2.imread('car.jpeg')
plt.imshow(img)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

# Set the model and Threshold to 0.5
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)#255/2=127.5
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)
ClassIndex,confidence,bbox=model.detect(img,confThreshold=0.5)
print(ClassIndex)


#Flatten the index,confidence.
font_scale=3
font=cv2.FONT_HERSHEY_PLAIN
for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
    cv2.rectangle(img,boxes,(0,0,255),2)
    cv2.putText(img,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(255,0,0),thickness=1)


# Display the result.
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

```
## OUTPUT:
## I) Perform ROI from an image
![Screenshot 2024-11-18 233504](https://github.com/user-attachments/assets/4dc711fc-6853-4008-8ddc-a8b7d6f0a95a)
![Screenshot 2024-11-18 233510](https://github.com/user-attachments/assets/d94448d6-c0b5-4ab0-956b-4d33d8156f7d)


## II) Perform handwritting detection in an image

![WhatsApp Image 2024-11-15 at 11 00 32_bd8c602b](https://github.com/user-attachments/assets/93bebd78-43af-4c2b-b525-496a49f60141)

## III) Perform object detection with label in an image

![WhatsApp Image 2024-11-15 at 11 04 03_a8cf2636](https://github.com/user-attachments/assets/a8de1547-acda-46c9-ab9e-408bbb6f172d)

## RESULT:
Thus, The python program using OpenCV to do the image manipulations is executed successfully.



