
# coding: utf-8

# In[14]:


# Importing necessary packages for OCR pre-processing
import cv2
import numpy as np
import copy
import math
from scipy.spatial import distance as dist
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import os

import glob
import imutils

from math import atan2, cos, sin, sqrt, pi,degrees
from scipy import stats
import shutil
RESULT_FOLDER = 'static/generated_images/'
ALL_FOLDER = 'all/'
# In[15]:


# Importing image
def read_image(path,out_folder):
    image = cv2.imread(path)
#     cv2.imwrite(r"generated_images\original.png",image)
    cv2.imwrite(out_folder+"original.png",image)
    return(image)


# In[16]:


# Deep Copy created for cropping to avoid overriding of images
def create_img_copy(image):
    image_copy = copy.deepcopy(image)
    return(image_copy)


# In[17]:


# cropping
def crop_image(image_copy,x,y,w,h,out_folder):
    crop = image_copy[y:y+h, x:x+w]
    flag = cv2.imwrite(out_folder+"cropped.png",crop)
    return(crop,flag)


# In[18]:


def rotate_image3(out_folder):
    rotated3 = cv2.imread(out_folder+"cropped.png")
    
    height, width = rotated3.shape[:2]
    res = cv2.resize(rotated3,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
#     img = convert_to_gray(res)
#     equ = cv2.equalizeHist(img)
#     cv2.imwrite('generated_images\\contrast.png',equ)

    rot = pytesseract.image_to_osd(res).split("\n")
    print(rot)
    rot1 = rot[2].split(": ")
    rotated_image3 = imutils.rotate_bound(rotated3, int(rot1[1]))

    flag = cv2.imwrite(out_folder+"rotated3.png",rotated_image3)

    return(rotated_image3,flag)


# In[19]:


############ Utility functions ############
def decode(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            # If score is lower than threshold score, move to next x
            if(score < scoreThresh):
                continue

            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
            center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
            detections.append((center, (w,h), -1*angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]



def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
 
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
 
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
 
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
 
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


# def get_text(img):
#     #preprocess
#     img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

#      # Convert to gray
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Apply dilation and erosion to remove some noise
#     kernel = np.ones((1, 1), np.uint8)
#     img = cv2.dilate(img, kernel, iterations=1)
#     img = cv2.erode(img, kernel, iterations=1)

#     # Apply blur to smooth out the edges
#     img = cv2.GaussianBlur(img, (5, 5), 0)

#     # Apply threshold to get image with only b&w (binarization)
#     img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#     #cv2.imshow("processed sub", img)
#     result = pytesseract.image_to_string(img)
#     return result


def east(rotated,flag,out_folder):
    
    # Read and store arguments
    confThreshold = 0.5
    nmsThreshold = 0.7
    inpWidth = 800
    inpHeight = 800
    model = "frozen_east_text_detection.pb"
    listHorizontal = []
    listVertical = []
    lst = []
    rotation_angle = 0
    h_dist_counter = 0
    v_dist_counter = 0

    # Load network
    net = cv2.dnn.readNet(model)

    outputLayers = []
    outputLayers.append("feature_fusion/Conv_7/Sigmoid")
    outputLayers.append("feature_fusion/concat_3")

    
    # Read frame
    image = rotated
    image_copy = create_img_copy(image)

    img = rotated
    h, w, _ = img.shape
    
    minX = w
    maxX = 0
    minY = h
    maxY = 0
    
    frame = img
    orig_frame = frame.copy()


    # Get frame height and width
    height_ = h
    width_ = w
    rW = width_ / float(inpWidth)
    rH = height_ / float(inpHeight)

    # Create a 4D blob from frame.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

    # Run the model
    net.setInput(blob)
    output = net.forward(outputLayers)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())


    sub_image_rects = []

    # Get scores and geometry
    scores = output[0]
    geometry = output[1]
    [boxes, confidences] = decode(scores, geometry, confThreshold)
    # Apply NMS
    indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, confThreshold,nmsThreshold)
    
    if flag==0:
        for i in indices:
            # get 4 corners of the rotated rect
            vertices = cv2.boxPoints(boxes[i[0]])
            # scale the bounding box coordinates based on the respective ratios
            for j in range(4):
                vertices[j][0] *= rW
                vertices[j][1] *= rH
            for j in range(4):
                p1 = (vertices[j][0], vertices[j][1])
                p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
                cv2.line(frame, p1, p2, (0, 255, 0), 10, cv2.LINE_AA)
                # cv2.putText(frame, "{:.3f}".format(confidences[i[0]]), (vertices[0][0], vertices[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)


            ordered_vertices = order_points(vertices)
            topleft_pt = ordered_vertices[0]
            botright_pt = ordered_vertices[2]
            topright_pt = ordered_vertices[1]

            x1,x2 = int(round(topleft_pt[0])), int(round(botright_pt[0]))
            y1,y2 = int(round(topleft_pt[1])), int(round(botright_pt[1]))
            x3,y3 = int(round(topright_pt[0])), int(round(topright_pt[1]))

            #sub_image_rects.append((x1,x2,y1,y2))    

            #print("dx=", x2-x1, " dy=", y2-y1)

            if (y2-y1) > 6:
                sub_image_rects.append((x1,x2,y1,y2))


        # Put efficiency information
        #cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        #cv2.imwrite("out-{}".format(args.input),frame)

        for sub_image_rect in sub_image_rects:
            x1,x2,y1,y2 = sub_image_rect
            sub_image = orig_frame[y1:y2, x1:x2]
            #cv2.imshow("sub", sub_image)

            minX = min(minX,x1)-1                    
            maxX = max(maxX,x2)+1
            minY = min(minY,y1)-2
            maxY = max(maxY,y2)+2

    #         text = get_text(sub_image)
    #         cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    # #         print("Got: ", text)
    #         print(text)


        if(minX<0):
                minX = 0
        if(maxX>image.shape[1]):
                maxX = image.shape[1]
        if(minY<0):
                minY = 0
        if(maxY>image.shape[0]):
                maxY = image.shape[0]

        cv2.rectangle(frame,(minX,minY),(maxX,maxY),(255,0,0),5)

        flag1 = cv2.imwrite(out_folder+"east_output.png",frame)

        crop,flag2 = crop_image(image_copy,minX,minY,maxX-minX,maxY-minY,out_folder)
        
        return(crop,flag1,flag2)


# In[20]:


def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    return angle


# In[21]:


def getOrientation(pts, img): 
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    angle = drawAxis(img, cntr, p1, (0, 255, 0), 10)
    return angle


# In[22]:



def pca(src,copy,out_folder):
    angles = []
    areas =  []
    # Check if image is loaded successfully
    if src is None:
        print('Could not open or find the image: ', args.input)
        exit(0)
    # Convert image to grayscale
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # Convert image to binary
    _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours,_ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    l,b,_ = src.shape
    AREA = l*b
    if AREA < 650 * 500:
        print('case0')
        min_area = 10
        max_area  = 100
    elif AREA < 1600 * 800:
        print('case1')
        min_area = 30
        max_area  = 200
    elif AREA < 4500 * 3500:
        print('case2')
        min_area = 100
        max_area  = 1000
    else:
        print('case3')
        min_area = 100
        max_area  = 100000
        
    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area = cv2.contourArea(c)
        areas.append(area)
        # Ignore contours that are too small or too large
        #if area < min_area or max_area < area:
        if area < 0 or max_area < 1000:    
            continue
        # Draw each contour only for visualisation purposes
    #     cv.drawContours(src, contours, i, (0, 0, 255), 2);
        # Find the orientation of each shape
        angle = getOrientation(c, src)

        rotation_number = np.degrees(angle)
        angles.append(round(rotation_number,1))
        
    
    m = stats.mode(angles)
    print(m)
    m = -(round(m[0][0])+90)
    print("mode:{}".format(m))
    print(src.shape)
    

    rotated = imutils.rotate_bound(copy, m)
#     cv.imwrite('images_temp/'+str(cnt)+'.png',src)
    flag1 = cv2.imwrite(out_folder+"pca_lines.png",src)
    flag2 = cv2.imwrite(out_folder+"pca_rotated.png",rotated)
    return rotated,flag1,flag2
#     cv.imwrite('images_temp/'+str(cnt)+'out.png',rotated)


# In[23]:
def create_outFolder(path,all_flag):
    global RESULT_FOLDER
    global ALL_FOLDER
    if all_flag:
    	out_folder =ALL_FOLDER + path.split("/")[-1].split('.')[0]+'_'
    else:
        if  os.path.exists(RESULT_FOLDER):
            shutil.rmtree(RESULT_FOLDER)
        out_folder =RESULT_FOLDER + path.split("/")[-1].split('.')[0]+'/'
    if not os.path.exists(out_folder) and not all_flag:
        os.makedirs(out_folder)
    print(out_folder)
    return out_folder


def pre_process(path,all_flag):
    print(path)
    out_folder = create_outFolder(path,all_flag)
    image = read_image(path,out_folder)
    image_copy = create_img_copy(image) 
    ############################PCA Rotation########################################################
    try:
        rotated,flag1,flag2 =  pca(image,image_copy,out_folder) 
    except Exception as e:
        print("Skipping PCA rotation")
        print(e)
        rotated = image_copy
        flag1,flag2 = True,True		
    try:
        crop,flag3,flag4 = east(rotated,0,out_folder)
    except Exception as e:
        print("Skipping crop")
        print(e)
        crop = rotated
        cv2.imwrite(out_folder+"cropped.png",crop) 
        flag3,flag4 = True,True		
    try:
        rotated3,flag5 = rotate_image3(out_folder)
    except Exception as e:
        print("Skipping rotation3")
        print(e)
        rotated3 = crop
        flag5 = cv2.imwrite(out_folder+"rotated3.png",rotated3)
    return flag1,flag2,flag3,flag4,flag5