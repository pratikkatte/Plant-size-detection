
import cv2
import numpy as np
import core 
from imutils import perspective
from scipy.spatial import distance as dist


image = cv2.imread('data/coin10_2.jpg')
image = cv2.resize(image,(512,1024))

cv2.imshow("original image", image)

img_whitebalance = core.white_balance(image)



lab = cv2.cvtColor(img_whitebalance, cv2.COLOR_BGR2Lab)
# split image
light ,green, blue = cv2.split(lab)


ret, image_binary = cv2.threshold(green, 110, 255, cv2.THRESH_BINARY_INV)	


mask = np.copy(image_binary)


fill_image = core.fill(image_binary,1)


kernel2 = np.ones((7,7), np.uint8)
dilated = cv2.dilate(fill_image, kernel=kernel2)



mask1 = np.copy(dilated)
ori_img = np.copy(image)
object_id, obj_hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

for i, cnt in enumerate(object_id):
    cv2.drawContours(ori_img, object_id, i, (255, 102, 255), -1, lineType=8, hierarchy=obj_hierarchy)



roi, roi_hierarchy = core.rectangle(x=10,y=70, h=770, w=450, img=ori_img)

ref_img =np.copy(ori_img)
cv2.drawContours(ref_img, roi, -1, (255, 0, 0), 5)



roi_object, roi_obj_hierarchy, kept_mask, obj_area = core.roi_objects(ori_img, 'partial', roi, roi_hierarchy,object_id, obj_hierarchy)



k ,clusters_i, contours_plant = core.cluster_contours(ori_img, roi_object, 4,6)





hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h,s,v  = cv2.split(hsv)

_, contours_coin, hierarchy = cv2.findContours(s, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour_list = []
for contour in contours_coin:
    approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
    
    area = cv2.contourArea(contour)
    if ((len(approx) > 8) & (area > 150 )):
        contour_list.append(contour)

for contour in clusters_i:
    contour_list.append(contour)

print(len(contour_list))
objects_detected = image.copy()
objects_detected = cv2.drawContours(objects_detected, contour_list, -1, (255,0,0),-1)


count1 = contour_list[0]
img_cnt = image.copy()


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0])* 0.5, (ptA[1]+ptB[1])* 0.5)


pixelsPerMetric = None
orig = image.copy()
for n,c in enumerate(contour_list):
    
    #bounding box of the contour
    box = cv2.minAreaRect(c) 
    #vertices of the box
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    #sort contour
    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
    
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
        (255, 0, 255), 2)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
        (255, 0, 255), 2)
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    if pixelsPerMetric is None:
        pixelsPerMetric = dB / 0.955
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric
    print( dimA, dimB, n)
    

cv2.imshow("contour_inage", orig)
cv2.waitKey()

