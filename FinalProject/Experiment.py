#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
import pytesseract
from mtgsdk import Card
from mtgsdk import Set



def showIm(img):
    plt.axis('off')
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)) #plt is rgb, cv2 is bgr
    plt.show()
    
    
def showImResult(x, img, nameimg, name):
    
    top = int(0.02 * img.shape[0])  # shape[0] = rows
    bottom = top
    left = int(0.02 * img.shape[1])  # shape[1] = cols
    right = left
    borderType = cv.BORDER_CONSTANT
    value = None
    
    if(name == None):
        value = [0,0,196]
    else:
        value = [45, 196, 0]
        
    img = cv.copyMakeBorder(img, top, bottom, left, right, borderType, None, value)
    
    plt.figure(figsize=[30,15]);
    if(name != None):
        plt.title(name)
    else:
        plt.title("Unidentified Card")
    ax = plt.axes()
    plt.subplot(121); plt.axis('off'); plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)); plt.title("Original Scan");
    #if(nameimg):
    try:
        plt.subplot(122); plt.axis('off'); plt.imshow(cv.cvtColor(nameimg, cv.COLOR_BGR2RGB)); plt.title("Detected Name Plate");
    except:
        print('Card Name Plate was not located')
    
    plt.show()
    cv.imwrite(('./test_results/test' + str(x+1) + 'Result.jpg'), img)
    cv.imwrite(('./test_results/test' + str(x+1) + 'NamePlate.jpg'), nameimg)
    


# In[2]:


# SHOULD be split up into more functions, but it's late in the day, metaphorically
def findName(filepath):
    img = cv.imread(filepath)
    text = None

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #_, img_thresh = cv.threshold(img_gray, 125,255, cv.THRESH_BINARY)
    #_, img_thresh = cv.threshold(img_gray, 155,255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    img_thresh = cv.adaptiveThreshold(img_gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,13,2)
    #img_thresh = cv.adaptiveThreshold(img_gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,15,2)
    kernel = np.ones((3,3),np.uint8)

    dst = cv.Laplacian(img_thresh, -1, ksize=13)
    dst = cv.morphologyEx(dst,cv.MORPH_CLOSE,kernel)
    dst = cv.morphologyEx(dst, cv.MORPH_OPEN, kernel)

    contours, h = cv.findContours(dst, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_KCOS)
    culledContours = []
    for x in contours:
        #print(cv.contourArea(x))
        contA = cv.contourArea(x)
        if(contA > 1000 and contA < 1000000):
            #print(cv.contourArea(x))
            culledContours.append(x)
            #print(cv.arcLength(x, False))


    drawn = img.copy()
    cv.drawContours(drawn, culledContours, -1, (255,255,255), 3)
    #print(len(culledContours))
    for i in culledContours:
        rect = cv.minAreaRect(i)
        box = cv.boxPoints(rect)
        box = np.int0(box)

        width = int(rect[1][0])
        height = int(rect[1][1])
        src_pts = box.astype("float32")

        dst_pts = np.array([[0, height-1],
                           [0,0],
                           [width-1, 0],
                           [width-1, height-1]], dtype="float32")

        M = cv.getPerspectiveTransform(src_pts,dst_pts)

        warped = cv.warpPerspective(img, M, (width, height))
        #showIm(warped)
        flipped = False
        if(height > width):
            #warped = Rotate(warped,90)
            flipped = True
            temp = height
            height = width
            width = temp
            warped = cv.rotate(warped, cv.ROTATE_90_CLOCKWISE)
        if(abs((width/height) - (734/64)) > 1):
            continue
        crop = warped.copy()
        crop = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
        _, crop = cv.threshold(crop, 150,255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        #crop = 255 - crop
        #print(crop)
        crop = crop[0:int(height * 1),0:int(width * .8)]
        #showIm(crop)
        #crop = cv.Laplacian(crop, -1, ksize=7)
        #crop = cv.morphologyEx(crop, cv.MORPH_CLOSE, kernel)
        crop = 255 - crop
        crop = cv.morphologyEx(crop, cv.MORPH_OPEN, kernel+8)
        crop = 255 - crop
        #crop = cv.morphologyEx(crop, cv.MORPH_CLOSE, kernel+2)
        
        #showIm(crop)
        newtext = None
        try:
            #print("Inside try")
            #showIm(crop)
            newtext = (pytesseract.image_to_string(crop, timeout=0, output_type=pytesseract.Output.STRING).strip()).strip("[]—{}‘' (:)-,.|\n")
            if('’' in newtext):
                newtext = newtext.replace("’", "'")
                newtext = newtext.replace("''", "'")
            #print(text)
            #print("newtext = " + newtext)
            if(text != None):
                text = newtext[len(text):]
            else:
                text = newtext
        except:
            #print("except")
            continue
        #print("after try")
        if(text != None):
            #print(len(text))
            # This is not an efficient way to filter the Type box out, buuuuuut...
            if("\n" in text or "Equipment" in text or "Land" in text or "Creature" in text or "Enchantment" in text or "Artifact" in text or "Instant" in text or "instant" in text or len(text) < 3):
                text = None
            elif(("Equipment" not in text and "Land" not in text and "Creature" not in text and "Enchantment" not in text and "Artifact" not in text and "Sorcery" not in text and "Instant" not in text and "instant" not in text) and len(text) < 30):
                #print("wtf")
                #showIm(crop)
                #print(text)
                break
    card = None
    
    #print("printing text")
    #print("what is happening?" + str(text))
    #print(type(text))
    #print(text.shape)
    if(text and len(text) < 30):
        #print("API call")
        card = Card.where(name=text).all()
    if(card == None or len(card) == 0):
        print("Card could not be detected")
        #print(text)
        return None, crop
    if(card[0].name != text):
        print("Card could not be detected")
        #print(text)
        return None, crop
    print("Detected card name is: " + text)
    #print(card[0].name)
    #print(card[0].image_url)
    return card[0], crop


# In[ ]:


#card = findName('./test_assets/test3.jpg')

#if(card!=None):
    #print(card.name)
    
totalTests = 25
correct = 0
cards = []
namePlates = []
for x in range(totalTests):
    filename = './test_assets/test' + str(x+1) + '.jpg'
    card, namePlate = findName(filename)
    if(card != None):
        correct += 1
        cards.append(card)
        #namePlates.append(namePlate)
    else:
        cards.append(None)
        #namePlates.append(None)
    namePlates.append(namePlate)
print(str(correct) + ' out of ' + str(totalTests) + ' cards correctly identified.')
print("Success rate of " + str(float(100 * (correct/totalTests))) + "%")
    


# In[ ]:





# In[ ]:


for x in range(totalTests):
    card = cards[x]
    name = None
    if(card != None):
        name = card.name
    filename = './test_assets/test' + str(x+1) + '.jpg'
    img = cv.imread(filename)
    if(x < 16):
        showImResult(x,img,namePlates[x], name)


# In[ ]:




