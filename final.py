import pyfirmata, pyfirmata.util
import cv2 as cv
import time
import csv
import os

def clickimage(counter):
    cam = cv.VideoCapture(0) 
    s, img = cam.read()
    # cv.namedWindow("cam-test")
    # cv.imshow("cam-test",img)
    cv.imwrite(f'image__{counter}.jpg',img)
    # cv.waitKey(1)
    import csv
    
    
    return img




def ocr(counter):
    import cv2
    from matplotlib import pyplot as plt
    import numpy as np
    import imutils
    import easyocr

    # --------------------------------
    img = cv2.imread(f'image__{counter}.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    cv.imwrite('1_gray.jpg',gray)
    # plt.show()
    # --------------------------------------
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
    edged = cv2.Canny(bfilter, 30, 200) #Edge detection
    cv.imwrite('2_edged.jpg',edged)
    # plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
    # plt.show()
    # --------------------------------------

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # --------------------------------------------

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    # --------------------------------------------------

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0,255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    cv.imwrite('3_new_image.jpg',new_image)
    # plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    # plt.show()



    # ----------------------------------------------------

    (x,y) = np.where(mask==255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]

    cv.imwrite('4_cropped.jpg',cropped_image)
    # plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    # plt.show()
    # ------------------------------------------------------
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    lenofarr=len(result)
    if lenofarr!=0:
        print(result)
        seconds = time.time()
        local_time = time.ctime(seconds)
        with open('newcsv.csv','a',newline='') as f:
            thewriter = csv.writer(f)
            thewriter.writerow([local_time,result])
    else:
        print("Numberplate not detected")

    # --------------------------------------------------------

    # text = result
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
    # res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
    # plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    # plt.show()




board = pyfirmata.Arduino('COM6', baudrate=57600)

iterator = pyfirmata.util.Iterator(board)
print("handshake finished")
iterator.start()

# time.sleep(1)

button = board.get_pin('d:2:i')

# time.sleep(1)
button.enable_reporting()
# delay for a second
# time.sleep(1)
counter=0
while(True):
    # print("Button state: %s" % button.read())
    # The return values are: True False, and None
    board.digital[8].write(1)
    time.sleep(3)
    board.digital[8].write(0)
    board.digital[6].write(1)
    time.sleep(3)
    board.digital[6].write(0)
    board.digital[4].write(1)
    x=board.digital[4].read()
    timeout = time.time() + 15

    while(True):
        if time.time() > timeout:
            break

        if str(button.read()) == 'True' and x==1:
            time.sleep(0.5)
            print("crossed")
            
            img= clickimage(counter)
            ocr(counter)
            counter+=1
            time.sleep(1)
        elif str(button.read()) == 'False':
            os.system('CLS')
            print("scanning...")

        else: 
            print("00")
    board.digital[4].write(0)
   


board.exit()



