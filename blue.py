from PIL import Image
import numpy
import cv2
from collections import defaultdict
# import pytesseract as pt


def dominant_color_my(img):
    dict = {}
    (height, width) = img.shape[:2]
    for w in range(width):
        for h in range(int(height/2 - 2), int(height/2 + 2)):
            color = img[h, w]
            color = tuple(color)
            if color not in dict:
                dict[color] = 1
            elif color in dict:
                dict[color] += 1

    return max(dict, key=tuple)

def dominant_color_num(img):
    colors, count = numpy.unique(img.reshape(-1,img.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]

def upscale(img, x, y, row, col):
    up = numpy.zeros((x*row, y*col), numpy.uint8)
    i, m = 0, 0
    while m < row:
        j, n = 0, 0
        while n < col:
            up[i, j] = img[m, n]
            j+=y
            n+=1
        m+=1
        i+=x

    return up

# im = cv2.imread('Untitled.png')
# cv2.imshow('orig', im)
# row, col, c = im.shape
# blue = im[:,:,0]
# green = im[:,:,1]
# red = im[:,:,2]
# x, y = 2, 2
# upscale_img = numpy.zeros((x*row,y*col,c),numpy.uint8)
#
# upscale_img[:,:,0] = upscale(blue, x,y,row,col)
# upscale_img[:,:,1] = upscale(green, x,y,row,col)
# upscale_img[:,:,2] = upscale(red, x,y,row,col)
# cv2.imshow('up', upscale_img)
# cv2.waitKey(0)

(W, H) = (None, None)
(newW, newH) = (640, 640)
(rW, rH) = (None, None)

layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]


print('loading EAST')
model = 'frozen_east_text_detection.pb'
net = cv2.dnn.readNet(model)

frame = cv2.imread('text.png')
# im_big2 = im.resize((1290, 726))

orig = frame.copy()

if W is None or H is None:
    (H, W) = frame.shape[:2]
    rW = W/float(newW)
    rH = H/float(newH)

frame = cv2.resize(frame, (newW, newH))


blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH),
                             (123.68, 116.78, 103.94), swapRB=True, crop=False)
print(blob.shape)
net.setInput(blob)
scores = net.forward("feature_fusion/Conv_7/Sigmoid")
geo = net.forward("feature_fusion/concat_3")
print(scores.shape)
print(geo.shape)

# (rects, confidences) = predictions(scores, geo, .55)
# boxes = non_max_suppression(numpy.array(rects), probs=confidences)
#
# for (startX, startY, endX, endY) in boxes:
#     startX = int(startX * rW)
#     startY = int(startY * rH)
#     endX = int(endX * rW)
#     endY = int(endY * rH)
#
#     paddingX = int((endX - startX) * 0.05)
#     paddingY = int((endY - startY) * 0.05)
#
#     startX = max(0, startX - paddingX)
#     startY = max(0, startY - paddingY)
#     endX = min(W, endX + (paddingX))
#     endY = min(H, endY + (paddingY))
#
#     roi = orig[startY:endY, startX:endX]
#     (h, w) = roi.shape[:2]
#     roi = roi.resize((h*2, w*2), 1)
#     cv2.rectangle(orig, (startX, startY), (endX, endY),
#               (0, 0, 255), 2)
    #
    # try:
    #     config = ("-l eng --oem 1 --psm 7")
    #     text = pt.image_to_string(roi, config=config)
    #     frame_text.append(text)
    #     for s in text:
    #         if not s.isalnum() and s != ' ':
    #             text = text.replace(s, '')
    #
    #     # if text not in sentence_list:
    #     #     print(text)
    #     #     sentence_list.append(text)
    #     # while len(sentence_list) > 50:
    #     #     sentence_list.remove(sentence_list[0])
    # except Exception:
    #     print('error')
    #     continue

# im_big1.show()
# im_big2.show()
