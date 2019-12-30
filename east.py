from imutils.object_detection import non_max_suppression
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import numpy
import cv2
import time
import pytesseract as pt

def predictions(scores, geo, threshold):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geo[0, 0, y]
        xData1 = geo[0, 1, y]
        xData2 = geo[0, 2, y]
        xData3 = geo[0, 3, y]
        anglesData = geo[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < threshold:
                continue
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = numpy.cos(angle)
            sin = numpy.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return(rects, confidences)

def separate_roi_y(box):
    boxes = box.tolist()
    boxes.sort(key=lambda x: x[1])
    separated_boxes = []
    stop = []
    for i in range(len(boxes)):
        cutoffY = boxes[i][3] - boxes[i][1]
        try:
            if abs(boxes[i][1] - boxes[i+1][1]) < cutoffY - 3:
                continue
            else:
                stop.append(i + 1)
                continue
        except:
            break

    return [boxes[i:j] for i, j in zip([0]+stop, stop+[None])]

def separate_roi_x(box):
    stop = []
    cutoffX = 15
    for arr in box:
        start = 0
        arr.sort(key=lambda x: x[0])
        # print('ind arr is: {}'.format(arr))
        for i in range(len(arr)):
            try:
                if abs(arr[i][2] - arr[i+1][0]) < cutoffX or arr[i][2] - arr[i+1][0] > 0:
                    continue
                else:
                    stop.append(arr[start:i+1])
                    start = i+1
            except IndexError:
                stop.append(arr[start:])
                break
    return stop



def meld_roi(boxes):
    final_rois = []
    for line in boxes:
        roi_line = []
        index = 0
        while index < 4:
            if len(line) > 0:
                if index < 2:
                    line.sort(key=lambda x: x[index])
                    roi_line.append(int(line[0][index]))
                    index += 1
                else:
                    line.sort(key=lambda x: x[index], reverse=True)
                    roi_line.append(int(line[0][index]))
                    index += 1
            elif len(line) == 1:
                roi_line = line
        final_rois.append(roi_line)

    return final_rois


(W, H) = (None, None)
(newW, newH) = (480, 480)
(rW, rH) = (None, None)

layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]


print('loading EAST')
model = 'frozen_east_text_detection.pb'
net = cv2.dnn.readNet(model)

print('starting stream')
vs = VideoStream(src=0).start()
time.sleep(1.0)
fps = FPS().start()
sentence_list = []
while True:
    frame = vs.read()
    if frame is None:
        break
    frame = imutils.resize(frame, width=1000)
    orig = frame.copy()

    if W is None or H is None:
        (H, W) = frame.shape[:2]
        rW = W/float(newW)
        rH = H/float(newH)

    frame = cv2.resize(frame, (newW, newH))

    blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geo) = net.forward(layerNames)

    (rects, confidences) = predictions(scores, geo, .5)
    boxes = non_max_suppression(numpy.array(rects), probs=confidences)
    if not len(boxes) == 0:
        # print('boxes is: {}\n'.format(boxes))
        separated_boxes_y = separate_roi_y(boxes)
        # print('by y is: {}\n'.format(separated_boxes_y))
        separated_boxes_x = separate_roi_x(separated_boxes_y)
        # print('by x is: {}\n'.format(separated_boxes_x))
        final = meld_roi(separated_boxes_x)
        # print('final is: {}'.format(final))
        # frame_text = []

        for (startX, startY, endX, endY) in final:
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            paddingX = int((endX - startX) * .05)
            paddingY = int((endY - startY) * .1)

            startX = max(0, startX - paddingX)
            startY = max(0, startY - paddingY)
            endX = min(W, endX + (paddingX))
            endY = min(H, endY + (paddingY))

            roi = orig[startY:endY, startX:endX]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hsv[...,1] = hsv[...,1] * 1.4
            roi = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            # color = dom_color(roi)
            # print(color)
            # text_color = sig_change(roi)
            # print(text_color)
            # if back == 0:
            #     roi = cv2.bitwise_not(roi)
            #     cv2.rectangle(orig, (startX, startY), (endX, endY),
            # 	      (255, 0, 0), 2)
            # elif all(el < 240 or el > 2:
            #     roi = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 6)
            #     cv2.rectangle(orig, (startX, startY), (endX, endY),
            # 	      (0, 0, 255), 2)
            # else:
            cv2.rectangle(orig, (startX, startY), (endX, endY),
        	      (0, 0, 255), 2)
            try:
                config = ("-l eng --oem 1 --psm 7")
                text = pt.image_to_string(roi, config=config)
                # frame_text.append(text)
                for s in text:
                    if not s.isalnum() and s != ' ':
                        text = text.replace(s, '')
                if text not in sentence_list:
                    print(text)
                    sentence_list.append(text)
                while len(sentence_list) > 50:
                    sentence_list.remove(sentence_list[0])
            except Exception:
                print('error')
                continue
    else:
        pass

    fps.update()
    cv2.imshow('detection', orig)
    key = cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break

fps.stop()
vs.stop()
cv2.destroyAllWindows()
