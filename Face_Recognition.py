#Face Detector via OpenCV and DNN
import argparse
import time
from os.path import exists
from urllib.request import urlretrieve
import cv2
import numpy as np
from imutils.video import WebcamVideoStream

prototxt = "deploy.prototxt"
caffemodel = "res10_300x300_ssd_iter_140000.caffemodel"


#Download DNN
if not exists(prototxt) or not exists(caffemodel):
    urlretrieve(f"https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/{prototxt}", prototxt)
    urlretrieve(f"https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/{caffemodel}",caffemodel)
#Modul initialization
net = cv2.dnn.readNetFromCaffe(prototxt=prototxt, caffeModel=caffemodel)


def detect(img, min_confidence=0.5):
    (h, w) = img.shape[:2]#Get size of img

    #Modeling
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    #Get result of training
    net.setInput(blob)
    detectors = net.forward()


    rects = []
    for i in range(0, detectors.shape[2]):
        #Get confidence values
        confidence = detectors[0, 0, i, 2]

        #Filter out lower confidences 
        if confidence < min_confidence:
            continue

        box = detectors[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x0, y0, x1, y1) = box.astype("int")
        rects.append({"box": (x0, y0, x1 - x0, y1 - y0), "confidence": confidence})
    return rects


#Main
#Arguments Initialization
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter detecteions")
args = vars(ap.parse_args())

#Get WebCam
vs = WebcamVideoStream(src=0).start()
time.sleep(2.0)
start = time.time()
fps = vs.stream.get(cv2.CAP_PROP_FPS)

#Loop over the detections
while True:
    frame = vs.read()
    rects = detect(frame, args["confidence"])


    for rect in rects:
        (x, y, w, h) = rect["box"]
        confidence = rect["confidence"]

        #Plot Frame Box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #Label
        text = f"YOU"
        y = y - 10 if y - 10 > 10 else y + 10
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)

    #Show FPS value
    end = time.time()
    cv2.putText(frame, f"FPS: {str(int(1 / (end - start)))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255, 255, 255), 2)
    start = end

    #Show the frame to our screen
    cv2.imshow("Frame", frame)
    #if the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
#Cleanup the camera and close any open windows
cv2.destroyAllWindows()
vs.stop()
