import numpy as np
import imutils as imu
import cv2 as opcv
from tensorflow.keras.models import load_model
from imutils.video import VideoStream as vdst
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import imutils
from PIL import ImageGrab
import time
import pyautogui
from random import randint
import random
import string

path_protoText = r"mask\face_detector\deploy.prototxt"
path_weights = r"mask\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
Netfc = opcv.dnn.readNet(path_protoText, path_weights)
networkMask = load_model("mask\mask_detector.model")


def DetectMask(frm, Netfc, networkMask):
	prd = []
	Inputfaces = []
	locates = []
	(height, width) = frm.shape[:2]
	blb_ = opcv.dnn.blobFromImage(frm, 1.0, (224, 224), (104.0, 177.0, 123.0))
	Netfc.setInput(blb_)
	recognizations_ = Netfc.forward()
	print(recognizations_.shape)

	for shp in range(0, recognizations_.shape[2]):
		conf = recognizations_[0, 0, shp, 2]
		point_five = 0.5
		if point_five < conf:
			bx = recognizations_[0, 0, shp, 3:7] * np.array([width, height, width, height])
			(X_st, Y_st, X_end, Y_end) = bx.astype("int")
			hminus = height - 1
			wminus = width - 1
			(X_st, Y_st) = (max(0, X_st), max(0, Y_st))
			(X_end, Y_end) = (min(wminus, X_end), min(hminus, Y_end))
			Fcdt = frm[Y_st:Y_end, X_st:X_end]
			Fcdt = opcv.cvtColor(Fcdt, opcv.COLOR_BGR2RGB)
			Fcdt = opcv.resize(Fcdt, (224, 224))
			Fcdt = img_to_array(Fcdt)
			Fcdt = preprocess_input(Fcdt)
			Inputfaces.append(Fcdt)
			locates.append((X_st, Y_st, X_end, Y_end))

	Inputfaces_len = len(Inputfaces)
	if Inputfaces_len > 0:
		Inputfaces = np.array(Inputfaces, dtype="float32")
		prd = networkMask.predict(Inputfaces, batch_size=32)

	return (locates, prd)

print("____________ TURNING THE CAMERA ON TO DETECT MASK ____________")
vs = vdst(src=0).start()
# while True:
#     img = ImageGrab.grab(bbox=(0, 1000, 100, 1100))
#     img_np = np.array(img)
#     frame = opcv.cvtColor(img_np, opcv.COLOR_BGR2GRAY)
#     opcv.imshow("frame", frame)
#     if opcv.waitKey(1) & 0Xff == ord('s'):
#         break
infi = 1
while infi < 10:
	frm = vs.read()
	frm = imu.resize(frm, width=400)
	(locates, prd) = DetectMask(frm, Netfc, networkMask)
	zp = zip(locates, prd)
	for (bx, pred) in zp:
		(X_st, Y_st, X_end, Y_end) = bx
		(mask, wo_mask) = pred
		if wo_mask < mask:
			label = "MASK"
		else:
			label = "NO MASK"
		if label == "MASK":
			color = (0, 255, 0)
		else:
			color = (0, 0, 255)
		label = "{}: {:.2f}%".format(label, max(mask, wo_mask) * 100)
		opcv.putText(frm, label, (X_st, Y_st - 10),
			opcv.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		opcv.rectangle(frm, (X_st, Y_st), (X_end, Y_end), color, 2)

	opcv.imshow("Mask Detection", frm)
	key = opcv.waitKey(1) & 0xFF
	cnt = 0
	if key == ord("q"):
		break

# id = randint(1,10000)

# random.seed(10)
# letters = string.ascii_lowercase
# rand_letters = random.choices(letters,k=1)
	

# time.sleep(5)
# screenshot = pyautogui.screenshot()
# screenshot.save(f'./face/ImagesForAttendance/student.jpg')


opcv.destroyAllWindows()
vs.stop()