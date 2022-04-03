import numpy as np
import os
import face_recognition as frecog
import cv2
from datetime import datetime

path = 'face/ImagesForAttendance'
pic_names_List = []
picturesNames = []
pictures = []
picturesList = os.listdir(path)

for cl in picturesList:
    ImgCurrent = cv2.imread(f'{path}/{cl}')
    pictures.append(ImgCurrent)
    picturesNames.append(os.path.splitext(cl)[0])

def checkEncodings(pictures):
    enList = []
    for pic in pictures:
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        encode = frecog.face_encodings(pic)[0]
        enList.append(encode)
    return enList

def Attendance_Mark(name):
    with open('face/AttendanceMark.csv', 'r+') as fli:
        myData = fli.readlines()
        for l in myData:
            ent = l.split(',')
            pic_names_List.append(ent[0])
        if name not in pic_names_List:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            fli.writelines(f'\n{name},{dtString}')

enKnown = checkEncodings(pictures)
print('Encoding Completed Successfully')

captr = cv2.VideoCapture(0)
infi = 0

while infi < 10:
    fnd, pic = captr.read()
    imgS = cv2.resize(pic, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    fcFrame = frecog.face_locations(imgS)
    enFrame = frecog.face_encodings(imgS, fcFrame)
    fc_zip = zip(enFrame, fcFrame)
    for Face_encode, Face_Location in fc_zip:
        mtch = frecog.compare_faces(enKnown, Face_encode)
        distance_Face = frecog.face_distance(enKnown, Face_encode)
        mtchInd = np.argmin(distance_Face)
        if mtch[mtchInd]:
            name = picturesNames[mtchInd].upper()
            y1, x2, y2, x1 = Face_Location
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(pic, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(pic, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(pic, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            Attendance_Mark(name)
    cv2.imshow('Frame', pic)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()





