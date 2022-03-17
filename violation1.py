import os
import cv2
import json
import math

# Delete video file after violation detection
def deleteFile(videoOutputString, bboxOutputFile):
    if os.path.exists(videoOutputString) and os.path.exists(bboxOutputFile):
        os.remove(videoOutputString)
        os.remove(bboxOutputFile)

# Video ile ilgili metadatayi txt dosyasindan al
def get_metadata(metadata_path):
    detected_boxes_file = open(metadata_path)
    detected_boxes_json_str = detected_boxes_file.read()
    detected_boxes_file.close()
    detected_boxes = json.loads(detected_boxes_json_str)
    return detected_boxes


def center_points(bboxes):
    center_points_arr = []
    for bbox in bboxes:
        (x, y, w, h) = bbox
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        center_points_arr.append((cx, cy, x, y, w, h))
    return center_points_arr

VIOLATION_NAME = "violation1"
fourcc = cv2.VideoWriter_fourcc(*'H264')

def start_violation_detection_for_video(video_id):
    video_metadata_path = "./received_output/{}.txt".format(video_id)
    video_path = "./received_output/{}.mp4".format(video_id)
    # detected_boxes = [
    #    [[frame_numarasi], [id,cx,cy,x,y,w,h], [id,cx,cy,x,y,w,h],...]
    #                   .
    #                   .
    #                   .
    #                   ]
    detected_boxes = get_metadata(video_metadata_path)

    # Video capture ac, ilk frame'i oku ve frame'i resize et.
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    height, width, channels = frame.shape
    videoWriter = cv2.VideoWriter("./final_output/{}.mp4".format(video_id), fourcc, 20.0, (width,  height), True)
    
    # Ilk frame'e ait metadata bilgisiyle ilk trackerlar olusturulur
    tracker = None
    trackers = cv2.legacy.MultiTracker_create()
    tracking_object = {}  # { object_id: (cx,cy,x,y,w,h), ... }
    for frames in detected_boxes:
        if frames[0][0] == 1:
            for i in range(len(frames)):
                if i == 0:
                    continue
                object_id, cx, cy, x, y, w, h, class_id = frames[i]
                tracker = cv2.legacy.TrackerMedianFlow_create()
                trackers.add(tracker, frame, (x, y, w, h))
                tracking_object[object_id] = (cx, cy, x, y, w, h, class_id)

    frame_counter = 1  # Frame sayisi

    # Violation icin tanimlanmis degiskenler
    count = 0  # Violation Detection icin gerekli
    violationCars = []  # Hata yapan araclarin tekrar bastirilmamasi icin kullanilan liste
    tracking_object_prev = []
    hataKontrol = {}  # hatanin sayisini id ile beraber tutar

    while True:
        # read frame continously inside while loop
        ret, frame = cap.read()
        count += 1
        if ret != True:
            cv2.destroyAllWindows()
            # violation varsa 
            if (len(violationCars)>0):
                videoWriter.release()
                cap.release()
                deleteFile(video_path, video_metadata_path)
                return (VIOLATION_NAME, tuple(violationCars))
            else:
                return (None, tuple(violationCars)) # violation yoksa

        
        # Distance'in belirli bolgelere gore degismesi icin gerekli
        cy_frame = int(height/2)

        isDetectedInFrame = False
        # Eger okunan frame'de detection yapilmis ise trackerlari guncelle
        for frames in detected_boxes:
            if frames[0][0] == frame_counter:  # Eger okunan frame'de tespit yapilmis ise calisir
                isDetectedInFrame = True
                tracker = None
                trackers = cv2.legacy.MultiTracker_create()
                tracking_object.clear()
                hataKontrol.clear()
                for i in range(len(frames)):
                    if i == 0:
                        continue
                    object_id, cx, cy, x, y, w, h, class_id = frames[i]
                    tracker = cv2.legacy.TrackerMedianFlow_create()
                    trackers.add(tracker, frame, (x, y, w, h))
                    tracking_object[object_id] = (cx, cy, x, y, w, h, class_id)

        if isDetectedInFrame:
            tracking_object_prev = tracking_object.copy()

        frame_counter += 1

        # Tracker'a yeni frame'i gondererek guncellenmis bbox bilgileri alinir
        # success false oldugunda object detection yapildigi icin zaten trackerlar otomatik guncellenecek
        # bu yuzden successi kontrol etmeye gerek yok.
        success, boxes = trackers.update(frame)

        # Asagidaki bir islemler id guncelleme islemleridir.
        center_points_cur_frame = center_points(boxes)

        tracking_obj_copy = tracking_object.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, pt2 in tracking_obj_copy.items():
            objects_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0]-pt[0], pt2[1] - pt[1])
                lookup_threshold = pt[4] / 2
                
                if distance < lookup_threshold:
                    tracking_object[object_id] = (*pt, pt2[6])
                    objects_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue


        for object in tracking_object:
            if object in tracking_object_prev:
                # tracking_object = (cx, cy, x, y, w, h)
                dist = tracking_object[object][3] - tracking_object_prev[object][3]
                if dist < -0.1:
                    if object not in hataKontrol:
                        hataKontrol[object] = (True, 1)

                    elif object in hataKontrol:
                        if hataKontrol[object][0] == True:
                            trueSayac = hataKontrol[object][1]+1
                            hataKontrol[object] = (True, trueSayac)
                        # else:
                        #    hataKontrol[object] = (True,1)

                    if(hataKontrol[object][1] >= 3 and (object not in violationCars)):
                        # ihlal işleyen araçlar tekrar yazdırılmasın
                        violationCars.append(int(object))
                        print("geri geri gitme ihlali vardır. Araç id: "+str(object))
        else:
            # mesafe pozitif çıkıyorsa: idler kontrol ediliyor ve hataKontrol sözlüğünde olan araçlara False değeri atanıyor ve True sayaçları sıfırlanıyor.
            if object in hataKontrol:
                hataKontrol.pop(object)

        tracking_object_prev = []
        if (count % 5 == 0):  # 10 frame de bir
            tracking_object_prev = tracking_object.copy()
        
        # Son olarak takip edilen objeler ekrana yazdirilir.
        for object_id, bbox in tracking_object.items():
            font = cv2.FONT_HERSHEY_PLAIN
            cx, cy, x, y, w, h, class_id = bbox
            cv2.rectangle(frame, (int(x), int(y)),
                          (int(x + w), int(y + h)), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, str(object_id),
                        (cx, cy - 7), 0, 1, (0, 0, 255), 2)
            if class_id == 2:
                cv2.putText(frame, "car",
                        (cx - int(w/2), cy- int(h/2)), 0, 1, (255, 0, 0), 2)
            elif class_id == 3:
                cv2.putText(frame, "motorbike",
                        (cx - int(w/2), cy- int(h/2)), 0, 1, (255, 0, 0), 2)
            elif class_id == 5:
                cv2.putText(frame, "bus",
                        (cx - int(w/2), cy- int(h/2)), 0, 1, (255, 0, 0), 2)
            elif class_id == 7:
                cv2.putText(frame, "car",
                        (cx - int(w/2), cy- int(h/2)), 0, 1, (255, 0, 0), 2)

        videoWriter.write(frame)

        cv2.imshow("Tracking", frame)  # frame ekranda gosterilir.
        key = cv2.waitKey(1)  # ESC tusuyla islem sonlandirilabilir.
        if key == 27:
            cv2.destroyAllWindows()
            return None

    cap.release()

# violation_name = start_violation_detection_for_video("6f14e9bd-4c51-488e-9f15-47e6fec61793")