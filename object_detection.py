import os
import math
import cv2
import socket
import numpy as np
from uuid import uuid4
from confluent_kafka import SerializingProducer
from confluent_kafka.serialization import StringSerializer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer


def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i-1]
                     for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers


def load_image(img_path):
    img = cv2.imread(img_path)
    #img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    return img, height, width, channels


class Video:
    def __init__(self, video_data, video_name, bboxes_data):
        self.video_data = video_data
        self.video_name = video_name
        self.bboxes_data = bboxes_data


def video_to_dict(video_obj, ctx):
    return dict(video_data=video_obj.video_data, video_name=video_obj.video_name, bboxes_data=video_obj.bboxes_data)


def delivery_report(err, msg):
    if err is not None:
        print("Delivery failed for User record {}: {}".format(msg.key(), err))
        return
    print('User record {} successfully produced to {} [{}] at offset {}'.format(
        msg.key(), msg.topic(), msg.partition(), msg.offset()))


def bboxList_to_bboxes_data(lst):
    newList = []
    for bbox in lst:
        newList.extend(bbox)
    return newList


schema_str = """
    {
        "namespace": "confluent.io.examples.serialization.avro",
        "name": "Video",
        "type": "record",
        "fields": [
            {"name": "video_data", "type": "bytes"},
            {"name": "video_name", "type": "string"},
            {"name": "bboxes_data", "type": ["null", {"type":"array", "items":{ "type":"array", "items":{"type":"array", "items":"int"}}, "default": []}]}
        ]
    }
    """
schema_registry_conf = {'url': 'http://localhost:8081'}
schema_registry_client = SchemaRegistryClient(schema_registry_conf)
avro_serializer = AvroSerializer(schema_str=schema_str,
                                 schema_registry_client=schema_registry_client,
                                 to_dict=video_to_dict)

producer_conf = {'bootstrap.servers': 'localhost:9092,localhost:9092',
                 'client.id': socket.gethostname(),
                 'key.serializer': StringSerializer('utf_8'),
                 'value.serializer': avro_serializer,
                 "message.max.bytes": 1024*1024*50}
producer = SerializingProducer(producer_conf)


def load_to_kafka(outputString, boxes):
    print(boxes)
    bboxes_data = boxes
    try:
        producer.poll(0.0)
        video_file = open(outputString, "rb")
        video_data = video_file.read()
        video_name = "output_video"
        video_obj = Video(video_data, video_name, bboxes_data)
        producer.produce("ExampleTopic", key=str(
            uuid4()), value=video_obj, on_delivery=delivery_report)
        video_file.close()
    except ValueError:
        print("Invalid input, discarding record...")


def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(
        608, 608), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.5:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids


def draw_labels(boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    #img = cv2.resize(img, None, fx=0.4, fy=0.4)
    cv2.imshow("Image", img)


def image_detect(img_path):
    model, classes, colors, output_layers = load_yolo()
    image, height, width, channels = load_image(img_path)
    blob, outputs = detect_objects(image, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    print("boxes")
    print(boxes)
    print("confs")
    print(confs)
    print("class_ids")
    print(class_ids)
    draw_labels(boxes, confs, colors, class_ids, classes, image)
    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break


def center_points(bboxes):
    center_points_arr = []
    for bbox in bboxes:
        (x, y, w, h) = bbox
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        center_points_arr.append((cx, cy, x, y, w, h))
    return center_points_arr


def crop_bottom_half(image):
    height, width, channels = image.shape
    cropped_img = image[int(height / 3 + 70):int(height), 0:int(2 * width / 3)]
    return cropped_img


def start_video(video_path):

    length_counter = 0
    MAX_FILE_SIZE = 20000000  # 6MB
    model, classes, colors, output_layers = load_yolo()
    #fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    cap = cv2.VideoCapture(video_path)

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    _, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.7, fy=0.7)

    height, width, channels = frame.shape
    print("width: {}".format(width))
    print("height: {}".format(height))
    cy_frame = int(height/2)

    blob, outputs = detect_objects(frame, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)

    # Video parçasının ilk frame'indeki nesnelerin bbox bilgileri kaydedilir.
    isObjectDetected = False
    isTrackerLost = False
    if len(class_ids) != 0:
        isObjectDetected = True

    tracker = None
    trackers = cv2.legacy.MultiTracker_create()

    tracking_object = {}
    track_id = 0
    first_boxes = []
    for i in indexes:
        bbox = boxes[i]
        (x, y, w, h) = bbox
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)

        tracking_object[track_id] = (cx, cy, x, y, w, h)
        first_boxes.append((track_id, cx, cy, x, y, w, h))
        track_id += 1
        tracker = cv2.legacy.TrackerMedianFlow_create()
        trackers.add(tracker, frame, tuple(bbox))

    all_bbox = []
    frame_counter = 1
    isDetectorWorked = False
    counter = 1
    outputCounter = 1
    #outputString = "./output/output{}.avi".format(outputCounter)
    outputString = "./output/output{}.mp4".format(outputCounter)
    videoWriter = cv2.VideoWriter(
        outputString, fourcc, 20.0, (width,  height), True)
    firstFrameBoxesController = True
    length_counter += 1
    while True:
        _, frame = cap.read()
        # frame = crop_bottom_half(frame) # kesilmis kisim
        if _ != True:
            cv2.destroyAllWindows()
            break
        length_counter += 1
        frame = cv2.resize(frame, None, fx=0.7, fy=0.7)
        height, width, channels = frame.shape

        # Her 40 frame'de nesne tespiti gerçekleştirilir.
        if counter % 40 == 0:
            isDetectorWorked = True
            counter = 1
            blob, outputs = detect_objects(frame, model, output_layers)
            boxes, confs, class_ids = get_box_dimensions(
                outputs, height, width)
            indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
            trackers = cv2.legacy.MultiTracker_create()
            for i in indexes:
                bbox = boxes[i]
                tracker = cv2.legacy.TrackerMedianFlow_create()
                trackers.add(tracker, frame, tuple(bbox))
        else:
            counter += 1

        success, boxes = trackers.update(frame)
        if(success == False):
            counter = 40

        center_points_cur_frame = center_points(boxes)

        tracking_obj_copy = tracking_object.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, pt2 in tracking_obj_copy.items():
            objects_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0]-pt[0], pt2[1] - pt[1])
                lookup_threshold = pt[4] / 2
                """
                if(pt[1] < cy_frame/3):
                    lookup_threshold = pt[4] / 2
                elif (pt[1] < 2*cy/3):
                    lookup_threshold = pt[4] / 2
                """
                # print(distance)
                if distance < lookup_threshold:
                    tracking_object[object_id] = pt
                    objects_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue

            # Remove IDs lost
            if not objects_exists:
                tracking_object.pop(object_id)

        # Add new IDs found
        for pt in center_points_cur_frame:
            tracking_object[track_id] = pt
            track_id += 1

        if isDetectorWorked:
            isDetectorWorked = False
            bbox_list = []
            bbox_list.append((frame_counter,))
            for object_id, pt in tracking_object.items():
                cx, cy, x, y, w, h = pt
                bbox_list.append((object_id, int(cx), int(
                    cy), int(x), int(y), int(w), int(h)))
            all_bbox.append(bbox_list)

        if firstFrameBoxesController and len(boxes) != 0:
            first_boxes.clear()
            first_boxes.append((frame_counter,))
            for object_ids, pt in tracking_object.items():
                cx, cy, x, y, w, h = pt
                first_boxes.append((object_ids, int(cx), int(
                    cy), int(x), int(y), int(w), int(h)))
            all_bbox.append(first_boxes.copy())
            firstFrameBoxesController = False

        isFileMaxSize = False
        if len(boxes) != 0:
            file_stats = os.stat(outputString)
            # video sonu.
            if(file_stats.st_size >= MAX_FILE_SIZE or length_counter == video_length):
                print("length_counter: {}".format(length_counter))
                print("video_length: {}", format(video_length))
                isFileMaxSize = True

        if (len(boxes) == 0 and isTrackerLost) or isFileMaxSize:
            isObjectDetected = False
            isTrackerLost = False
            videoWriter.release()

            # first_boxes yerine all_bbox
            load_to_kafka(outputString, all_bbox)
            frame_counter = 1
            all_bbox = []

            outputCounter += 1
            firstFrameBoxesController = True
            #outputString = "./output/output{}.avi".format(outputCounter)
            outputString = "./output/output{}.mp4".format(outputCounter)
            videoWriter = cv2.VideoWriter(
                outputString, fourcc, 20.0, (width,  height), True)
        else:
            frame_counter += 1

        if len(boxes) != 0:
            isObjectDetected = True
            isTrackerLost = True

        if isObjectDetected:
            videoWriter.write(frame)

        # frame uzerinde tespit edilen nesneler cizdirilir.
        for object_id, bbox in tracking_object.items():
            font = cv2.FONT_HERSHEY_PLAIN
            cx, cy, x, y, w, h = bbox
            # label = str("car")
            color = colors[0]
            cv2.rectangle(frame, (int(x), int(y)),
                          (int(x + w), int(y + h)), color, 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, str(object_id),
                        (cx, cy - 7), 0, 1, (0, 0, 255), 2)
            # cv2.putText(frame, label, (int(x), int(y + 30)),
            #            font, 3, color, 3)

        #boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        #draw_labels(boxes, confs, colors, class_ids, classes, frame)

        cv2.imshow("Tracking", frame)  # frame ekranda gosterilir.
        key = cv2.waitKey(1)  # ESC tusuyla islem sonlandirilabilir.
        if key == 27:
            cv2.destroyAllWindows()
            break
    cap.release()
    producer.flush()


# image_detect("insan4.jpg")
start_video("./videos/video1.mp4")
