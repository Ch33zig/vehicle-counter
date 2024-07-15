from scipy import spatial
import numpy as np
import time
import os
import cv2
import argparse

FRAMES_BEFORE_CURRENT = 10
INPUT_WIDTH, INPUT_HEIGHT = 416, 416
LIST_OF_VEHICLES = ['bicycle', 'car', 'motorbike', 'bus', 'truck', 'train']

def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True,
        help='path to input video')
    ap.add_argument('-o', '--output', required=True,
        help='path to output video')
    ap.add_argument('-c', '--confidence', type=float, default=0.5,
        help='minimum probability to filter weak detections')
    ap.add_argument('-t', '--threshold', type=float, default=0.3,
        help='threshold when applying non-maxima suppression')

    args = vars(ap.parse_args())
    return args['input'], args['output'], args['confidence'], args['threshold']

INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH, PRE_DEFINED_CONFIDENCE, PRE_DEFINED_THRESHOLD = parse_arguments()

labels_path = os.path.join('yolo-coco', 'coco.names')
LABELS = open(labels_path).read().strip().split('\n')

weights_path = os.path.join('yolo-coco', 'yolov3.weights')
config_path = os.path.join('yolo-coco', 'yolov3.cfg')
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')

def display_vehicle_count(frame, vehicle_count):
    cv2.putText(
        frame,
        f'Detected Vehicles: {vehicle_count}',
        (20, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0xFF, 0),
        2,
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
    )

def display_fps(start_time, num_frames):
    current_time = int(time.time())
    if current_time > start_time:
        print(f'FPS: {num_frames}')
        return current_time, 0
    return start_time, num_frames

def draw_detection_boxes(idxs, boxes, class_ids, confidences, frame):
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[class_ids[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f'{LABELS[class_ids[i]]}: {confidences[i]:.4f}'
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def initialize_video_writer(video_width, video_height, video_stream):
    source_video_fps = video_stream.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    return cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, source_video_fps, (video_width, video_height), True)

def box_in_previous_frames(previous_frame_detections, current_box, current_detections):
    center_x, center_y, width, height = current_box
    dist = np.inf
    frame_num = coord = None

    for i in range(FRAMES_BEFORE_CURRENT):
        coordinate_list = list(previous_frame_detections[i].keys())
        if not coordinate_list:
            continue

        temp_dist, index = spatial.KDTree(coordinate_list).query([(center_x, center_y)])
        if temp_dist < dist:
            dist = temp_dist
            frame_num = i
            coord = coordinate_list[index[0]]

    if dist > (max(width, height) / 2):
        return False

    current_detections[(center_x, center_y)] = previous_frame_detections[frame_num][coord]
    return True

def count_vehicles(idxs, boxes, class_ids, vehicle_count, previous_frame_detections, frame):
    current_detections = {}
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            center_x = x + (w // 2)
            center_y = y + (h // 2)

            if LABELS[class_ids[i]] in LIST_OF_VEHICLES:
                current_detections[(center_x, center_y)] = vehicle_count 
                if not box_in_previous_frames(previous_frame_detections, (center_x, center_y, w, h), current_detections):
                    vehicle_count += 1

                id = current_detections.get((center_x, center_y))
                if list(current_detections.values()).count(id) > 1:
                    current_detections[(center_x, center_y)] = vehicle_count
                    vehicle_count += 1 

                cv2.putText(frame, str(id), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 2)

    return vehicle_count, current_detections

def main():
    print('Loading YOLO from disk...')
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    video_stream = cv2.VideoCapture(INPUT_VIDEO_PATH)
    video_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    previous_frame_detections = [{(0,0):0} for _ in range(FRAMES_BEFORE_CURRENT)]
    num_frames, vehicle_count = 0, 0
    writer = initialize_video_writer(video_width, video_height, video_stream)
    start_time = int(time.time())

    while True:
        os.system('clear')
        num_frames += 1
        print(f'FRAME:\t {num_frames}')
        
        start_time, num_frames = display_fps(start_time, num_frames)
        (grabbed, frame) = video_stream.read()

        if not grabbed:
            break

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(ln)
        
        boxes, confidences, class_ids = [], [], []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > PRE_DEFINED_CONFIDENCE:
                    box = detection[0:4] * np.array([video_width, video_height, video_width, video_height])
                    (center_x, center_y, width, height) = box.astype('int')
                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, PRE_DEFINED_CONFIDENCE, PRE_DEFINED_THRESHOLD)

        draw_detection_boxes(idxs, boxes, class_ids, confidences, frame)
        vehicle_count, current_detections = count_vehicles(idxs, boxes, class_ids, vehicle_count, previous_frame_detections, frame)
        display_vehicle_count(frame, vehicle_count)
        writer.write(frame)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break	
        
        previous_frame_detections.pop(0) 
        previous_frame_detections.append(current_detections)

    print('Cleaning up...')
    writer.release()
    video_stream.release()

if __name__ == '__main__':
    main()