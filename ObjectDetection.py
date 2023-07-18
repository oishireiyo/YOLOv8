# Standard modules
import os
import sys
import math
import time
import pprint

# Logging
import logging
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
handler_format = logging.Formatter('%(asctime)s : [%(name)s - %(lineno)d] %(levelname)-8s - %(message)s')
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

# Advanced modules
import numpy as np
from ultralytics import YOLO
import cv2
from Yolohelper import number_to_name

class ObjectDetection(object):
    def __init__(self, input_video: str, output_video: str, video_gen: bool = True) -> None:
        # Input video attributes
        self.input_video_name = input_video
        self.input_video = cv2.VideoCapture(input_video)

        self.length = int(self.input_video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.size = (self.width, self.height)

        self.fps = self.input_video.get(cv2.CAP_PROP_FPS)

        # Output video attributes
        if video_gen:
            self.output_video_name = output_video
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            self.output_video = cv2.VideoWriter(output_video, fourcc, self.fps, self.size)

        # YOLOv8
        self.yolo_architecture = 'yolov8n'
        self.yolo_model = YOLO('%s.pt' % (self.yolo_architecture))

        # Print
        self._print()

    def _print(self) -> None:
        logger.info('--------------------------------------------------')
        logger.info('Input Video:')
        logger.info('  Name: {}'.format(self.input_video_name))
        logger.info('  Length: {} sec'.format(self.length / self.fps))
        logger.info('  Count: {}'.format(self.length))
        logger.info('  Size: {}'.format(self.size))
        logger.info('  FPS: {}'.format(self.fps))
        logger.info('Output Video:')
        logger.info('  Enable: {}'.format(hasattr(self, output_video)))
        logger.info('  Name: {}'.format(self.output_video_name))
        logger.info('--------------------------------------------------')

    def decorata_frame(self, frame: np.ndarray, results: list) -> None:
        for result in results:
            boxes = result.boxes
            for bbox, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
                lefttop = (int(bbox[0]), int(bbox[3]))
                rightbottom = (int(bbox[2]), int(bbox[1]))
                cv2.rectangle(frame, lefttop, rightbottom, (255, 0, 0), 1)
                cv2.putText(frame, '%s (%.4f)' % (number_to_name[int(cls)], conf), lefttop,
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

    def detect_one_frame(self, iframe: int) -> np.ndarray:
        self.input_video.set(cv2.CAP_PROP_POS_FRAMES, iframe)
        ret, frame = self.input_video.read()
        logger.info('Frame: %4d / %d, read frame: %s' % (iframe, self.length, ret))
        if ret:
            results = self.yolo_model(frame, verbose=False, stream=True)
            self.decorata_frame(frame, results)
        return frame

    def detect_given_frames(self, tuple: tuple) -> None:
        for i in range(*tuple):
            frame = self.detect_one_frame(iframe=i)
            self.output_video.write(frame)

    def detect_all_frames(self) -> None:
        for i in range(self.length):
            frame = self.detect_one_frame(iframe=i)
            self.output_video.write(frame)

    def release_all(self) -> None:
        self.output_video.release()
        self.input_video.release()
        cv2.destroyAllWindows()
        logger.info('All jobs done.')

if __name__ == '__main__':
    start_time = time.time()

    input_video = '../Inputs/Solokatsu.mp4'
    output_video = '../Outputs/Solokatsu_ObjectDetector.mp4'

    Detector = ObjectDetection(input_video=input_video, output_video=output_video, video_gen=True)
    Detector.detect_all_frames()
    Detector.release_all()

    end_time = time.time()
    logger.info('Duration: %.4f sec' % (end_time - start_time))