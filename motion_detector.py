from imutils.video import VideoStream
import imutils
import time
import cv2


class MotionDetector:
    def __init__(self):
        self.min_area = 500
        self.stream = 0
        self.path = "/Users/dtrenhaile/projects/motion_detection/motion_detection_basic/md_images_camera"
        self.first_frame = None
        self.last_frame = None
        self.i = 1
        self.vs = VideoStream(src=self.stream).start()

    def main(self):
        while True:
            time.sleep(0.7)
            frame = self.vs.read()
            frame = frame

            if frame is None:
                break

            frame = imutils.resize(frame, width=500)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if self.first_frame is None:
                self.first_frame = gray
                self.last_frame = self.first_frame
                continue

            frame_delta = cv2.absdiff(self.last_frame, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

            thresh = cv2.dilate(thresh, None, iterations=2)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]

            for c in cnts:
                if cv2.contourArea(c) < self.min_area:
                    print("MOTION " + str(self.i).zfill(5))
                    # motion has been detected; insert call to desired script here
                    self.last_frame = gray
                    continue

                path = "{}/{}.jpg".format(self.path, str(self.i).zfill(5))
                cv2.imwrite(path, frame)
                self.i += 1

        self.vs.stop()
