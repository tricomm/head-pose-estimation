"""Demo code shows how to estimate human head pose.
Currently, human face is detected by a detector from an OpenCV DNN module.
Then the face box is modified a little to suits the need of landmark
detection. The facial landmark detection is done by a custom Convolutional
Neural Network trained with TensorFlow. After that, head pose is estimated
by solving a PnP problem.
"""
from argparse import ArgumentParser
from multiprocessing import Process, Queue

import math
import cv2
import numpy as np

from mark_detector import MarkDetector
from os_detector import detect_os
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer

print("OpenCV version: {}".format(cv2.__version__))

# multiprocessing may not work on Windows and macOS, check OS for safety.
detect_os()

CNN_INPUT_SIZE = 128

# Take arguments from user input.
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=None,
                    help="The webcam index.")
parser.add_argument("--image",type=str,default=None,
                    help="image file to be processed")
args = parser.parse_args()

#专为多线程编写的函数
def get_face(detector, img_queue, box_queue):
    """Get face from image queue. This function is used for multiprocessing"""
    #untial process.terminate
    while True:
        image = img_queue.get()
        box = detector.extract_cnn_facebox(image)
        box_queue.put(box)


def main():
    """MAIN"""
    # Video source from webcam or video file.
    #读一帧图片从视频cap中
    image_src = args.image

    sample_frame = cv2.imread(image_src)


    # Introduce mark_detector to detect landmarks.
    #类声明 来自于mark——detector.py 由dnn网络构造的面部特征识别器
    mark_detector = MarkDetector()




    # Introduce pose estimator to solve pose. Get one frame to setup the
    # estimator according to the image size.
    height, width = sample_frame.shape[:2]
    #可处理不同大小图片 作为构造函数参数传入
    pose_estimator = PoseEstimator(img_size=(height, width))

    # Introduce scalar stabilizers for pose.
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    #计时器
    tm = cv2.TickMeter()
    # kill -9 $(pidof ethminer)
    #/home / zhangsiyu / ethminer/autorestart.sh
    #不断读取摄像头数据

    # Crop it if frame is larger than expected.
    # frame = frame[0:480, 300:940]

    # If frame comes from webcam, flip it so it looks like a mirror.k
    #翻转


    # Pose estimation by 3 steps:
    # 1. detect face;
    # 2. detect landmarks;
    # 3. estimate pose

    # Feed frame to image queue.

    # Get face from box queue.
    facebox = mark_detector.extract_cnn_facebox(sample_frame)

    if facebox is not None: #如果检测到了脸
        # Detect landmarks from image of 128x128.
        #切割图片保留facebox中的部分
        face_img = sample_frame[facebox[1]: facebox[3],
                         facebox[0]: facebox[2]]
        #插值或抽像素到指定大小
        face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        tm.start()
        #返回2*68的脸部特征
        marks = mark_detector.detect_marks([face_img])
        tm.stop()

        # Convert the marks locations from local CNN to global image.
        #右上x-左下x
        #marks R(68*2)
        marks *= (facebox[2] - facebox[0])
        marks[:, 0] += facebox[0]
        marks[:, 1] += facebox[1]

        # Uncomment following line to show raw marks.
        # mark_detector.draw_marks(
        #     frame, marks, color=(0, 255, 0))

        # Uncomment following line to show facebox.
        # mark_detector.draw_box(frame, [facebox])

        # Try pose estimation with 68 points.
        pose = pose_estimator.solve_pose_by_68_points(marks) #返回rotation_vector, translation_vector

        # Stabilize the pose.
        steady_pose = []
        pose_np = np.array(pose).flatten()
        for value, ps_stb in zip(pose_np, pose_stabilizers):
            ps_stb.update([value])
            steady_pose.append(ps_stb.state[0])
        steady_pose = np.reshape(steady_pose, (-1, 3))

        # Uncomment following line to draw pose annotation on frame.
        # pose_estimator.draw_annotation_box(
        #     frame, pose[0], pose[1], color=(255, 128, 128))



        # Uncomment following line to draw head axes on frame.
        # pose_estimator.draw_axes(frame, stabile_pose[0], stabile_pose[1])
        roll = float(pose[0][2])/math.pi*180+180
        yaw = float(pose[0][0])/math.pi*180
        pitch =float(pose[0][1])/math.pi*180
        cv2.putText(sample_frame, "roll: " + "{:7.2f}".format(roll), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 0), thickness=2)
        cv2.putText(sample_frame, "pitch: " + "{:7.2f}".format(pitch), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 0), thickness=2)
        cv2.putText(sample_frame, "yaw: " + "{:7.2f}".format(yaw), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 0), thickness=2)
        cl = (128, 255, 128)
        if yaw<-30 or yaw>30:
            cv2.putText(sample_frame, "please watch your text paper", (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 255), thickness=2)
            cl=(0,0,255)
        # Uncomment following line to draw stabile pose annotation on frame.
        pose_estimator.draw_annotation_box(
            sample_frame, pose[0], pose[1], color=cl)
    else:
        cv2.putText(sample_frame, "not detected face", (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 255), thickness=2)
    # Show preview.
    cv2.imshow("Preview", sample_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
