"""Human facial landmark detector based on Convolutional Neural Network."""
import cv2
import numpy as np
import tensorflow as tf


class FaceDetector:
    """Detect human face from image"""

    def __init__(self,
                 dnn_proto_text='assets/deploy.prototxt',
                 dnn_model='assets/res10_300x300_ssd_iter_140000.caffemodel'):
        """Initialization"""
        ##读入网络模型（计算图模型）以及预训练参数
        self.face_net = cv2.dnn.readNetFromCaffe(dnn_proto_text, dnn_model)
        self.detection_result = None

    #返回可能性大于threshold的face的confidence_list and locationxyxy_list
    def get_faceboxes(self, image, threshold=0.5):
        """
        Get the bounding box of faces in image using dnn.
        """
        #image.shape 返回高宽位深(rbg)
        rows, cols, _ = image.shape

        confidences = []
        faceboxes = []

        #向读入的dnn网络中输入样本X
        #blobFromImage 返回一个预处理后的结果 缩放1倍 输入图像尺寸（300*300）各通道减去的值 不交换RB 不建材
        self.face_net.setInput(cv2.dnn.blobFromImage(
            image, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False))

        #计算网络输出
        detections = self.face_net.forward()
        #返回了一个1*1*200*7的向量
        #分别是 0、1、fidence、左下角x、左下角y、右上角x、右上角y
        for result in detections[0, 0, :, :]:
            confidence = result[2]
            if confidence > threshold:
                x_left_bottom = int(result[3] * cols)
                y_left_bottom = int(result[4] * rows)
                x_right_top = int(result[5] * cols)
                y_right_top = int(result[6] * rows)
                confidences.append(confidence)
                faceboxes.append(
                    [x_left_bottom, y_left_bottom, x_right_top, y_right_top])

        self.detection_result = [faceboxes, confidences]

        return confidences, faceboxes

    def draw_all_result(self, image):
        """Draw the detection result on image"""
        for facebox, conf in self.detection_result:
            cv2.rectangle(image, (facebox[0], facebox[1]),
                          (facebox[2], facebox[3]), (0, 255, 0))
            label = "face: %.4f" % conf
            label_size, base_line = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(image, (facebox[0], facebox[1] - label_size[1]),
                          (facebox[0] + label_size[0],
                           facebox[1] + base_line),
                          (0, 255, 0), cv2.FILLED)
            cv2.putText(image, label, (facebox[0], facebox[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

#用卷积网络识别脸部特征
class MarkDetector:
    """Facial landmark detector by Convolutional Neural Network"""

    def __init__(self, saved_model='assets/pose_model'):
        """Initialization"""
        # A face detector is required for mark detection.
        self.face_detector = FaceDetector()

        self.cnn_input_size = 128
        self.marks = None

        # Get a TensorFlow session ready to do landmark detection
        # Load a Tensorflow saved model into memory.
        self.graph = tf.Graph()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        #生成tf环境（计算图、参数）
        self.sess = tf.compat.v1.Session(graph=self.graph, config=config)

        # Restore model from the saved_model file, that is exported by
        # TensorFlow estimator.
        #将已有模型存到环境中
        tf.compat.v1.saved_model.loader.load(self.sess, ["serve"], saved_model)

    @staticmethod
    def draw_box(image, boxes, box_color=(255, 255, 255)):
        """Draw square boxes on image"""
        for box in boxes:
            cv2.rectangle(image,
                          (box[0], box[1]),
                          (box[2], box[3]), box_color, 3)

    @staticmethod
    def move_box(box, offset):
        """Move the box to direction specified by vector offset"""
        left_x = box[0] + offset[0]
        top_y = box[1] + offset[1]
        right_x = box[2] + offset[0]
        bottom_y = box[3] + offset[1]
        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def get_square_box(box):
        """Get a square box out of the given box, by expanding it."""
        left_x = box[0]
        top_y = box[1]
        right_x = box[2]
        bottom_y = box[3]

        box_width = right_x - left_x
        box_height = bottom_y - top_y

        # Check if box is already a square. If not, make it a square.
        diff = box_height - box_width
        delta = int(abs(diff) / 2)

        if diff == 0:                   # Already a square.
            return box
        # 瘦
        elif diff > 0:                  # Height > width, a slim box.
            left_x -= delta
            right_x += delta
            if diff % 2 == 1:
                right_x += 1
        #胖
        else:                           # Width > height, a short box.
            top_y -= delta
            bottom_y += delta
            if diff % 2 == 1:
                bottom_y += 1

        # Make sure box is always square.
        assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def box_in_image(box, image):
        """Check if the box is in image"""
        rows = image.shape[0]
        cols = image.shape[1]
        return box[0] >= 0 and box[1] >= 0 and box[2] <= cols and box[3] <= rows

    def extract_cnn_facebox(self, image):
        """Extract face area from image."""

        #获得图片中左下右上xxyy组成的list
        _, raw_boxes = self.face_detector.get_faceboxes(
            image=image, threshold=0.9)
        #对get_faceboxess得到的数据进行规格处理
        for box in raw_boxes:
            # Move box down.
            # diff_height_width = (box[3] - box[1]) - (box[2] - box[0])
            #十分之一的 高度（y方向）
            offset_y = int(abs((box[3] - box[1]) * 0.1))
            #向上移动十分之一高度
            box_moved = self.move_box(box, [0, offset_y])

            # Make box square.
            facebox = self.get_square_box(box_moved)

            if self.box_in_image(facebox, image):
                return facebox

        return None

    def detect_marks(self, image_np):
        """Detect marks from image"""
        # Get result tensor by its name.
        logits_tensor = self.graph.get_tensor_by_name(
            'layer6/final_dense:0')

        # Actual detection.
        #执行计算图 feed_dict 更改图中一个参数
        #predictions为68个特征点坐标
        predictions = self.sess.run(
            logits_tensor,
            feed_dict={'image_tensor:0': image_np})

        # Convert predictions to landmarks.
        #将【rediction】平铺为一个向量
        marks = np.array(predictions).flatten()[:136]
        #-1根据其他维度推断 所以68个特征是一个2*68的矩阵
        marks = np.reshape(marks, (-1, 2))

        return marks

    @staticmethod
    def draw_marks(image, marks, color=(255, 255, 255)):
        """Draw mark points on image"""
        for mark in marks:
            cv2.circle(image, (int(mark[0]), int(
                mark[1])), 1, color, -1, cv2.LINE_AA)
