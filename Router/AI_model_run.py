from flask import Flask, render_template, Response
import io

from .ai_model import AI_model_func, vision
import cv2, glob
import numpy as np
import pandas as pd

###################################################

import onnx
from .ai_model.vision.utils import box_utils_numpy as box_utils
from caffe2.python.onnx import backend

# onnx runtime
import onnxruntime as ort

###################################################

from deepface import DeepFace

#app = Flask(__name__)

onnx_path = "./Router/ai_model/models/onnx/version-RFB-320.onnx"

session, input_name = AI_model_func.set_session(onnx_path)

representations = AI_model_func.get_representation(AI_model_func.get_db_images(0, "./Router/ai_model/itzy"), "Facenet",
                                                   enforce_detection=False)

# cap = cv2.VideoCapture("./ITZY_ICY.mp4")  # capture from camera
cap = cv2.VideoCapture(0)

threshold = 0.7

def gen():
    while True:
        ret, orig_image = cap.read()

        if ret and True:
            image = AI_model_func.get_preprocess_image(orig_image, (320, 240))

            confidences, boxes = session.run(None, {input_name: image})

            boxes, labels, probs = AI_model_func.get_face_area(orig_image.shape[1], orig_image.shape[0], confidences, boxes,
                                                 threshold)

            for box in boxes:
                try:
                    face = orig_image[box[1] - 30: box[3] + 30, box[0] - 20: box[2] + 20]

                    cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

                    name = AI_model_func.face_recognition(face, representations)

                    orig_image = cv2.putText(orig_image, name, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                             (255, 0, 0),
                                             2)
                except Exception as e:
                    print(e)

        encode_return_code, image_buffer = cv2.imencode('.jpg', orig_image)
        io_buf = io.BytesIO(image_buffer)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + io_buf.read() + b'\r\n')




# if __name__ == "__main__":
#     app.run(host='0.0.0.0', debug=True, threaded=True)
