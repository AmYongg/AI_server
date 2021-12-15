import cv2, glob
import numpy as np
import pandas as pd

###################################################

import onnx
import Router.ai_model.vision.utils.box_utils_numpy as box_utils
from caffe2.python.onnx import backend

# onnx runtime
import onnxruntime as ort

###################################################

from deepface import DeepFace

#####################################################################################################

def get_db_images( df_class, db_path ):
    dirs = glob.glob(db_path+"/*.*")
    imgs = [ ( d.split("\\")[-1].split(".")[0][:-1] ,cv2.imread(d,1) ) for d in dirs ]
    #print( [ i[0] for i in imgs ] )
    return imgs

#####################################################################################################

def get_preprocess_image(image, size=(320, 240) ):
    image = cv2.resize(image[:,:,::-1], size, fx=-1, fy=-1, interpolation=cv2.INTER_AREA)
    image = image / 128 - 0.5
    image = np.transpose(image, [2, 0, 1]) # channel first
    image = np.expand_dims(image, axis=0) # one image
    image = image.astype(np.float32)

    return image

#####################################################################################################

def get_face_area(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

#####################################################################################################

def face_recognition( face, representations=None, class_info=None ):
    if representations is None:
        print("No DB Image")
        return "unknown"
    else :
        df = DeepFace.custom(img_path = face, representations=representations, enforce_detection=False, model_name="Facenet")
        #df = DeepFace.find(img_path = face, db_path = "./itzy", model_name="Facenet", enforce_detection=False)

        #name = list( set( [ n[7:-5]for n in df["identity"].values] ) )
        #name = list( set( [ n for n in df["name"].values] ) )
        name = df["name"].mode()

        if len(name) == 0:
            print("unknown")
            return "unknown"
        else:
            name = name[0]
            print(name)
            # n = name.mode()
            #
            # print("n : ", n)
            # print("name :", name)
            # if len(name) == 0:
            #     return "unknown"
            # else:
            #    return ", ".join(name)
            # if len(name) ==
            # print(name)
            return str(name)

#####################################################################################################

def get_representation( images, model_name ='VGG-Face', distance_metric = 'cosine', model = None, enforce_detection = True, 
                        detector_backend = 'opencv', align = True, prog_bar = True, normalization = 'base' ):

    representations = []

    for name, img in images :
        instance = []
        instance.append(name)
        instance.append(img)

        custom_model = model

        representation = DeepFace.represent(img_path = img
            , model_name = model_name, model = custom_model
            , enforce_detection = enforce_detection, detector_backend = detector_backend
            , align = align
            , normalization = normalization
            )

        instance.append(representation)
        
        #-------------------------------

        representations.append(instance)

    return representations

#####################################################################################################

def set_session( model_path ):
    predictor = onnx.load( model_path )
    onnx.checker.check_model(predictor)
    onnx.helper.printable_graph(predictor.graph)
    predictor = backend.prepare(predictor, device="CPU")  # default CPU

    ort_session = ort.InferenceSession( model_path )
    input_name = ort_session.get_inputs()[0].name

    return ort_session, input_name

#####################################################################################################


