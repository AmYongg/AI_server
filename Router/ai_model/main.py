from AI_model import *

onnx_path = "models/onnx/version-RFB-320.onnx"

session, input_name = set_session( onnx_path )

representations = get_representation( get_db_images(0, "./itzy"), "Facenet", enforce_detection = False )

#cap = cv2.VideoCapture("./ITZY_ICY.mp4")  # capture from camera
cap = cv2.VideoCapture(0)

threshold = 0.7

while True:
    ret, orig_image = cap.read()

    if ret and True:
        image = get_preprocess_image( orig_image , (320,240) )

        confidences, boxes = session.run(None, {input_name: image})

        boxes, labels, probs = get_face_area(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)

        ns = []

        for box in boxes:
            try :
                face = orig_image[ box[1]-30: box[3]+30 , box[0]-20: box[2]+20 ]

                cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

                name = face_recognition(face, representations)
                    
                orig_image = cv2.putText(orig_image, name, (box[0], box[1]) , cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

                ns.apped( name )
            except Exception as e:
                print(e)

    cv2.imshow('annotated', orig_image)
    if cv2.waitKey(int(1000/30)) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
