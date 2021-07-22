import cv2

#trained data
face_trained_data = cv2.CascadeClassifier('face.xml')
smile_trained_data = cv2.CascadeClassifier('smile.xml')
eyes_trained_data = cv2.CascadeClassifier('eyes.xml')

video = cv2.VideoCapture(0)


while True:
    (success,frame) = video.read()

    if not success:
        break
    
    #coordintes of the face and drawing rectangle around it

    grayScaled_video = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face_coordinates = face_trained_data.detectMultiScale(grayScaled_video,)
    for (x,y,w,h) in face_coordinates: 
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,266,0),2)

        #cropping and getting only the face
        the_face = frame[y:y+h,x:x+w]
       
        #changing the font face to black and white
        grayScaled_face = cv2.cvtColor(the_face,cv2.COLOR_BGR2GRAY)
       
        #smile coordinates
        smile = smile_trained_data.detectMultiScale(grayScaled_face,scaleFactor=1.7,minNeighbors=20)
       
        #smile condition
        if len(smile) > 0:
            cv2.putText(frame,'Smiling',(w,y+h+50),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=1,color=(0,0,266))
       
        #eyes coordinates and drawing rectangle around it
        eyes = eyes_trained_data.detectMultiScale(grayScaled_face,scaleFactor=1.1,minNeighbors=30)
        for (x_,y_,w_,h_) in eyes:
            cv2.rectangle(the_face,(x_,y_),(x_+w_,y_+h_),(250,0,0),2)    




    cv2.imshow('smile-eyes detector',frame)
    key = cv2.waitKey(1)
    if key==81 or key==113:
        break

video.release()
cv2.destroyAllWindows()    