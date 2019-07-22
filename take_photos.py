import cv2
import time
import os

cap = cv2.VideoCapture(0)

name = 'a'

for i in range(600):
    ret, frame = cap.read()
        
    cv2.imshow("capture", frame)
    cv2.waitKey(1)
    
    file = './faceImages/' + name + '/'
    if not os.path.exists(file):
      os.makedirs(file)
    
    print(cv2.imwrite(file + str(i) + '.jpg', frame))
    time.sleep(0.1)
    
cv2.destroyAllWindows()
cap.release()