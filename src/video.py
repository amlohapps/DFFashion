import numpy as np
import cv2
from rem_bg import remove_background as rem
  

cap = cv2.VideoCapture('/home/prajwal/Downloads/79147536_192199808488125_4519891993979281994_n.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('out.avi',fourcc, 23.99, (640,480))

while(cap.isOpened()):
    
    try:
        ret, frame = cap.read()
        frame = rem(frame)
        out.write(frame)
        out.write(frame)
        cv2.imshow('frame',frame)
    except Exception as e:
        
        print("\n",e)
        break
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
