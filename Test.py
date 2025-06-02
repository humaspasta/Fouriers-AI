# import cv2
# import numpy as np
# import math
# import time
# from CustomCircle import CustomCircle



# # Set up canvas
# width, height = 600, 600
# center = (width // 2, height // 2)
# radius = 150

# # Frame setup
# angle = 0
# fps = 60
# delay = 1 / fps
# circ = CustomCircle(None, 300 , 300, 100 , 0.02) # there is no frame initially. The frame is updated in the loop
# circ2 = CustomCircle(None , int(circ.calculate_rotate()[0]), int(circ.calculate_rotate()[1]), 30 , 0.05)
# circ3 = CustomCircle(None , int(circ2.calculate_rotate()[0]), int(circ2.calculate_rotate()[1]) , 10 , 0.07, isTip=True)
# frame = np.ones((height, width, 3), dtype=np.uint8) * 255
# frame_trace = np.ones((height, width, 3), dtype=np.uint8) * 255


# frame = np.ones((height, width, 3), dtype=np.uint8) * 255
# cv2.circle(frame , (300, 300), 100 , (200 , 200 ,200) , 2)

# cv2.imshow("Rotating Radius - Sin/Cos", frame)

# time.sleep(delay)

# cv2.destroyAllWindows()



    
    
    