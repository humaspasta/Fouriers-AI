import cv2
import numpy as np


class DataProcessing:
    def  __init__(self, frame_to_sample , time_to_sample=10):
        self.time_to_sample = time_to_sample
        self.frame = frame_to_sample

    def sample_frame(self):
        '''
        Should return a sampled dataframe with time(t), x, and y values. 
        Everything here is the processing code from before. Use np.arange to determine if a given value of time is within the sampling list
        '''
        points = []
        for y in range(len(self.frame)):
            for x in range(len(self.frame[y])):
                if(self.frame[y, x][0] == 0):
                    points.append((y,x))

            points.sort(key=lambda x: (x[0] , x[1]) ,reverse=True)

            top_half_points = [point for point in points if point[1] > 300]

            top_half_x_points = [point[0] for point in top_half_points]
            top_half_y_points = [point[1] for point in top_half_points]

            bottom_half_points = [point for point in points if point[1] < 300]

            bottom_half_x_points = [point[0] for point in bottom_half_points]
            bottom_half_y_points = [point[1] for point in bottom_half_points]


            x_in_order = top_half_x_points + bottom_half_x_points
            y_in_order = bottom_half_y_points + bottom_half_y_points

            coords_in_order = []
            for i in range(len(x_in_order)):
                coords_in_order.append((x_in_order[i] , y_in_order[i]))


            time_in_order = np.arange(start=0 , stop=2 * math.pi , step=(2 * math.pi) / len(coords_in_order))

            final_df = pd.DataFrame({
                'theta' : theta_in_order, 
                'coords': coords_in_order, 
                'X coords' : x_in_order,
                'Y coords' : y_in_order
            })

            #drop duplicate points
            labels = final_df.groupby("X coords").first()

            #rounding theta down 
            labels['theta'].apply(lambda x : round(x , 3))

            
