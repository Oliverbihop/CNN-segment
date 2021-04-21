import numpy as np
import cv2
import time
#from scipy.misc import imresize
#from IPython.display import HTML
from keras.models import load_model
from PIL import Image

# Class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []


def road_lines(image):
    # Get image ready for feeding into model
    image=image[130:290,:,:]
    small_img = cv2.resize(image, (160, 80))
    
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]

    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = model.predict(small_img)[0] * 255
    prediction=np.where(prediction < 125, 0, 255)
    # prediction = Image.fromarray(prediction, 'RGB')
    # Add lane prediction to list for averaging
    # lanes.recent_fit.append(prediction)
    # # Only using last five for average
    # if len(lanes.recent_fit) > 1:
    #     lanes.recent_fit = lanes.recent_fit[1:]
    prediction=prediction.reshape(80,160)
    print(prediction.shape)
    # # Calculate average detection
    # lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

    # # Generate fake R & B color dimensions, stack with G
    # blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    # lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # Re-size to match the original image
    # lane_image = cv2.resize(np.array(prediction), (640, 360))
    # image=cv2.resize(np.array(image), (160, 640))
    # Merge the lane drawing onto the original image
    # print(image.shape, lane_image.shape)
    # result = cv2.addWeighted(image, 1, lane_image, 1, 0)
    prediction = prediction.astype(np.uint8)
    
    return prediction


if __name__ == '__main__':
    # Load Keras model
    model = load_model('model-021.h5')
    # Create lanes object
    lanes = Lanes()

    # Where to save the output video
    # vid_output = 'proj_reg_vid.mp4'
    cap = cv2.VideoCapture("./output_01_ahead.avi")
    count=0
    while True:
        pre_time = time.time()
        _, img = cap.read()
        cv2.imshow("result1", img)
        img=road_lines(img)
        cv2.imshow("result", img)
        #if count %15==0:
        #    cv2.imwrite("./void/"+str(count)+".jpg",img)
        # if count %25==0:
        #     img=img[125:260,:,:]
        #     img = cv2.resize(img, (160, 80))
        #     cv2.imwrite('reio/'+str(count)+".jpg",img)
        # img=img[130:290,:,:]
        
        # road_lines(img)
        count=count+1
        if cv2.waitKey(0) == ord("q"):
            break
    
        print(1/(time.time() - pre_time))
    
    cap.release()
