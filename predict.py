import json
import Utils
import requests
import cv2
import numpy as np
import scipy.misc
from PIL import Image
import matplotlib.pyplot as plt



image_height = 384
image_width = 384
image_h_lr = image_height /4
image_w_lr = image_width /4
amount = 2
batch_size = 2
image_folder = './train2019'
model_path = './model'
image_shape = (image_height,image_width, 3)
number_of_images = 10

x_train_lr, x_train_hr, x_test_lr, x_test_hr\
        = Utils.load_training_data(image_folder, '.jpg', number_of_images, 0.6)

x_test_lrs = Utils.denormalize(x_test_lr)

for i in range(len(x_test_lrs)):
	cv2.imwrite("./images/input_image_%i.jpg"%i,cv2.cvtColor(x_test_lrs[i], cv2.COLOR_RGB2BGR))
	data = json.dumps({"signature_name": "predict", "instances": [x_test_lr[i].tolist()]})
	headers = {"content-type": "application/json"}
	json_response = requests.post('http://192.168.1.128:8502/v1/models/keras_sr:predict', data=data, headers=headers)
	predictions = json.loads(json_response.text)['predictions']
	img=predictions

	img = np.asarray(img, dtype=np.float32)

	print("===========%d" % i)
	img = Utils.denormalize(img)

	cv2.imwrite("./images/predict_image_%i.jpg"%i,cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR))