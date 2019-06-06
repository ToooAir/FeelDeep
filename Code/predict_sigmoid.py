import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"]='-1' 

from PIL import ImageFile
# Settings
ImageFile.LOAD_TRUNCATED_IMAGES = True

classifier = load_model('./sigmoid_1.h5')

one = glob.glob('../PornImage/other/TEST/*')
one = one[:100]
total = len(one)
count1 = 0
for o in one:
    print(o)
    test_image = image.load_img(o, target_size = (512, 512))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)

    if result[0][0] == 1:
        prediction = '豪哥喜歡'
        count1 += 1
    else:
        prediction = '豪哥不喜歡'
    print(prediction)
print('豪哥喜歡率: ' + str(count1/total*100) + '%')
print('豪哥不喜歡率: ' + str(100-count1/total*100) + '%')