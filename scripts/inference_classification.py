import cv2
import numpy as np
import tensorflow as tf

model = None
model_path = 'saved_model_sketch2ppt/model_fully_connected'


class Inference:
    def __init__(self):
        self.model = tf.keras.models.load_model(model_path)

    def get_prediction(self, image):
        return self.model.predict(image)[0]


def main():
    addr = 'images/Arrow_136_747_944_1251.jpg'
    image_width = 64
    image_height = 64
    img = cv2.imread(addr, 0)
    img = cv2.resize(img, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
    print(img.shape)
    img = np.expand_dims(img,axis=0)
    inf = Inference()
    pred = inf.get_prediction(img)
    print(pred)


if __name__ == "__main__":
    main()
