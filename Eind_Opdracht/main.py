from objdetection1 import *
from clustering import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub



from sklearn.model_selection import train_test_split
from tensorflow import keras
# from tensorflow.keras import layers as L
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from keras import layers as L
from keras.models import Sequential
from keras.applications.mobilenet_v2 import MobileNetV2
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.preprocessing.image import load_img
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

dirs = ["train", "test", "valid"]

IMAGE_SHAPE = (224, 224)

pill_names = {
    1:'Xyzall 5mg',
    2: 'Cipro 500',
    3: 'Ibuphil Cold 400-60',
    4:'red',
    5:'pink',
    6:'white',
    7:'blue',
    8:'Ibuphil 600 mg'
    
}


  
pic = cv.imread("test5_nopill.jpg")

pill_candidates, bboxes = createCandidatePositions(pic)
print(bboxes)
print(pill_names[1])

# cv.imshow("pleasefortheloveofgodwork",pill_candidates[3])


# cv.waitKey(0)
# cv.destroyAllWindows()





X, Y = import_dataset(dirs)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

print(len(X_train))
print(len(X_test))

print(len(y_train))
print(len(y_test))


X_train_scaled = X_train / 255
X_test_scaled = X_test / 255







classifier = tf.keras.Sequential([
hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", input_shape=IMAGE_SHAPE+(3,))])



x0_resized = cv.resize(X[0], IMAGE_SHAPE)
x1_resized = cv.resize(X[1], IMAGE_SHAPE)
x2_resized = cv.resize(X[2], IMAGE_SHAPE)

predicted = classifier.predict(np.array([x0_resized, x1_resized, x2_resized]))
predicted = np.argmax(predicted, axis=1)
print(predicted)

feature_extractor_model ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
pretrained_model_without_top_layer = hub.KerasLayer(feature_extractor_model, input_shape=(224, 224, 3), trainable=False)


tmpClassifier = tf.keras.Sequential([
pretrained_model_without_top_layer
])
tmp_hub_layer = tmpClassifier.layers[0]



tmp_hub_layer = tmpClassifier.layers[0]
num_of_pills = 9

model = tf.keras.Sequential([
pretrained_model_without_top_layer,
tf.keras.layers.Dense(num_of_pills)
])
model.summary()

model = tf.keras.Sequential([
pretrained_model_without_top_layer,

tf.keras.layers.BatchNormalization(),
tf.keras.layers.Dense(3000,activation='relu'),
tf.keras.layers.Dense(num_of_pills)
])

model.summary()
model.compile(
optimizer="adam",
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics=['acc'],)


model.fit(X_train_scaled, y_train, epochs=10, validation_data=(X_test_scaled, y_test)) 




model.evaluate(X_test_scaled,y_test)

convertedCandidates = []
for i in range(len(pill_candidates)-1):
    convertedCandidates.append(change_bboximg(pill_candidates[i], (224,224)))

pill_candidates = np.array(convertedCandidates)/255.0

print(pill_candidates.shape)

prediction = model.predict(pill_candidates)
print(prediction)


for i in range(len(prediction)):
    highest = prediction[i].argmax()
    if prediction[i][0] < 1:
        cv.rectangle( pic, (bboxes[i][0], bboxes[i][1]), (bboxes[i][2], bboxes[i][3]), (255,0,0), 3 )
        cv.putText(
        pic,
        pill_names[highest],
        (int(bboxes[i][0]), int(bboxes[i][1]) - 10),
        fontFace = cv.FONT_HERSHEY_SIMPLEX,
        fontScale = 0.6,
        color = (255, 0, 0),
        thickness=2
        )
        print(prediction[i][0])


model.summary()
# window_name = 'image'

cv.imshow("pleasefortheloveofgodwork",pic) 
cv.waitKey(0)
cv.destroyAllWindows()






# cv.imshow(window_name,X[0])

# cv.waitKey(0)
  
# closing all open windows
# cv.destroyAllWindows()