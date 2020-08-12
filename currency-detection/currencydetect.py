import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import speech_recognition as sr
import pyttsx3

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

currency = ["two thousand rupee note", 'five hundred rupee note','two hundred ruppee note', 'one hundred rupee note', 'one hundred rupee note','fifty rupee note','fifty rupee note']


# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

total = 0

while 1:
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    imgfile = input("Enter image name")
    
    image = Image.open(imgfile)

#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

#turn the image into a numpy array
    image_array = np.asarray(image)

# display the resized image
    image.show()

# Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

# Load the image into the array
    data[0] = normalized_image_array

# run the inference
    prediction = model.predict(data)

    prediction = np.array(prediction)
    index=0

    for i in range(0,7):
        if prediction[0,i]>0.6:
            index=i

    result = (currency[index])
 
    print(result)


    def speak(string):
        eng = pyttsx3.init()
        eng.say(string)
        eng.runAndWait()
    
    speak(result)
    
    speak("if u want to quit press q")
    inp = input()
    if inp=='q':
        break


