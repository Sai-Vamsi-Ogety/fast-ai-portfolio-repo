# Detect an Isolation Zoom action

## Introduction 
This isolated camera zoom follows one or more players and/or coaches in the frame and ends as soon as there is a camera change.The following figures are examples when there was an Isolated zoom action.

![](/images/action_1.jpg "Figure_1")


![](/images/action_2.jpg "Figure_2")

The motivation is to improve sports Analytics through Computer Vision and improve sports coverage which helps in better customer engagement.

## How we built it?
We started with a simple 3-layer CNN model as a base model and later on used **Inception V3 model** to train on our dataset using AWS SageMaker.

## Challenges
1. Loading the initial dataset into the instance
2. creating a balanced dataset for all the classes.

## Accomplishments
Built a full-scale CNN model and deployed it on the cloud and achieved test accuracy close to 90%.

## What I learnt ?
Usage of AWS SageMaker for training and deploying our Machine Learning models and to do transfer learning using a pre-trained model.

![](/images/action_3.jpg "Figure_3")


## Future Scope
Collect and augment more data and improve the test accuracy and train it on all actions to create a generalized model that can detect any type of action.

## code 
The model is trained on InceptionV3 Model for the given dataset.The model weights and the model itself is saved into the disk. Upon running the following code snippet the model and it's weights are loaded and prediction are made on the test set.

The link to the Github repo is available [here](https://github.com/Sai-Vamsi-Ogety/Detect-an-isolated-camera-zoom/).
```
import keras
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from keras import optimizers
from sklearn.metrics import precision_recall_fscore_support

def inference(csv_file):
    json_file = open('model_isolation.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model_wieghts_isolation.h5")
    df = pd.read_csv(csv_file, header=None, names=["id", "label"], dtype=str)
    df = df.replace({'true':'isolation','false':'noaction', 'True':'isolation', 'False':'noaction'})
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    valid_generator=test_datagen.flow_from_dataframe(
        dataframe=df,
        directory="./",
        x_col="id",
        y_col="label",
        class_mode="categorical",
        batch_size=1,
        target_size=(320,180))
    loaded_model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])
    result1 = loaded_model.predict_generator(valid_generator, steps=valid_generator.samples)
    result = []
    result1 = result1.argmax(axis=1)
    for i in result1:
        if i == 0:
            result.append('isolation')
        else:
            result.append('noaction')
    accuracy = (df['label'].values ==result).mean()
    ans = precision_recall_fscore_support(df['label'].values, result, average='macro')
    return {'accuracy':accuracy, 'recall':ans[1], 'precision':ans[0]}
```







