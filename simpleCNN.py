from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import random
from keras.preprocessing.image import image
import numpy as np
import os
import argparse


class simpleCNN:
    def __init__(self, trainDataPath, testDataPath, img_row, img_column, categories, af, regularization=0, drop=0):
        self.trainDataPath = trainDataPath
        self.testDataPath = testDataPath
        self.img_row = img_row
        self.img_column = img_column
        self.categories = categories
        self.af = af
        self.regularization = regularization
        self.drop = drop

        trainingData, trainingDataLabel = self.readData(self.trainDataPath)
        testData, testDataLabel = self.readData(self.testDataPath)
        # reshape input data to keras input format
        trainingData = np.array(trainingData).reshape(-1,
                                                      self.img_row, self.img_column, 3)
        testData = np.array(testData).reshape(-1,
                                              self.img_row, self.img_column, 3)
        # one-hot-encoding label
        trainingDataLabel = np_utils.to_categorical(trainingDataLabel)
        trainingDataLabel = np.array(
            trainingDataLabel).reshape(-1, self.categories)

        # data dataAugmentation
        training_DAset, test_DAset = self.dataAugmentation(
            self.trainDataPath, self.testDataPath)
        # train model without tunning
        model1 = self.setModel(self.img_row, self.img_column,
                               self.categories, self.af, 0, 0)

        model2 = self.setModel(self.img_row, self.img_column,
                               self.categories, self.af, self.regularization, self.drop)

        model3 = self.setModel(self.img_row, self.img_column,
                               self.categories, self.af,  0, 0)

        model4 = self.setModel(self.img_row, self.img_column,
                               self.categories, self.af, self.regularization, self.drop)
        # train model
        model1.fit(trainingData, trainingDataLabel, epochs=10)
        model2.fit(trainingData, trainingDataLabel, epochs=10)
        model3.fit_generator(
            training_DAset,
            steps_per_epoch=30,
            epochs=10,
            validation_data=test_DAset,
            validation_steps=30)
        model4.fit_generator(
            training_DAset,
            steps_per_epoch=30,
            epochs=10,
            validation_data=test_DAset,
            validation_steps=30)
        # predict & score
        predict1 = model1.predict(testData)
        predict2 = model2.predict(testData)
        predict3 = model3.predict(testData)
        predict4 = model4.predict(testData)
        score1 = self.showAccuracy(predict1, testDataLabel)
        score2 = self.showAccuracy(predict2, testDataLabel)
        score3 = self.showAccuracy(predict3, testDataLabel)
        score4 = self.showAccuracy(predict4, testDataLabel)

        print("accuracy(without tunning): ", score1)
        print("accuracy(with weight-decay and dropout): ", score2)
        print("accuracy(with data augmentation ): ", score3)
        print("accuracy(with data augmentation and weight-decay and dropout ): ", score4)

    def __del__(self):
        print("object deleted")

    def readData(self, path):
        dirs = os.listdir(path)
        dataImgPath = []
        metaData = []
        metaDataLabel = []
        temp = []
        for folder in dirs:
            if not folder.startswith("."):
                subFolder = path+folder
                subFolders = os.listdir(subFolder)
                if folder not in temp:
                    temp.append(folder)
                for img in subFolders:
                    if not img.startswith("."):
                        imgPath = subFolder+'/'+img
                        dataImgPath.append(imgPath)

        random.shuffle(dataImgPath)
        for path in dataImgPath:
            img = image.load_img(path)
            img = image.img_to_array(img)
            metaData.append(img)
            label = path.split("/")[2]
            for index in range(0, len(temp)):
                if label == temp[index]:
                    metaDataLabel.append(index)

        return metaData, metaDataLabel

    def setModel(self, row, column, categories, activation_function, regularization, drop):
        # initial the CNN
        model = Sequential()
        # convolution  Convolution2D(32,3,3)=>32 3*3feature detector
        model.add(Convolution2D(
            32, 3, 3, input_shape=(row, column, 3), activation='relu', kernel_regularizer=l2(regularization), bias_regularizer=l2(regularization)))
        # pooling
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # convolution
        model.add(Convolution2D(
            64, 3, 3, activation='relu', kernel_regularizer=l2(regularization), bias_regularizer=l2(regularization)))
        # pooling
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # flattening
        model.add(Flatten())

        model.add(Dropout(drop))
        # fully connection
        model.add(Dense(output_dim=128, activation='relu', kernel_regularizer=l2(
            regularization), bias_regularizer=l2(regularization)))
        # output layer using softmax activation fuction beacuse we have 3 categories
        model.add(Dense(output_dim=categories, activation=activation_function))
        # compile CNN  loss=>categorical_crossentropy  beacuse we have 3 categories
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()

        return model

    def dataAugmentation(self, trainDataPath, testDataPath):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1./255)

        training_set = train_datagen.flow_from_directory(
            trainDataPath,
            target_size=(128, 128),
            batch_size=32,
            class_mode='categorical')
        test_set = test_datagen.flow_from_directory(
            testDataPath,
            target_size=(128, 128),
            batch_size=32,
            class_mode='categorical')

        return training_set, test_set

    def showAccuracy(self, predict, answer):
        predict = np.argmax(predict, axis=1)
        score = 0
        for index in range(0, len(predict)):
            if predict[index] == answer[index]:
                score += 1
        accuracy = score / len(predict)
        return accuracy


def process_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', metavar=('TRAIN_DATA_PATH'),
                        type=str, required=True, help='enter train data path')
    parser.add_argument('-test', metavar=('TEST_DATA_PATH'),
                        type=str, required=True, help='enter test data path')
    parser.add_argument('-size', metavar=('ROW_SIZE', 'COLUMN_SIZE'), type=int,
                        required=True, nargs=2, help="enter the image row size and column size")
    parser.add_argument('-c', metavar=('CATEGORIES_NUMBER'), type=int,
                        required=True, help="enter categories number")
    parser.add_argument('-af', metavar=('ACTIVATION_FUNCTION'),
                        type=str, required=True, help='enter activation function type')
    parser.add_argument('-r', metavar=('WEIGHT_DECAY_RATE'),
                        type=float,  help='enter weight decay rate,0~0.1')
    parser.add_argument('-d', metavar=('DROPOUT_RATE'),
                        type=float,  help='enter dropout rate, 0~0.5')

    args = parser.parse_args()
    simpleCNN(trainDataPath=args.train, testDataPath=args.test,
              img_row=args.size[0], img_column=args.size[1], categories=args.c, af=args.af, regularization=args.r, drop=args.d)

    return parser.parse_args()


if __name__ == '__main__':
    process_command()
    # command example
    # python simpleCNN.py -train dataset/training_set/ -test dataset/test_set/ -size 128 128 -c 3 -af softmax -r 0.01 -d 0.2
