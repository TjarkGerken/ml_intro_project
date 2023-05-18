
# Basic Packages
import tensorflow
import random
import numpy as np
import os
import pandas as pd

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt

# ML
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer


# Display the predictions and the ground truth visually.
def display_prediction (images, true_labels, predicted_labels, export :bool = True):
    fig = plt.figure(figsize=(20, amount_of_samples +10))
    for i in range(len(true_labels)):
        truth = true_labels[i]
        prediction = predicted_labels[i]
        plt.subplot(amount_of_samples, 4, 1 + i)
        plt.axis('off')
        color = 'green' if truth == prediction else 'red'
        plt.text(30, 10, f"Truth:           {classes[truth]}\nPrediction:   {classes[prediction]}",
                 fontsize=12, color=color)
        plt.imshow(images[i])
    if export:
        plt.savefig(plot_path + "predictions" + output_format)


if __name__ == "__main__":
    # Verändere diese Variable, um das Modell neu zu trainieren:
    retrain = False

    sns.axes_style("whitegrid")

    cache_path = "./models/"
    plot_path = "./output/"
    model_name = "american_dataset.h5"
    output_format = ".svg"

    training_history = "training_history.csv"

    if not os.path.isdir(cache_path):
        os.mkdir(cache_path)

    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)

    if not os.path.exists(cache_path + model_name) or not os.path.exists(cache_path + training_history):
        retrain = True

    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["figure.figsize"] = [20, 15]

    train_df = pd.read_csv("data/sign_mnist_train/sign_mnist_train.csv")
    test_df = pd.read_csv("data/sign_mnist_test/sign_mnist_test.csv")

    print(retrain)

    classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
               "W", "X", "Y"]

    train_num_samples = len(train_df)
    test_num_samples = len(test_df)
    print(f"{'=' * 10} Exploration of the Data Set {'=' * 10}")
    print(f'Anzahl an Trainingsdaten: \t\t\t\t\t {train_num_samples} Einträge')
    print(f'Anzahl an Testdaten: \t\t\t\t\t\t {test_num_samples} Einträge')
    print(
        f"Durchschnittliche Einträge pro Buchstabe \t {round(train_df.groupby(by=['label'])['pixel1'].count().mean(), 2)}")
    print(f"{'=' * 10} Exploration of the Data Set {'=' * 10}")

    plt.figure(figsize=(10, 10))
    cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.70, reverse=True)

    sns.countplot(x=train_df['label'], palette=cmap)
    plt.title("Verteilung der Label")
    plt.savefig(plot_path + "label_distribution" + output_format)

    y_train = train_df['label']
    y_test = test_df['label']
    del train_df['label']
    del test_df['label']

    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.fit_transform(y_test)

    x_train = train_df.values
    x_test = test_df.values

    x_train = x_train / 255
    x_test = x_test / 255

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    print("=" * 10 + " Ausgabe der Dimensionen der Datensätze " + "=" * 10)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    print("=" * 10 + " Ausgabe der Dimensionen der Datensätze " + "=" * 10)

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(x_train)

    model = Sequential()
    model.add(Conv2D(75, (3, 3), strides=1, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))
    model.add(Conv2D(50, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))
    model.add(Conv2D(25, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))
    model.add(Flatten())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=24, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    if retrain:
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5,
                                                    min_lr=0.00001)
        history = model.fit(datagen.flow(x_train, y_train, batch_size=128), epochs=20, validation_data=(x_test, y_test),
                            callbacks=[learning_rate_reduction])
        model.save(cache_path + model_name)
        history = pd.DataFrame(history.history)
        history.to_csv(cache_path + training_history)
    else:
        model = load_model(cache_path + model_name)
        history = pd.read_csv(cache_path + training_history)

    # define a function to plot the learning curves
    if retrain:
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5,
                                                    min_lr=0.00001)
        history = model.fit(datagen.flow(x_train, y_train, batch_size=128), epochs=20, validation_data=(x_test, y_test),
                            callbacks=[learning_rate_reduction])
        model.save(cache_path + model_name)
        history = pd.DataFrame(history.history)
        history.to_csv(cache_path + training_history)
    else:
        model = load_model(cache_path + model_name)
        history = pd.read_csv(cache_path + training_history)

    # define a function to plot the learning curves

    plt.figure(figsize=(12, 16))

    plt.subplot(4, 2, 1)
    plt.plot(history['loss'], label='train loss')
    plt.plot(history['val_loss'], label='val loss')
    plt.legend()
    plt.title("Zeitliche Entwicklung des Loss")

    plt.subplot(4, 2, 2)

    plt.plot(history['accuracy'], label='train acc')
    plt.plot(history['val_accuracy'], label='val acc')

    plt.legend()
    plt.title("Zeitliche Entwicklung der Accuracy")

    plt.subplot(4, 2, 3)
    plt.plot(history['loss'], label='train loss')
    plt.plot(history['val_loss'], label='val loss')
    plt.legend()
    plt.ylim([0, 0.1])
    plt.title("Zeitliche Entwicklung des Loss (Vergrößert)")

    plt.subplot(4, 2, 4)

    plt.plot(history['accuracy'], label='train acc')
    plt.plot(history['val_accuracy'], label='val acc')
    plt.ylim([0.95, 1.01])
    plt.legend()
    plt.title("Zeitliche Entwicklung der Accuracy (Vergrößert)")

    plt.savefig(plot_path + "learning_curves" + output_format)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{'=' * 10} Evaluation of the Test Data {'=' * 10}")
    print(f"n Training Samples: \t{train_num_samples}\nn Test Samples:\t\t\t{test_num_samples}")
    print(f"Loss: \t\t\t\t\t{loss:.5f}\nAccuracy:\t\t\t\t{acc:.5f}")
    print(f"{'=' * 10} Evaluation of the Test Data {'=' * 10}")

    predictions = model.predict(x_test)
    prediction_classes = np.argmax(predictions, axis=-1)

    gt_classes = np.argmax(y_test, axis=-1)
    confusion_matrix = metrics.confusion_matrix(gt_classes, prediction_classes)
    cmap = sns.cubehelix_palette(start=2, rot=0, dark=1, light=0, reverse=True, as_cmap=True)
    sns.heatmap(pd.DataFrame(confusion_matrix, index=classes, columns=classes), annot=True, cmap=cmap, fmt='d')

    plt.title('Confusion Matrix for predicting the American Sign Language', fontsize=28, pad=20)
    plt.ylabel('predicted', fontsize=20)
    plt.xlabel('ground truth', fontsize=20)
    plt.savefig(plot_path + "confusion_matrix" + output_format)
    plt.show()

    amount_of_samples = 24

    random.seed(3)  # to make this deterministic
    sample_indexes = random.sample(range(len(x_test)), amount_of_samples)
    sample_images = [x_test[i] for i in sample_indexes]
    sample_labels = [y_test[i] for i in sample_indexes]

    ground_truth = np.argmax(sample_labels, axis=1)

    X_sample = np.array(sample_images)
    prediction = model.predict(X_sample)
    predicted_categories = np.argmax(prediction, axis=1)

    display_prediction(sample_images, ground_truth, predicted_categories)
