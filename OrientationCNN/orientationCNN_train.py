import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

BATCH_SIZE = 16

def create_model():
    input_tensor = tf.keras.layers.Input(shape=(224, 224, 3))
    vgg_model = tf.keras.applications.vgg16.VGG16(include_top=False, input_tensor=input_tensor)

    #for i in range(1, 5):
    #    print(f"Locking layer {i}")
    #    vgg_model.layers[i].trainable = False

    for l in vgg_model.layers:
        print(l.name, l.trainable)

    X = tf.keras.layers.Flatten()(vgg_model.output)
    X = tf.keras.layers.Dense(1024, activation="relu")(X)
    X = tf.keras.layers.Dense(1024, activation="relu")(X)
    X = tf.keras.layers.Dense(1, activation="sigmoid")(X)

    model = tf.keras.models.Model(inputs=input_tensor, outputs=X)

    return model

def train_model():
    print("creating model")
    model = create_model()
    model.load_weights("saved_models/model.04-0.0009-1.0000.hdf5")

    print("creating dataset")
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow_from_directory("dataset/train", target_size=(224, 224), batch_size=BATCH_SIZE, class_mode="binary")
    test_generator = datagen.flow_from_directory("dataset/test", target_size=(224, 224), batch_size=BATCH_SIZE, class_mode="binary")

    print("starting training")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00006)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('saved_models/model.{epoch:02d}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5',
                                          monitor='val_loss', verbose=1, period=1, save_best_only=True)

    history = model.fit_generator(train_generator, steps_per_epoch=train_generator.samples/BATCH_SIZE, epochs=50, validation_data=test_generator,
                        validation_steps=test_generator.samples/BATCH_SIZE, callbacks=[checkpoint_callback])

    model.save('final_model.hdf5', save_format='h5')

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('accuracy_history.png')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('loss_history.png')
    plt.show()


def test_model():
    classifier = tf.keras.models.load_model("saved_models/model.50-0.0000-1.0000.hdf5")
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_set = datagen.flow_from_directory("dataset/validation", target_size=(224, 224), batch_size=BATCH_SIZE, class_mode="binary")

    y_true = None
    y_predicted = None
    is_first_iter = True
    iteration = 1
    images_predicted = 0
    incorrect_predictions = 0
    steps = test_set.samples / BATCH_SIZE
    for X_batch, y_batch in test_set:
        current_batch_size = X_batch.shape[0]
        if (iteration - 1) >= steps:
            break
        print('Iteration ', iteration)
        y_batch_predicted = np.zeros((current_batch_size, 1))
        for i in range(current_batch_size):
            images_predicted += 1
            image_to_predict = X_batch[i].reshape(1, X_batch[i].shape[0], X_batch[i].shape[1], X_batch[i].shape[2])
            prediction = classifier.predict(image_to_predict)
            thresholded_prediction = 1 if prediction[0, 0] > 0.6 else 0
            print("pred: {0}, y_batch: {1}".format(thresholded_prediction, y_batch[i]))
            y_batch_predicted[i, 0] = thresholded_prediction
            prediction_class_name = None
            true_class_name = None
           # for key, value in test_set.class_indices.items():
           #     if value == prediction:
           #         prediction_class_name = key
           #     elif value == y_batch[i].argmax():
            #        true_class_name = key

            if thresholded_prediction != y_batch[i]:
                plt.imshow(X_batch[i])
                plt.title("P: {0}, T: {1}".format(prediction[0, 0], y_batch[i]))
                plt.show()
                incorrect_predictions += 1
        if is_first_iter:
            y_true = y_batch
            y_predicted = y_batch_predicted
            is_first_iter = False
        else:
            y_true = np.concatenate((y_true, y_batch))
            y_predicted = np.concatenate((y_predicted, y_batch_predicted))

        iteration += 1
        print('Images predicted: ', images_predicted)
    y_predicted = y_predicted.reshape((-1,)).astype(np.int32)
    accuracy = 1. - (incorrect_predictions / images_predicted)
    print('Class labels: ', test_set.class_indices)
    print('Test accuracy: {0} ({1}/{2})'.format(accuracy, incorrect_predictions, images_predicted))

    class_array = []
    for key, value in test_set.class_indices.items():
        name_with_spaces = ''
        name_index = 0
        for c in key:
            if c.isupper() and name_index != 0:
                name_with_spaces += ' ' + c
            else:
                name_with_spaces += c
            name_index += 1
        class_array.append(name_with_spaces)

    confusion_m = confusion_matrix(y_true, y_predicted)
    plot_confusion_matrix(confusion_m, class_array)


def plot_confusion_matrix(cm, classes):
    title = 'Object Detection confusion matrix'
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    normalize = True
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=(12.8, 9.2))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.GnBu)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig("confusion_matrix.pdf")
    return ax


if __name__ == "__main__":
    #train_model()
    test_model()