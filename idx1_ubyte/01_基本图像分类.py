from tensorflow.python.keras.utils import get_file
import tensorflow as tf
from tensorflow import keras

import gzip
import numpy as np
import matplotlib.pyplot as plt


print(tf.__version__)

"""读取数据"""
def load_data():
    base = "file:///E:\\code\\tensorflow_gpu\\tfds\\fs\\"
    files = [
        'train-labels-idx1-ubyte.gz', # 训练集
        'train-images-idx3-ubyte.gz', # 训练集
        't10k-labels-idx1-ubyte.gz', # 测试集
        't10k-images-idx3-ubyte.gz' # 测试集
    ]

    paths = []
    for fname in files:

        paths.append(tf.keras.utils.get_file(fname, origin=base + fname))

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)


(train_images, train_labels), (test_images, test_labels) = load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)
print(len(train_images))
print(test_images.shape)
print(len(test_labels))

plt.figure(figsize=(5,5))
plt.imshow(train_images[100])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

"""建立模型"""
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])
"""编译模型"""
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
"""训练模型"""
model.fit(train_images, train_labels, epochs=10)
"""评估准确性"""
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

"""做出预测"""
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[0])


