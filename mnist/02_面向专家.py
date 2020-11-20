import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# 使用tf.data来讲数据集切分为batch以及混淆矩阵数据集
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 使用 Keras 模型子类化（model subclassing） API 构建 tf.keras 模型
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation= 'relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation= 'relu')
        self.d2 = Dense(10, activation= 'softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

model = MyModel()

# 为训练选择优化器与损失函数
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# 选择衡量指标来度量模型的损失值(loss)和准确率(accuracy).这些指标在 epoch 上累积值，然后打印出整体结果。
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# 使用 tf.GradientTape 来训练模型
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

# 测试模型
@tf.function
def test_step(images, labels):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

EPOCHS = 5

for epoch in range(EPOCHS):
  # 在下一个epoch开始时，重置评估指标
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100))

"""
Epoch 1, Loss: 0.14114607870578766, Accuracy: 95.7316665649414, Test Loss: 0.05401783064007759, Test Accuracy: 98.25
Epoch 2, Loss: 0.04215998202562332, Accuracy: 98.69999694824219, Test Loss: 0.052269529551267624, Test Accuracy: 98.22999572753906
Epoch 3, Loss: 0.022514233365654945, Accuracy: 99.2733383178711, Test Loss: 0.04868340864777565, Test Accuracy: 98.43000030517578
Epoch 4, Loss: 0.013666493818163872, Accuracy: 99.5616683959961, Test Loss: 0.058275070041418076, Test Accuracy: 98.2699966430664
Epoch 5, Loss: 0.009711191058158875, Accuracy: 99.66999816894531, Test Loss: 0.06348905712366104, Test Accuracy: 98.31999969482422
"""