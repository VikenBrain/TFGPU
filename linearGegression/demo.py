import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

print('Tensorflow Version:.{}'.format(tf.__version__))

"""f(x) = ax + b"""

class linearG():

    def __init__(self):
        self.data = pd.read_csv('income.csv', index_col=0)

    def graph(self):
        """
        绘图
        :return:
        """
        plt.scatter(self.data.Education, self.data.Income)
        # plt.show()

    def model(self):
        x = self.data.Education
        y = self.data.Income
        mode = tf.keras.Sequential()
        mode.add(tf.keras.layers.Dense(1, input_shape=(1, ))) # 一维, 输入model为1个特征
        # print(model.summary()) # description
        """
        Model: "sequential"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        dense (Dense)                (None, 1)                 2         
        =================================================================
        Total params: 2
        Trainable params: 2
        Non-trainable params: 0
        _________________________________________________________________
        None
        """
        mode.compile(
            optimizer= 'adam', # 梯度下降
            loss= 'mse', # 均方误差
        )
        history = mode.fit(x, y, epochs= 5000) # 训练5000次, 每次训练结果不一
        # print(mode.predict(x)) # 预测x下的y
        print(mode.predict(pd.Series([20])))  # 预测20下的y


    def start(self):
        # t = self.graph()
        m = self.model()
        # pass


if __name__ == '__main__':
    linearG().start()




