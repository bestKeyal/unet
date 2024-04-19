from keras import Model
from tensorflow import keras
import tensorflow as tf
from keras.layers import *
from keras import backend as K


def jaccard(y_true, y_pred):
    tp = tf.reduce_sum(tf.multiply(y_true, y_pred), 1)
    fn = tf.reduce_sum(tf.multiply(y_true, 1 - y_pred), 1)
    fp = tf.reduce_sum(tf.multiply(1 - y_true, y_pred), 1)
    return 1 - (tp / (tp + fn + fp))

def voe(y_true, y_pred):
    return 1 - jaccard(y_true, y_pred)


def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    spec = true_negatives / (possible_negatives + K.epsilon())
    return spec


# 敏感性（召回率）
recall = tf.keras.metrics.Recall(name='recall')

# 精确度
precision = tf.keras.metrics.Precision(name='precision')


def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return 1 - score


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, label_smoothing=1e-2)
    dice = dice_loss(y_true, y_pred)
    return bce + dice


def iou_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


def r_squared(y_true, y_pred):
    """
    计算决定系数R^2，用作Keras模型的评估指标。

    :param y_true: 真实值
    :param y_pred: 预测值
    :return: R^2的值
    """
    # 总平方和（Total Sum of Squares, SSE）
    ss_total = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    # 回归平方和（Residual Sum of Squares, SSR）
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    # R^2 计算
    r2 = 1 - ss_res / ss_total
    return r2


def unet(pretrained_weights=None, input_size=(128, 128, 1), learningRate=1e-5, decayRate=1e-7):
    ModInputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(ModInputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)  # sigmoid

    model = Model(inputs=ModInputs, outputs=conv10)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                  loss=bce_dice_loss,
                  metrics=[
                      'accuracy',
                      bce_dice_loss,
                      dice_loss,
                      iou_coefficient,
                      r_squared,
                      jaccard,
                      recall,
                      precision,
                      specificity,
                      voe,
                  ]

                  )

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


if __name__ == '__main__':
    import numpy as np


    def generate_fake_data(num_samples, input_size):
        # 生成随机数据作为样本
        x = np.random.random((num_samples,) + input_size)
        # 生成随机二进制标签作为样本标签
        y = np.random.randint(0, 2, (num_samples,) + input_size)
        return x, y


    num_samples = 10
    input_size = (128, 128, 1)
    x_train, y_train = generate_fake_data(num_samples, input_size)

    model = unet(input_size=input_size)
    model.fit(x_train, y_train, epochs=5, batch_size=1)
