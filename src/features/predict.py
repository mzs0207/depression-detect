#!/usr/bin/env python
# coding:utf8
"""
通过语音文件输入到模型得到出结果
    1 语音转图片
    2 预处理
        (1) 切片
        (2) 正则化
    3 输入模型
        预测的结果是一个数组，是因为把图片切片分成很多部分。

"""
from spectrograms import stft_matrix
from keras.models import load_model
from random_sampling import get_random_samples
import numpy as np

np.random.seed(15)  # for reproducibility


def preprocess(mat):
    """

    :param mat:
    :return:
    """
    crop_width = 125
    samples = mat.shape[1] / crop_width
    mats = get_random_samples(mat, samples, crop_width)
    mats = np.array(mats)
    mats = mats.astype("float32")
    mats = np.array([(x - x.min())/(x.max() - x.min()) for x in mats])
    mats = mats.reshape(mats.shape[0], mats.shape[1], crop_width, 1)

    return mats


def predict_from_wav(wav_file):
    """

    :param wav_file:
    :return:
    """
    mat = stft_matrix(wav_file)
    mats = preprocess(mat)
    model = load_model("../models/cnn_first.h5")
    results = model.predict(mats)
    print(results)


if __name__ == '__main__':
    predict_from_wav("../../data/interim/P301/P301_no_silence.wav")