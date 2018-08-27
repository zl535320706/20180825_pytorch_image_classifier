# -*- coding: utf-8 -*-
# file: predict.py
# author: JinTian
# time: 10/05/2017 9:52 AM
# Copyright 2017 JinTian. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""
this file predict single image using the model we previous trained.
"""
from models.fine_tune_model import fine_tune_model
from global_config import *
import torch
import os
import sys
from data_loader.data_loader import DataLoader
from torch.autograd import Variable

def predict_single_image(inputs, classes_name):
    model = fine_tune_model()
    if USE_GPU:
        inputs = inputs.cuda()
    if not os.path.exists(MODEL_SAVE_FILE):
        print('can not find model save file.')
        exit()
    else:
        if USE_GPU:
            model.load_state_dict(torch.load(MODEL_SAVE_FILE))
        else:
            model.load_state_dict(torch.load(MODEL_SAVE_FILE, map_location=lambda storage, loc: storage))
        outputs = model(inputs)
        _, prediction_tensor = torch.max(outputs.data, 1)
        if USE_GPU:
            prediction = prediction_tensor.cpu().numpy()[0][0]
            print('predict: ', prediction)
            print('this is {}'.format(classes_name[prediction]))
        else:
            prediction = prediction_tensor.numpy()[0][0]
            print('predict: ', prediction)
            print('this is {}'.format(classes_name[prediction]))


def predict():
        data_loader = DataLoader(data_dir=DATA_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, mode='test')
        model = fine_tune_model()
        model.load_state_dict(torch.load(MODEL_SAVE_FILE))
        model.train(False)
        img_list = data_loader.data_sets['test'].imgs
        count = 1
        idx_all = torch.zeros(TEST_SIZE, 11)
        for batch_data in data_loader.load_data(data_set='test'):
            inputs, _ = batch_data
            inputs = Variable(inputs.cuda())
            inputs_size = inputs.size()
            outputs = model(inputs)
            outputs_ls = torch.nn.functional.softmax(outputs)
            if inputs_size[0] == BATCH_SIZE:
                idx_all[(count - 1) * (inputs_size[0]):count * (inputs_size[0]), :] = outputs_ls.data.cpu()
            else:
                idx_all[(TEST_SIZE // BATCH_SIZE * BATCH_SIZE): TEST_SIZE, :] = outputs_ls.data.cpu()
            print('{} Processing...'.format(count))
            count = count + 1

        f = open("result_20180827.csv", "w")
        label_list = ['defect_1', 'defect_10', 'defect_2', 'defect_3', 'defect_4', 'defect_5', 'defect_6',
                      'defect_7', 'defect_8', 'defect_9', 'norm']
        print('filename|defect,probability', file=f)
        for i in range(TEST_SIZE):
            path = img_list[i]
            name = path[0].split('/')[-1]
            for j in range(11):
                if idx_all[i, j] > 0.9999999999:    idx_all[i, j] = 0.9999999999
                if idx_all[i, j] < 0.0000000001:    idx_all[i, j] = 0.0000000001
                record = name + '|' + label_list[j] + ',' + '{:.10f}'.format(idx_all[i, j])
                print(record, file=f)
                print('{}/{} printing...'.format(i + 1, j + 1))
        f.close()

if __name__ == '__main__':
    predict()



