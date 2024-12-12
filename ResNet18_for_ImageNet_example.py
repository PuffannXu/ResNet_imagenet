"""
======================================================================================
 !/usr/bin/env python
 -*- coding: utf-8 -*-
 @Time : 2024/12/12 9:29
 @Author : Pufan Xu
 @Function : 
======================================================================================
"""

from torchinfo import summary
from utils.ResNet import ResNet18_for_Imagenet

n_class = 200

qn_on = 0
fp_on = 0
weight_bit = 0
output_bit = 0
isint = 0
clamp_std = 0
quant_type = "none"
group_number = 0
left_shift_bit = 0

device = 'cuda'

model = ResNet18_for_Imagenet(num_classes=n_class,
                              qn_on=qn_on,
                              fp_on=fp_on,
                              weight_bit=weight_bit,
                              output_bit=output_bit,
                              isint=isint, clamp_std=clamp_std,
                              quant_type=quant_type,
                              group_number=group_number,
                              left_shift_bit=left_shift_bit)

model = model.to(device)
print(model)
summary(model, input_size=(1, 3, 224, 224))  # 输入大小为 (通道数, 高度, 宽度)

print()
