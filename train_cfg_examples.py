"""
======================================================================================
 !/usr/bin/env python
 -*- coding: utf-8 -*-
 @Time : 2024/12/13 19:23
 @Author : Pufan Xu
 @Function : 
======================================================================================
"""

# 训练20个epoch
EPOCH = 20

# ============= FP8 Software Baseline =============
qn_on = 0
fp_on = 1
weight_bit = 0
output_bit = 0
isint = 0
clamp_std = 0
quant_type = "none"
group_number = 0
left_shift_bit = 0

model_name = "ResNet18_imagenet_fp8_wo_hw_epoch20.pth"

# ============= FP8 Layer Align =============
qn_on = 0
fp_on = 2
weight_bit = 0
output_bit = 0
isint = 0
clamp_std = 0
quant_type = "layer"
group_number = 0
left_shift_bit = 0

model_name = "ResNet18_imagenet_fp8_w_hw_layer_epoch20.pth"

# ============= FP8 Channel Align =============
qn_on = 0
fp_on = 2
weight_bit = 0
output_bit = 0
isint = 0
clamp_std = 0
quant_type = "channel"
group_number = 0
left_shift_bit = 0

model_name = "ResNet18_imagenet_fp8_w_hw_channel_epoch20.pth"

# ============= FP8 Group9 Align =============
qn_on = 0
fp_on = 2
weight_bit = 0
output_bit = 0
isint = 0
clamp_std = 0
quant_type = "group"
group_number = 9
left_shift_bit = 0

model_name = "ResNet18_imagenet_fp8_w_hw_group9_epoch20.pth"

# ============= FP8 Group72 Baseline =============
qn_on = 0
fp_on = 2
weight_bit = 0
output_bit = 0
isint = 0
clamp_std = 0
quant_type = "group"
group_number = 72
left_shift_bit = 0

model_name = "ResNet18_imagenet_fp8_w_hw_group72_epoch20.pth"

# ============= FP8 Group288 Baseline =============
qn_on = 0
fp_on = 2
weight_bit = 0
output_bit = 0
isint = 0
clamp_std = 0
quant_type = "group"
group_number = 288
left_shift_bit = 0

model_name = "ResNet18_imagenet_fp8_w_hw_group288_epoch20.pth"

# ============= I8W8 Software Baseline =============
qn_on = 1
fp_on = 0
weight_bit = 8
output_bit = 8
isint = 0
clamp_std = 0
quant_type = "none"
group_number = 0
left_shift_bit = 0

model_name = "ResNet18_imagenet_I8W8_epoch20.pth"

# ============= I4W4 Software Baseline =============
qn_on = 1
fp_on = 0
weight_bit = 4
output_bit = 4
isint = 0
clamp_std = 0
quant_type = "none"
group_number = 0
left_shift_bit = 0

model_name = "ResNet18_imagenet_I4W4_epoch20.pth"