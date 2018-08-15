# Rain-Detection-And-Removal

It is the basic implementation of "Clearing the Skies: A Deep Network Architecture
for Single-Image Rain Removal" by "Xueyang Fu, Jiabin Huang, Xinghao Ding, Yinghao Liao, and John Paisley".
Each picture element is divided into base layer(low filter passed) and detailed layer. The detailed layer is then passed through a CNN called RESNET. The obtained enhanced detailed layer is combined to base layer to obtain the output image(without rain). 
