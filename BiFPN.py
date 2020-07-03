import torch
import torch.nn as nn
import torch.nn.functional as F



###### BiFPN ######
class BiFPN(nn.Module):
    def __init__(self, num_chs, conv_chs, first_time=False, epsilon=1e-4, attenion=True):
        super().__init__()
        self.epsilon = epsilon
        self.first_time = first_time

        ## Conv Layers
        if self.first_time:
            self.p3_down_chs = nn.Sequential(
                nn.Conv2d(conv_chs[0], num_chs, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(num_chs, momentum=0.01, eps=1e-3)
            )
            self.p45_down_chs = nn.Sequential(
                nn.Conv2d(conv_chs[1], num_chs, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(num_chs, momentum=0.01, eps=1e-3)
            ) 
            self.p_down_chs = nn.Sequential(
                nn.Conv2d(conv_chs[2], num_chs, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(num_chs, momentum=0.01, eps=1e-3)
            )

            self.p5_to_p6 = nn.Sequential(
                nn.Conv2d(conv_chs[2], num_chs, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(num_chs, momentum=0.01, eps=1e-3),
                nn.MaxPool2d(3, 2)
            )

    def forward(self, inputs):
        p3_out, p4_out, p5_out, p6_out, p7_out = (self._forward_fast_attention(inputs)
            if self.attenion else self._forward(inputs))

    def _forward_fast_attention(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_chs(p3)
            p4_in = self.p3_down_chs(p4)
            p5_in = self.p3_down_chs(p5)
