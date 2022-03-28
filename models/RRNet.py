import torch
import torch.nn as nn


class ResBlock(nn.Module):

    def __init__(self, 
                 input_channel=4,
                 output_channel=4,
                 ):
        super(ResBlock, self).__init__()

        self.ResBlock = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channel,
                out_channels=output_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=output_channel,
                out_channels=output_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
        )

        self.downsample = nn.Sequential()
        if input_channel != output_channel:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                ),
            )

    def forward(self, x):
        return self.ResBlock(x) + self.downsample(x) 


class RRNet(nn.Module):

    def __init__(self, 
                num_lable=1,
                mode="raw",
                ResBlock_inplanes_list= [4,8,32,64,128],
                ):
        super(RRNet, self).__init__()
        
        assert len(ResBlock_inplanes_list) < 6

        self.ResBlock_inplanes_list = ResBlock_inplanes_list.copy()
        self.ResBlock_inplanes_list.insert(0, 1)

        self.mode = mode

        if self.mode == "pre-RNN":
            self.rnn = nn.RNN(
                input_size=360,
                hidden_size=360,
                num_layers=1,
                batch_first=True,
            )
        
        self.ResBlock_list = []
        for i in range(len(self.ResBlock_inplanes_list)-1):
            self.ResBlock_list.append(
                nn.Sequential(
                    ResBlock(self.ResBlock_inplanes_list[i], self.ResBlock_inplanes_list[i+1]),
                    nn.AvgPool1d(2),
                )
            )
        self.ResBlock_list = nn.Sequential(*self.ResBlock_list)

        if self.mode == "post-RNN":
            self.rnn = nn.RNN(
                input_size=360,
                hidden_size=360,
                num_layers=1,
                batch_first=True,
            )

        self.fc1 = nn.Sequential(
            nn.Linear(
                in_features=self.ResBlock_inplanes_list[-1] * (7200 // (2**(len(self.ResBlock_inplanes_list)-1))),
                out_features=256,
            ),
            nn.ReLU(), 
            nn.Dropout(p=0.3),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(
                in_features=256,
                out_features=128,
            ),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )
        self.mu = nn.Linear(
            in_features=128,
            out_features=num_lable,
        )
        self.sigma = nn.Sequential(
            nn.Linear(
                in_features=128,
                out_features=num_lable,
            ),
            nn.Softplus()
        )
        
    def forward(self, x):
        B,L = x.size()

        if self.mode == "pre-RNN":
            x, h_n = self.rnn(x.view(-1,20,360))
            del h_n
     
        x = x.reshape(-1, 1, 7200)
     
        x = self.ResBlock_list(x)
        x = torch.relu(x)

        if self.mode == "post-RNN":
            x = x.permute(0, 2, 1).reshape(B, -1, 360)
            x, h_n = self.rnn(x)
            del h_n

        x = self.fc1(x.flatten(1))
        x = self.fc2(x)
        return self.mu(x), self.sigma(x)


    def get_loss(self, y_true, y_pred, y_sigma):

        return (torch.log(y_sigma)/2+ (y_true-y_pred)**2/(2*y_sigma)).mean() + 5