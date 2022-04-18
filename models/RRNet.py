import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """
    Residuals Module
    @input_channel: Size of input channels
    @output_channel: Size of output channels
    """

    def __init__(self, 
                 input_channel=1,
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
    """
    Residuals Module
    @mode: 
        "raw": The raw ResNet model.
        "pre-RNN": Pre-embedding RNN on ResNet model.
        "post-RNN": Post-embedding RNN on ResNet model.
    @num_lable: The number of labels.
    @list_ResBlock_inplanes: The number of channels per residual block in ResBlock list.
    @num_rnn_sequence: The number of RRN sequences.
    @len_spectrum: The length of input spectrum.
    """

    def __init__(self, 
                mode="raw",
                num_lable=3,
                list_ResBlock_inplanes= [4,8,16],
                num_rnn_sequence = 40,
                len_spectrum=7200,
                ):
        super(RRNet, self).__init__()



        
        assert len(list_ResBlock_inplanes) < 6
        assert len_spectrum % num_rnn_sequence == 0

        self.mode = mode
        self.len_spectrum = len_spectrum
        self.num_rnn_sequence = num_rnn_sequence
        self.list_ResBlock_inplanes = list_ResBlock_inplanes.copy()
        self.list_ResBlock_inplanes.insert(0, 1)


        if self.mode == "pre-RNN":
            self.rnn = nn.RNN(
                input_size=self.len_spectrum // self.num_rnn_sequence,
                hidden_size=self.len_spectrum // self.num_rnn_sequence,
                num_layers=1,
                batch_first=True,
            )
        
        self.ResBlock_list = []
        for i in range(len(self.list_ResBlock_inplanes)-1):
            self.ResBlock_list.append(
                nn.Sequential(
                    ResBlock(self.list_ResBlock_inplanes[i], self.list_ResBlock_inplanes[i+1]),
                    nn.AvgPool1d(2),
                )
            )
        self.ResBlock_list = nn.Sequential(*self.ResBlock_list)

        if self.mode == "post-RNN":
            self.rnn = nn.RNN(
                input_size=self.list_ResBlock_inplanes[-1] * 
                            (self.len_spectrum // (2**(len(self.list_ResBlock_inplanes)-1))) //
                            self.num_rnn_sequence,
                hidden_size=self.list_ResBlock_inplanes[-1] * 
                            (self.len_spectrum // (2**(len(self.list_ResBlock_inplanes)-1))) //
                            self.num_rnn_sequence,
                num_layers=1,
                batch_first=True,
            )

        self.fc1 = nn.Sequential(
            nn.Linear(
                in_features=self.list_ResBlock_inplanes[-1] * 
                            (self.len_spectrum // (2**(len(self.list_ResBlock_inplanes)-1))),
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

        assert L % self.num_rnn_sequence == 0

        if self.mode == "pre-RNN":
            x, h_n = self.rnn(x.view(B, self.num_rnn_sequence, -1))
            del h_n
     
        x = x.reshape(-1, 1, L)
     
        x = self.ResBlock_list(x)
        x = torch.relu(x)
        if self.mode == "post-RNN":
            x = x.permute(0, 2, 1).reshape(B, self.num_rnn_sequence, -1)
            x, h_n = self.rnn(x)
            del h_n

        x = self.fc1(x.flatten(1))
        x = self.fc2(x)
        return self.mu(x), self.sigma(x)


    def get_loss(self, y_true, y_pred, y_sigma):

        return (torch.log(y_sigma)/2+ (y_true-y_pred)**2/(2*y_sigma)).mean() + 5