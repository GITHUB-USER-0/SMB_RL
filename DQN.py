import torch
from torch import nn

class DQN(nn.Module):

    def __init__(self, actionSpaceSize, RGB = False, width = 256, height = 240):
        """ Initialize deep Q network 

        Inputs: 
            actionSpaceSize - the size of the discrete action space
                              determines the number of output nodes
        Output : 
            DQN - deep Q Network
            """

        # inherit from nn.Module to support load_state_dict(), see class head
        super(DQN, self).__init__()
        self.width = width
        self.height = height
        self.RGB = RGB
        
        IC1 = 3 if RGB else 1 # grayscale or RGB input channel count
        OC1 = 32
        IC2 = OC1
        OC2 = 64
        IC3 = OC2
        OC3 = 64
    
        # see a reference template for architecture here: https://www.tensorflow.org/tutorials/images/cnn

        # output size of a convolutional layer
        # W_out = \frac{W_in - Kernel_W + 2 * Padding}{Stride} + 1
        # 256 * 240 base image
        # may benefit from being clipped
        # but use of AdaptiveMax or AdaptiveAvg pool makes this somewhat more convenient
        # https://discuss.pytorch.org/t/what-is-adaptiveavgpool2d/26897

        PADDING = 0
        STRIDE = 2
        
        self.network = nn.Sequential(
            nn.Conv2d(in_channels = IC1, 
                      out_channels = OC1, 
                      kernel_size = 3, 
                      padding = PADDING,
                      stride = STRIDE),
            nn.AdaptiveMaxPool2d( (width // 2, height // 2) ), # note integer division
            nn.Conv2d(in_channels = IC2,
                      out_channels = OC2,
                      kernel_size = 3,
                      padding = PADDING,
                      stride = STRIDE),
            nn.AdaptiveMaxPool2d( (width // 4, height // 4) ),
            nn.Conv2d(in_channels = IC3,
                      out_channels = OC3,
                      kernel_size = 3,
                      padding = PADDING,
                      stride = STRIDE),
            nn.AdaptiveAvgPool2d( (width // 8, height // 8) ),
            nn.Flatten(),
            nn.Linear(OC3 * width // 8 * height // 8, actionSpaceSize * 4),
            nn.ReLU(),
            nn.Linear(actionSpaceSize * 4, actionSpaceSize)
        )


    def forward(self, x):
        """ Perform a forward pass on the model """
        y = self.network(x)

        return(y)

    # note, no need to define a loss function,
    # this will be handled automatically by loss.backward() and
    # through PyTorch convenience/magic
        
    def loadModel(self, filename):
        """ Load weights from a saved copy of the network """
        model = torch.load(filename, weights_only = False)
        self.load_state_dict(model)

    def saveModel(self, filename):
        """ Save existing weights of the network to a file """
        torch.save(self.state_dict(), filename)

    def __repr__(self):
        s = ''
        style = "RGB" if self.RGB else "grayscale"
        s += f"Network is being processing input images as {style}\n"
        s += f"Input width:  {self.width}\n"
        s += f"Input height: {self.height}\n"
        s += str(self.network)
        
        
        return( s )