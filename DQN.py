import torch
from torch import nn

class DQN(nn.Module):

#    def __init__(self, actionSpaceSize, RGB=False, width=256, height=240):

    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        c, h, w = input_shape
        
        self.featureExtractor = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(16, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU()
        )

        ## with input from generative AI
        # Compute conv output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            self.n_flat = self.featureExtractor(dummy).view(1, -1).size(1)
        
        self.fc = nn.Sequential(
            nn.Linear(self.n_flat, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
        self.network = nn.Sequential(self.featureExtractor, nn.Flatten(), self.fc)

    def forward(self, x):
        """ Perform a forward pass on the model """
        #print(f"Forward pass: {x.shape = }") # [1, 3, 100, 100]
        #print(f"Length of x: {len(x)}")
        #print(x.dtype)
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
        #style = "RGB" if self.RGB else "grayscale"
        #s += f"Network is processing input images as {style}\n"
        #s += f"Input width:  {self.width}\n"
        #s += f"Input height: {self.height}\n"
        s += f"Input shape: {self.input_shape}\n"
        s += f"First FC layer number of nodes: {self.n_flat}\n"
        s += str(self.network)
        
        
        return( s )