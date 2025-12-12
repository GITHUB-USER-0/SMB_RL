import numpy as np
#from PIL.ImageOps import grayscale
from PIL.ImageOps import fit as resize
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F

# environment
import gym_super_mario_bros 
import nes_py      
from nes_py.wrappers import JoypadSpace

import torch

def prefillBuffer(BUFFER_SIZE, env, actionSpace):
    """ prefill the replay buffer

    Note that if random levels are selected outside this function, it is likely that the 
    buffer size may need to be much larger than if one focused on just one level.
    """

    # grayscale
    rb = ReplayBuffer(BUFFER_SIZE, (3, ADJ_FRAME_HEIGHT, ADJ_FRAME_WIDTH) )

    # need an environment to get transitions
    state, info = env.reset(seed = seed) if SEED else env.reset()

    # uniform random action selection
    # it may be worth pre-weighting here with TAS inputs
    # see 'on TAS and ROMs.ipynb'
    # Space.sample(probability) as per: https://gymnasium.farama.org/api/spaces/
    state, reward, terminated, truncated, info = env.step(env.action_space.sample())


    resetCount              = 0
    transitionCount         = 0 # counter of transitions filled into buffer
    transitionsCurrentLevel = 0 # counter of per-level transitions, 
                                # helps to avoid having too many transitions from one level
    while transitionCount < BUFFER_SIZE:
        
        old_state = state
        state, reward, terminated, truncated, info = env.step(env.action_space.sample())

        # reset environment if dead, or too many samples from current environment
        if terminated or truncated or transitionsCurrentLevel % (BUFFER_SIZE // 10) == 0:
            state, info = env.reset(seed = seed) if SEED else env.reset()
            state, reward, terminated, truncated, info = env.step(env.action_space.sample())
            resetCount += 1
            transitionsCurrentLevel = 0
            next
            
        rb.storeTransition((
            preprocessFrame(old_state),
            torch.randint(low = 0, high = env.action_space.n, size = (1,)),
            torch.randint(high = 10, size = (1,)),   
            preprocessFrame(state)
        ))

        transitionCount += 1
        transitionsCurrentLevel += 1

        if BUFFER_SIZE >= 100:
            if i % (BUFFER_SIZE // 10) == 0:
                print(f"Filling buffer slot {i} of {BUFFER_SIZE}")
    print(f"Filling buffer sampled from {resetCount} level starts (not guaranteed unique)") 

    return(rb)


def preprocessFrame(frame,
                    VTRIM_TOP = 36, 
                    HTRIM_LEFT = 36, 
                    HTRIM_RIGHT = 16,
                    FRAME_WIDTH = 256,
                    ADJ_FRAME_HEIGHT = 100, 
                    ADJ_FRAME_WIDTH = 100,
                    method = 'torch'
                   ):
    """ Preprocess an input image frame

    Inputs:
        frame - frame of RGB image data, from the environment
                a numpy array of shape (240, 256, 3)
                                  Height  x Width  x Channels
                                      Rows  x Col x Channels

    Note that the stacking of sequential frames, as per Mnih et al. 2013 
    is to be handled separately.
    """
    # might be interesting to benchmark this preprocessing--how much time is spent on this part of the pipeline
    frame = frame[VTRIM_TOP:, :, :]
    frame = frame[:, HTRIM_LEFT:FRAME_WIDTH - HTRIM_RIGHT, :]
    frame = frame.copy() # makes contiguous to avoid negative strides error

    # generative AI input
    if method == "torch":
        frame = torch.from_numpy(frame).float() / 255.0
        frame = F.interpolate(
            frame.permute(2,0,1).unsqueeze(0), size=(ADJ_FRAME_HEIGHT,ADJ_FRAME_WIDTH)
        )
        frame = frame.squeeze(0).permute(1,2,0).numpy()

    # human-generated
    elif method == "PIL":
        # PIL presumably less efficient, but feels more human-readable and allows easier inspection
        # of the output
        # convert to PIL image
        frame = Image.fromarray(frame)
    
        # not converted to grayscale
        # avoid grayscale, see 'preprocessing frames.ipynb' for reference
        #frame = grayscale(frame)
    
        # downscale
        frame = resize(frame, (ADJ_FRAME_HEIGHT, ADJ_FRAME_WIDTH) )
    
        # convert back to numpy array
        frame = np.array(frame)
    
        # normalize -- necessary?
        frame = frame / 255.0
        
    return(frame)

def tensorify(frame, method = 'fast'):
    """ Convert from the numpy array into a happy tensor.

    The input numpy array is going to be of shape (H x W x C)
    The output tensor will be of shape (1 x C x H x W), it will have
    a new batch dimension, and be shifted to be C x H x W
    """
    if isinstance(frame, torch.Tensor):
        return(frame)
    
    elif method == "fast":
        return torch.from_numpy(frame.copy()).float().permute(2,0,1).unsqueeze(0)
        
    elif method == "slow":
    
        # address ValueError: at least one stride is negative ...
        # frame.copy to address?
        t = torch.from_numpy(frame)  
    
        # re-arrange the dimensions
        # dtype to address: RuntimeError: Input type (double) and bias type (float) should be the same
        t = torch.tensor(t, dtype = torch.float32).permute(2, 0, 1)
    
        # add the batch dimension
        t = t.unsqueeze(0)
    
        # UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
        return(t)

def initializeEnvironment(rom = 'v0', 
                          randomLevel = True, 
                          stagesList = ['1-1'], 
                          excludeList = None,
                          buttonList = [['NOOP']]):
    """Initialize environment in gymnasium. 
    Sets the list of acceptable actions.

    Inputs:
        mode - the set of available actions, either from JoypadSpace, or
               a custom set of actions provided as a list of lists of actions
        rom - the selected ROM
    Outputs:
        (env, actionSpace_init) - a tuple of the gymnasium environment and actionSpace

    Note that v0 offers a traditional view, corresponding 
    to 'super-mario-bros.nes' included with package
    with MD5 of: 673913a23cd612daf5ad32d4085e0760
    and is "Super Mario Bros. (E) (REVA) [!].nes SourceDB: GoodNES 3.23"
    as per: https://tasvideos.org/Games/1/Versions/List
    v3, in turn, is a simplified rectangular view
    this may have been generated by kautenja
    and does not appear in the TAS collections that I saw

    v0 is more visually appealing, but it seems plausible that v3
    would train faster.
    """
    if rom == 'v0':
        s = 'SuperMarioBros-v0'
    elif rom == 'v3':
        s = 'SuperMarioBros-v3'
    else:
        print("Error in ROM selection.")
        return(None)

    ALL_STAGES = [
        "1-1","1-2","1-3","1-4",
        "2-1","2-2","2-3","2-4",
        "3-1","3-2","3-3","3-4",
        "4-1","4-2","4-3","4-4",
        "5-1","5-2","5-3","5-4",
        "6-1","6-2","6-3","6-4",
        "7-1","7-2","7-3","7-4",
        "8-1","8-2","8-3","8-4"
    ]

    if stagesList and excludeList:
        raise ValueError("Specify either stagesList OR excludeList, not both.")

    if stagesList is not None:
        print("Playing selection of stages")
        for s in stagesList:
            if s not in ALL_STAGES:
                raise ValueError(f"Invalid stage '{s}'. Must be one of {ALL_STAGES}.")
        selected_stages = stagesList

    elif excludeList is not None:
        print("Excluding specific stages")
        selected_stages = [s for s in ALL_STAGES if s not in excludeList]
        if len(selected_stages) == 0:
            raise ValueError("Excluding all stages leaves no playable levels.")
    else:
        selected_stages = None

    # as per documentation, SuperMarioBrosRandomStages-v0 will randomly select world, level combinations
    if randomLevel:
        s = s.split('-')
        s = 'RandomStages-'.join(s)

    if selected_stages is not None:
        env = gym_super_mario_bros.make(s, stages = stagesList)
    else:
        env = gym_super_mario_bros.make(s)

    if buttonList == "simple":
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        actionSpace_init = SIMPLE_MOVEMENT
    elif buttonList == "complex":
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        actionSpace_init = COMPLEX_MOVEMENT
    elif buttonList == "rightOnly":
        env = JoypadSpace(env, RIGHT_ONLY)
        actionSpace_init = RIGHT_ONLY
    else:
        # provide a predefined list of string actions
        # eg., [['NOOP'], ['right', 'A'], ['right', 'B'], ['right', 'B', 'A']]
        env = JoypadSpace(env, buttonList)
        actionSpace_init = buttonList

    return( (env, actionSpace_init) )

def saveDiagnosticImage(folder, 
                        frame,
                        annotations,
                        step = None, 
                        action = None, 
                        x_pos = None, 
                        y_pos = None, 
                        epsilonFlag = None,
                        rectangle = None,
                       ):
    """ Save a snapshot with additional text info burned in.
    
    Inputs:
        folder - the folder in which to save images, NB., one folder per episode 
        frame - TODO, standardize, revise with prefillbuffer
        step, action - current step and action for burning in
        x_pos, y_pos - current x and y position for burning in
        
        """

    # this method had meaningful assistance from generativeAI
    # Convert torch.Tensor to NumPy
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu()
        if frame.shape[0] == 1:  # Grayscale
            frame = frame.squeeze(0).numpy()  # (H, W)
        else:  # RGB
            frame = frame.permute(1, 2, 0).numpy()  # (H, W, C)
    else:
        frame = frame.copy()

    # Ensure dtype is uint8
    frame = np.clip(frame, 0, 255).astype(np.uint8)

    image = Image.fromarray(frame)    
    draw = ImageDraw.Draw(image)

    # see: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes
    if image.mode == "L": # grayscale image
        backgroundColor = 0 # black
        textColor = 255 # white for grayscale
    if image.mode == "RGB":
        # white background, black text
        backgroundColor = (255, 255, 255)
        textColor = (0,   0,   0  )

    if annotations:
        ## the following line fails in TensorFlow environment
        #font = ImageFont.truetype("arial.ttf", size = 20)
        text_annotation = ""
        text_annotation += str(f"step: {step:0>7}\naction: {action}\n")
        text_annotation += "Epsilon: True\n" if epsilonFlag else "Epsilon:\n"         
        text_annotation += str(f"x: {x_pos:0>3}, y: {y_pos:0>3}\n")        
    
        #              x0, y0, x1, y1
        draw.rectangle(rectangle,          fill = backgroundColor)
        draw.text((0, 0), text_annotation, fill = textColor)
 
    # use of padding in filename is helpful for passing 
    # in to Kdenlive as an Image Sequence for video review
    # in quick testing, .png files were smaller than .jpeg
    image.save(f"{folder}/{step:0>7}.png")
