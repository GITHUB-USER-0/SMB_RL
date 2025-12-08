# helpers.py
import numpy as np
#from PIL.ImageOps import fit as resize
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

import torch

def initializeEnvironment(
    rom='v0',
    randomLevel=True,
    stagesList=['1-1'],
    buttonList='rightOnly'
):
    """Create a Mario environment and wrap with JoypadSpace."""
    if rom == 'v0':
        base_id = 'SuperMarioBros-v0'
    elif rom == 'v3':
        base_id = 'SuperMarioBros-v3'
    else:
        raise ValueError("rom must be 'v0' or 'v3'")

    if randomLevel:
        # e.g. SuperMarioBrosRandomStages-v0
        parts = base_id.split('-')
        env_id = 'RandomStages-'.join(parts)
    else:
        env_id = base_id

    if stagesList:
        env = gym_super_mario_bros.make(env_id, stages=stagesList)
    else:
        env = gym_super_mario_bros.make(env_id)

    if buttonList == "simple":
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        actionSpace = SIMPLE_MOVEMENT
    elif buttonList == "complex":
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        actionSpace = COMPLEX_MOVEMENT
    elif buttonList == "rightOnly":
        env = JoypadSpace(env, RIGHT_ONLY)
        actionSpace = RIGHT_ONLY
    else:
        # custom list of lists of strings, e.g. [['NOOP'], ['right', 'A']]
        env = JoypadSpace(env, buttonList)
        actionSpace = buttonList

    return env, actionSpace

def preprocessFrame(
    frame,
    VTRIM_TOP=36,
    HTRIM_LEFT=36,
    HTRIM_RIGHT=16,
    FRAME_WIDTH=256,
    ADJ_FRAME_HEIGHT=100,
    ADJ_FRAME_WIDTH=100,
):
    """
    Torch-only version.
    Input:  (240,256,3) numpy array or torch tensor
    Output: (ADJ_FRAME_HEIGHT, ADJ_FRAME_WIDTH, 3) float32 in [0,1]
    """
    # Ensure tensor and float32
    if not isinstance(frame, torch.Tensor):
        frame = torch.from_numpy(frame)
    frame = frame.to(torch.float32)

    # Crop vertically and horizontally
    frame = frame[VTRIM_TOP:, HTRIM_LEFT:FRAME_WIDTH - HTRIM_RIGHT, :]   # (H,W,3)

    # Rearrange to NCHW for F.interpolate
    frame = frame.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

    # Convert to [0,1]
    frame = frame / 255.0

    # Resize using bilinear interpolation
    frame = F.interpolate(
        frame,
        size=(ADJ_FRAME_HEIGHT, ADJ_FRAME_WIDTH),
        mode='bilinear',
        align_corners=False,
    )

    # Back to HWC numpy-like format (but still tensor)
    frame = frame.squeeze(0).permute(1, 2, 0)  # (H,W,3)

    return frame


# def preprocessFrame(
#     frame,
#     VTRIM_TOP=36,
#     HTRIM_LEFT=36,
#     HTRIM_RIGHT=16,
#     FRAME_WIDTH=256,
#     ADJ_FRAME_HEIGHT=100,
#     ADJ_FRAME_WIDTH=100,
# ):
#     """
#     Crop HUD/borders, downscale, normalize, keep RGB.
#     Input:  (240, 256, 3) numpy array
#     Output: (ADJ_FRAME_HEIGHT, ADJ_FRAME_WIDTH, 3) float32 in [0, 1]
#     """
#     # Crop vertically and horizontally
#     frame = frame[VTRIM_TOP:, :, :]
#     frame = frame[:, HTRIM_LEFT:FRAME_WIDTH - HTRIM_RIGHT, :]

#     # Convert to PIL, resize, back to numpy
#     frame = Image.fromarray(frame)
#     frame = resize(frame, (ADJ_FRAME_HEIGHT, ADJ_FRAME_WIDTH))
#     frame = np.array(frame).astype(np.float32)

#     # Normalize
#     frame /= 255.0
#     return frame


def tensorify(frame):
    """
    Convert HWC numpy -> NCHW torch tensor: (1, C, H, W), float32.
    """
    if isinstance(frame, torch.Tensor):
        # assume already (C, H, W) or (1, C, H, W)
        if frame.ndim == 3:
            return frame.unsqueeze(0)
        elif frame.ndim == 4:
            return frame
        else:
            raise ValueError("Unexpected tensor shape for frame")

    # ensure contiguous
    frame = np.array(frame, copy=True)

    # HWC -> CHW
    t = torch.from_numpy(frame).permute(2, 0, 1)   # (C, H, W)
    t = t.unsqueeze(0)                             # (1, C, H, W)
    return t

def saveDiagnosticImage(folder, frame, step, action, x_pos, y_pos, rectangle):
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
    ## fails in TensorFlow environment
    #font = ImageFont.truetype("arial.ttf", size = 20)
    text_annotation = ""
    text_annotation += str(f"step: {step:0>7}\naction: {action}\n")
    text_annotation += str(f"x: {x_pos:0>3}, y: {y_pos:0>3}\n")

    # see: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes
    if image.mode == "L": # grayscale image
        backgroundColor = 0 # black
        textColor = 255 # white for grayscale
    if image.mode == "RGB":
        # white background, black text
        backgroundColor = (255, 255, 255)
        textColor = (0,   0,   0  )

    #              x0, y0, x1, y1
    draw.rectangle(rectangle,          fill = backgroundColor)
    draw.text((0, 0), text_annotation, fill = textColor)
 
    # use of padding in filename is helpful for passing 
    # in to Kdenlive as an Image Sequence for video review
    # in quick testing, .png files were smaller than .jpeg
    image.save(f"./{folder}/{step:0>7}.png")
