from inspect import cleandoc
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import hashlib

CACHE_PATH = "./custom_nodes/canvas/cache/"
PREV_GENERATED_IMAGE = "prev_generated_image"

class CanvasLoader:
    """
    A class that either loads an existing image from a path or creates a new one based on the given parameters.
    """

    def __init__(self):
        self.GENERATION_SEED = 0
        self.prev_use_canvas = False
        self.prev_image_width = 1024
        self.prev_image_height = 1024
        self.prev_fill_img_with = "white"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "use_canvas": (
                    "BOOLEAN",
                    {
                        "tooltip": "Use an existing image as a canvas.",
                        "default": False,
                    },
                ),
                "update_canvas": (
                    "BOOLEAN",
                    {
                        "tooltip": "Replace the current canvas with the most recently generated image.",
                        "default": False,
                    },
                ),
                "image_width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 128,
                        "max": 4096,
                        "step": 64,
                        "display": "number",
                    },
                ),
                "image_height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 128,
                        "max": 4096,
                        "step": 64,
                        "display": "number",
                    },
                ),
                "fill_img_with": (
                    ["white", "black", "random"],
                    {
                        "default": "white"
                    },
                ),
            }
        }

    
    def execute(self, use_canvas, update_canvas, image_width, image_height, fill_img_with):
        if not os.path.exists(CACHE_PATH):
            os.makedirs(CACHE_PATH)
        cached_image_path = os.path.join(CACHE_PATH, f"{PREV_GENERATED_IMAGE}_1.png")
        canvas_path = os.path.join(CACHE_PATH, f"canvas.png")

        if update_canvas:
            if os.path.exists(cached_image_path):
                # open image as a PIL Image object and write the image to canvas_path
                image = Image.open(cached_image_path)
                image.save(canvas_path, "PNG")

        if use_canvas:            
            if os.path.exists(canvas_path):
                # open image as a PIL Image object and convert it to a PyTorch tensor
                image = Image.open(canvas_path).convert("RGB")
                image = torch.from_numpy(np.array(image)).float() / 255.0
                image = image.unsqueeze(0)
        else:
            image = torch.Tensor(torch.empty((1, image_height, image_width, 3), dtype=torch.float32))
            # fill image with the specified value
            if fill_img_with == "white":
                image.fill_(1.0)
            elif fill_img_with == "black":
                image.fill_(0.0)
            elif fill_img_with == "random":
                image = torch.rand((1, image_height, image_width, 3))

        self.prev_use_canvas = use_canvas
        self.prev_image_width = image_width
        self.prev_image_height = image_height
        self.prev_fill_img_with = fill_img_with

        return (image,)
    
    @classmethod
    def IS_CHANGED(cls, use_canvas, update_canvas, image_width, image_height, fill_img_with):
        return np.random.rand()
    
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("canvas_image",)
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "execute"

    OUTPUT_NODE = True
    # OUTPUT_TOOLTIPS = ("",) # Tooltips for the output node

    CATEGORY = "CanvasNodes"

class CanvasCacheUpdater:
    """
    A node that updates the cache with the generated image.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cache_image": (
                    "IMAGE",
                    {},
                ),
                "caching_enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                    },
                ),
            }
        }
    
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True
    CATEGORY = "CanvasNodes"
    FUNCTION = "execute"

    def execute(self, cache_image: torch.Tensor, caching_enabled: bool):
        print("execute canvas cache updater")
        if not caching_enabled:
            return ()

        print("caching enabled")
        # We assume batch_size is 1 for a single image conversion.
        if cache_image.dim() == 4:
            # Squeeze the batch dimension if it's 1
            if cache_image.shape[0] == 1:
                cache_image = cache_image.squeeze(0)
            else:
                raise ValueError("Batch size greater than 1 is not supported for single image conversion.")

        # Convert float values (0-1) to uint8 (0-255)
        image_np = (cache_image * 255).byte().cpu().numpy()

        # Create a PIL Image object
        pil_image = Image.fromarray(image_np)

        if os.path.exists(os.path.join(CACHE_PATH, f"{PREV_GENERATED_IMAGE}_5.png")):
            # Delete the oldest image
            os.remove(os.path.join(CACHE_PATH, f"{PREV_GENERATED_IMAGE}_5.png"))

        # shift all other images back one place
        for i in range(4, 0, -1):
            if os.path.exists(os.path.join(CACHE_PATH, f"{PREV_GENERATED_IMAGE}_{i}.png")):
                os.rename(
                    os.path.join(CACHE_PATH, f"{PREV_GENERATED_IMAGE}_{i}.png"),
                    os.path.join(CACHE_PATH, f"{PREV_GENERATED_IMAGE}_{i + 1}.png"),
                )

        # Save the new image
        pil_image.save(os.path.join(CACHE_PATH, f"{PREV_GENERATED_IMAGE}_1.png"))
        return ()
    
class CanvasSelector:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_canvas_image": (
                    "IMAGE",
                    {},
                ),
                "selector_mask": (
                    "MASK",
                    {},
                ),
                "scale_factor": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 4.0,
                        "step": 0.05,
                    },
                ),
                "pixel_padding": (
                    "INT",
                    {
                        "default": 128,
                        "min": 0,
                        "max": 1024,
                        "step": 8,
                    },
                ),
            }
        }
    
    RETURN_TYPES = ("IMAGE","MASK","CONTEXT")
    RETURN_NAMES = ("selection","selection_mask","canvas_context")
    OUTPUT_NODE = True
    CATEGORY = "CanvasNodes"
    FUNCTION = "execute"

    def adjust_scale(self, scale_factor, height, width):
        print(f"scale_factor: {scale_factor}, height: {height}, width: {width}")
        # after scaling, select the closest multiple of 8 that is smaller than or equal to the raw scaled height and width
        scaled_height = int(height * scale_factor) // 8 * 8
        scaled_width = int(width * scale_factor) // 8 * 8
        
        return (scaled_height, scaled_width)

    def execute(self, scale_factor, pixel_padding, input_canvas_image: torch.Tensor, selector_mask: torch.Tensor):
        # input_canvas_image is (B, H, W, C), selector_mask is either (H, W) or (B, H, W)
        print(f"input_canvas_image shape: {input_canvas_image.shape}, selector_mask shape: {selector_mask.shape}")

        # create selector bounding box
        nonzero = torch.nonzero(selector_mask, as_tuple=True)
        if nonzero[0].numel() == 0 or selector_mask.ndim != 3:
            # Handle empty mask case by creating a mask of the entire image with the shape (1, H, W)
            print("Mask is empty, setting selector mask to entire image...")
            selector_mask = torch.ones(1, input_canvas_image.shape[1], input_canvas_image.shape[2])
            ymin = 0
            xmin = 0
            ymax = selector_mask.shape[1]
            xmax = selector_mask.shape[2]
        else:
            # if we hit this point, we have a non-empty mask; create the bounding box for the nonzero pixels
            # Extract min/max row and column indices
        
            # ComfyUI appears to restrict the possible sizes of latent images to multiples of 8. Because of this, we need to ensure that
            # our calculations are done with multiples of 8.
            ymin = (torch.min(nonzero[1]) ) // 8 * 8
            ymax = (torch.max(nonzero[1]) ) // 8 * 8
            xmin = (torch.min(nonzero[2]) ) // 8 * 8
            xmax = (torch.max(nonzero[2]) ) // 8 * 8

        print(f"bbox top left is ({xmin}, {ymin}) and bottom right is ({xmax}, {ymax})")

        # padding values, i.e. top_pad, left_pad, bottom_pad, and right_pad, are RANGES. They're the number of pixels to add on each side of the bounding box.
        # if necessary, shrink the padding for each side to make sure it doesn't go out of bounds
        top_pad = (pixel_padding if ymin - pixel_padding >= 0 else ymin) // 8 * 8
        left_pad = (pixel_padding if xmin - pixel_padding >= 0 else xmin) // 8 * 8
        bottom_pad = (pixel_padding if ymax + pixel_padding <= input_canvas_image.shape[1] else input_canvas_image.shape[1] - ymax) // 8 * 8
        right_pad = (pixel_padding if xmax + pixel_padding <= input_canvas_image.shape[2] else input_canvas_image.shape[2] - xmax) // 8 * 8

        # padded values, i.e. xmin_padded, ymin_padded, xmax_padded, and ymax_padded, are POINTS. They're the coordinates of the top left and bottom right corners of the padded bounding box
        xmin_padded = xmin - left_pad
        ymin_padded = ymin - top_pad
        xmax_padded = xmax + right_pad
        ymax_padded = ymax + bottom_pad

        selection_height = (ymax_padded - ymin_padded)
        selection_width = (xmax_padded - xmin_padded)

        # create the new bounding box point based on the selection_height and selection_width which are multiples of 8
        ymin = ymin_padded
        xmin = xmin_padded
        ymax = ymin_padded + selection_height
        xmax = xmin_padded + selection_width
        
        # extract the selection from the image
        selection = input_canvas_image[0, ymin:ymax, xmin:xmax, :] 
        selection_mask = selector_mask[0, ymin:ymax, xmin:xmax]

        # interpolate function requires input to be (B, C, H, W), so we rearrange the tensors A G A I N
        selection = selection.permute(2, 0, 1)
        selection = selection.unsqueeze(0)
        selection_mask = selection_mask.unsqueeze(0).unsqueeze(0)

        # if scale_factor is not just 1, scale the selection based on scale_factor
        # we want to remove any floating point inaccuracies, as they lead to noticeable seams. Therefore, we cut off any
        # decimals beyond 10^-2.
        scale_factor = int(scale_factor * 100) / 100

        # the selection and selection_mask should now have the shapes (1, C, H, W) and (1, 1, H, W) respectively
        print(f"before interpolation: selection shape: {selection.shape}, selection_mask shape: {selection_mask.shape}")

        scaled_height, scaled_width = self.adjust_scale(scale_factor, selection.shape[2], selection.shape[3])
        if scale_factor != 1.0:
            scale_transform = transforms.Resize((scaled_height, scaled_width), interpolation=transforms.InterpolationMode.BILINEAR)
            selection = scale_transform(selection)
            selection_mask = scale_transform(selection_mask)
            print(f"after interpolation: selection shape: {selection.shape}, selection_mask shape: {selection_mask.shape}")

        # finally, you guessed it, shuffle back to (B, H, W, C) for the image and (H, W) for the mask
        selection = selection.permute(0, 2, 3, 1)
        selection_mask = selection_mask.squeeze(0)
        print(f"returning selection shape: {selection.shape}, selection_mask shape: {selection_mask.shape}")

        # context format is (image, mask, bbox_dict, original_height, original_width)
        return (selection, selection_mask, (input_canvas_image, {
                "top_left": (xmin, ymin), 
                "bottom_right": (xmax, ymax), 
                "scale_factor": scale_factor
            }
        ))
    
class CanvasMerger:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "canvas_selection": (
                    "IMAGE",
                    {},
                ),
                "canvas_context": (
                    "CONTEXT",
                    {"forceInput": True},
                ),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("merged_image",)
    OUTPUT_NODE = True
    CATEGORY = "CanvasNodes"
    FUNCTION = "execute"

    def execute(self, canvas_selection: torch.Tensor, canvas_context):
        # canvas_context is a tuple that contains the canvas image as the first element and a dictionary containing the top_left and bottom_right coordinates as the second element
        # unpack canvas_context
        canvas_image, bbox_dict = canvas_context
        scale_factor = float(bbox_dict["scale_factor"])
        print(f"scale factor for merging: {scale_factor}")

        top_left_x, top_left_y = bbox_dict["top_left"]
        bottom_right_x, bottom_right_y = bbox_dict["bottom_right"]

        # define the width and height of the bounding box
        width = bottom_right_x - top_left_x
        height = bottom_right_y - top_left_y

        # transform the selection shape from (B, H, W, C) to (B, C, H, W)
        canvas_selection = canvas_selection.permute(0, 3, 1, 2)

        print(f"selection shape: {canvas_selection.shape}")
        # if the image has been scaled, define the transform for the resizing of the selection then resize the selection
        if scale_factor < 0.99 or scale_factor > 1.01:
            print(f"resizing selection to {height}x{width}")
            resize_transform = transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR)
            canvas_selection = resize_transform(canvas_selection)
        print(f"selection shape: {canvas_selection.shape}")

        # transform selection shape back to (B, H, W, C)
        canvas_selection = canvas_selection.permute(0, 2, 3, 1)

        # paste the selection onto the canvas image
        canvas_image[0, top_left_y:bottom_right_y, top_left_x:bottom_right_x, :] = canvas_selection

        return (canvas_image,)


class CanvasTransform:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_canvas_image": (
                    "IMAGE", 
                    {},
                ),
                "operation": (
                    ["crop", "grow"],
                    {
                        "default": "crop",
                    },
                ),
                "grow_mode": (
                    ["constant", "reflect", "replicate", "circular"],
                    {
                        "default": "constant",
                    },
                ),
                "grow_fill_value": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Value to fill new pixels with if grow_mode is 'constant'",
                    },
                ),
                "margin_t": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 8,
                        "tooltip": "Number of pixels to crop/grow from the top of the canvas",
                    },
                ),
                "margin_b": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 8,
                        "tooltip": "Number of pixels to crop/grow from the bottom of the canvas",
                    },
                ),
                "margin_l": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 8,
                        "tooltip": "Number of pixels to crop/grow from the left of the canvas",
                    },
                ),
                "margin_r": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 8,
                        "tooltip": "Number of pixels to crop/grow from the right of the canvas",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("transformed_canvas",)
    OUTPUT_NODE = True
    CATEGORY = "CanvasNodes"
    FUNCTION = "execute"

    def execute(self, input_canvas_image, grow_mode, grow_fill_value, operation, margin_t, margin_b, margin_l, margin_r):

        if margin_t > 0 or margin_b > 0 or margin_l > 0 or margin_r > 0:
            if operation == "crop":
                input_canvas_image = input_canvas_image[:, margin_t:input_canvas_image.shape[1]-margin_b, margin_l:input_canvas_image.shape[2]-margin_r, :]
            elif operation == "grow":
                # shuffle dimensions for transformation, ofc
                input_canvas_image = input_canvas_image.permute(0, 3, 1, 2)
                # pad() only accepts the value kwarg if mode is 'constant'
                if grow_mode == "constant":
                    input_canvas_image = F.pad(input_canvas_image, (margin_l, margin_r, margin_t, margin_b), mode=grow_mode, value=grow_fill_value)
                else:
                    input_canvas_image = F.pad(input_canvas_image, (margin_l, margin_r, margin_t, margin_b), mode=grow_mode)

                # shuffle dimensions back
                input_canvas_image = input_canvas_image.permute(0, 2, 3, 1)

        return (input_canvas_image,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {"CanvasLoader": CanvasLoader,
                       "CanvasCacheUpdater": CanvasCacheUpdater,
                       "CanvasSelector": CanvasSelector,
                       "CanvasMerger": CanvasMerger,
                       "CanvasTransform": CanvasTransform}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {"CanvasLoader": "Canvas Loader",
                             "CanvasCacheUpdater": "Canvas Cache Updater",
                             "CanvasSelector": "Canvas Selector",
                             "CanvasMerger": "Canvas Merger",
                             "CanvasTransform": "Canvas Transform"}