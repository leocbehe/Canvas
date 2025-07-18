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

class Example:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict):
        Tell the main program input parameters of nodes.
    IS_CHANGED:
        optional method to control when the node is re executed.

    Attributes
    ----------
    RETURN_TYPES (`tuple`):
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
        Return a dictionary which contains config for all input fields.
        Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
        Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
        The type can be a list for selection.

        Returns: `dict`:
            - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
            - Value input_fields (`dict`): Contains input fields config:
                * Key field_name (`string`): Name of a entry-point method's argument
                * Value field_config (`tuple`):
                    + First value is a string indicate the type of field or a list for selection.
                    + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "image": ("Image", {"tooltip": "This is an image"}),
                "int_field": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,  # Minimum value
                        "max": 4096,  # Maximum value
                        "step": 64,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
                "float_field": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                        "round": 0.001,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "number",
                    },
                ),
                "print_to_screen": (["enable", "disable"],),
                "string_field": (
                    "STRING",
                    {
                        "multiline": False,  # True if you want the field to look like the one on the ClipTextEncode node
                        "default": "Hello World!",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    # RETURN_NAMES = ("image_output_name",)
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "test"

    # OUTPUT_NODE = False
    # OUTPUT_TOOLTIPS = ("",) # Tooltips for the output node

    CATEGORY = "Example"

    def test(self, image, string_field, int_field, float_field, print_to_screen):
        if print_to_screen == "enable":
            print(
                f"""Your input contains:
                string_field aka input text: {string_field}
                int_field: {int_field}
                float_field: {float_field}
            """
            )
        # do some processing on the image, in this example I just invert it
        image = 1.0 - image
        return (image,)

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    # @classmethod
    # def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""


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
            image = torch.Tensor(torch.empty((1, image_height, image_width, 3), dtype=torch.float16))
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
    def IS_CHANGED(cls, use_canvas, image_width, image_height, fill_img_with):
        # generate a hash of the image contents of prev_generated_image_1.png
        cached_image_path = os.path.join(CACHE_PATH, f"{PREV_GENERATED_IMAGE}_1.png")
        canvas_path = os.path.join(CACHE_PATH, "canvas.png")
        if use_canvas:
            if os.path.exists(canvas_path):
                with open(canvas_path, "rb") as f:
                    image_hash = hashlib.sha256(f.read()).hexdigest()
            else:
                image_hash = None
        else:
            if os.path.exists(cached_image_path):
                with open(cached_image_path, "rb") as f:
                    image_hash = hashlib.sha256(f.read()).hexdigest()
            else:
                image_hash = None

        prev_params = (image_hash, use_canvas, image_width, image_height, fill_img_with)
        return prev_params
    
    
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

    def execute(self, scale_factor, pixel_padding,input_canvas_image: torch.Tensor, selector_mask: torch.Tensor):
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
            # Extract min/max row and column indices
            ymin = torch.min(nonzero[1])  # Corrected: Use nonzero[1] for height (row)
            ymax = torch.max(nonzero[1])  # Corrected: Use nonzero[1] for height (row)
            xmin = torch.min(nonzero[2])  # Corrected: Use nonzero[2] for width (column)
            xmax = torch.max(nonzero[2])  # Corrected: Use nonzero[2] for width (column)

        print(f"bbox top left is ({xmin}, {ymin}) and bottom right is ({xmax}, {ymax})")

        # if necessary, shrink the padding for each side to make sure it doesn't go out of bounds
        top_pad = pixel_padding if ymin - pixel_padding >= 0 else ymin
        left_pad = pixel_padding if xmin - pixel_padding >= 0 else xmin
        bottom_pad = pixel_padding if ymax + pixel_padding <= input_canvas_image.shape[1] else input_canvas_image.shape[1] - ymax
        right_pad = pixel_padding if xmax + pixel_padding <= input_canvas_image.shape[2] else input_canvas_image.shape[2] - xmax

        xmin_padded = xmin - left_pad
        ymin_padded = ymin - top_pad
        xmax_padded = xmax + right_pad
        ymax_padded = ymax + bottom_pad
        
        # extract the selection from the image
        selection = input_canvas_image[0, (ymin-top_pad):(ymax+bottom_pad), (xmin-left_pad):(xmax+right_pad), :] 
        selection_mask = selector_mask[0, (ymin-top_pad):(ymax+bottom_pad), (xmin-left_pad):(xmax+right_pad)]

        # the selection and selection_mask should now have the shapes (H, W, C) and (H, W) respectively
        print(f"selection shape: {selection.shape}, selection_mask shape: {selection_mask.shape}")

        # interpolate function requires input to be (B, C, H, W), so we rearrange the tensors A G A I N
        selection = selection.permute(2, 0, 1)
        selection = selection.unsqueeze(0)
        selection_mask = selection_mask.unsqueeze(0).unsqueeze(0)

        # the selection and selection_mask should now have the shapes (1, C, H, W) and (1, 1, H, W) respectively
        print(f"before interpolation: selection shape: {selection.shape}, selection_mask shape: {selection_mask.shape}")
        

        # scale the selection based on scale_factor
        selection = F.interpolate(selection, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        selection_mask = F.interpolate(selection_mask, scale_factor=scale_factor, mode='nearest')
        print(f"after interpolation: selection shape: {selection.shape}, selection_mask shape: {selection_mask.shape}")

        # finally, you guessed it, shuffle back to (B, H, W, C) for the image and (H, W) for the mask
        selection = selection.permute(0, 2, 3, 1)
        selection_mask = selection_mask.squeeze(0)
        print(f"returning selection shape: {selection.shape}, selection_mask shape: {selection_mask.shape}")
        return (selection,selection_mask,(input_canvas_image, {"top_left": (xmin_padded, ymin_padded), "bottom_right": (xmax_padded, ymax_padded)}))
    
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
        top_left_x, top_left_y = bbox_dict["top_left"]
        bottom_right_x, bottom_right_y = bbox_dict["bottom_right"]

        # define the width and height of the bounding box
        width = bottom_right_x - top_left_x
        height = bottom_right_y - top_left_y

        # define the transform for the resizing of the selection
        resize_transform = transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR)

        # transform the selection shape from (B, H, W, C) to (B, C, H, W)
        canvas_selection = canvas_selection.permute(0, 3, 1, 2)

        # resize the selection
        print(f"resizing selection to {height}x{width}")
        canvas_selection = resize_transform(canvas_selection)

        # transform selection shape back to (B, H, W, C)
        canvas_selection = canvas_selection.permute(0, 2, 3, 1)

        # paste the selection onto the canvas image
        canvas_image[0, top_left_y:bottom_right_y, top_left_x:bottom_right_x, :] = canvas_selection

        return (canvas_image,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {"Example": Example,
                       "CanvasLoader": CanvasLoader,
                       "CanvasCacheUpdater": CanvasCacheUpdater,
                       "CanvasSelector": CanvasSelector,
                       "CanvasMerger": CanvasMerger}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {"Example": "Example Node",
                             "CanvasLoader": "Canvas Loader",
                             "CanvasCacheUpdater": "Canvas Cache Updater",
                             "CanvasSelector": "Canvas Selector",
                             "CanvasMerger": "Canvas Merger"}
