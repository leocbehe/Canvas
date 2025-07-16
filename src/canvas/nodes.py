from inspect import cleandoc
import torch
import torchvision
from PIL import Image
import os
import numpy as np

import torchvision.transforms.functional

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
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "use_existing": (
                    "BOOLEAN",
                    {
                        "tooltip": "Use an existing image",
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
                "fill_img_with": (["white", "black", "random"],),
            }
        }

    
    def execute(self, use_existing, image_width, image_height, fill_img_with):
        print(f"execute canvas loader")

        if not os.path.exists(CACHE_PATH):
            os.makedirs(CACHE_PATH)
        cached_image_path = os.path.join(CACHE_PATH, f"{PREV_GENERATED_IMAGE}_1.png")

        if use_existing:
            if os.path.exists(cached_image_path):
                image = torchvision.io.decode_image(cached_image_path)
        else:
            image = torch.Tensor(torch.empty((1, image_height, image_width, 3), dtype=torch.float16))

        return (image,)
    
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

        if not caching_enabled:
            return

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


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {"Example": Example,
                       "CanvasLoader": CanvasLoader,
                       "CanvasCacheUpdater": CanvasCacheUpdater}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {"Example": "Example Node",
                             "CanvasLoader": "Canvas Loader",
                             "CanvasCacheUpdater": "Canvas Cache Updater"}
