from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")

# default one
default = True
if default:
    mask_generator = SamAutomaticMaskGenerator(sam)
else:
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=128,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=3,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )

image = cv2.imread('000005.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

masks = mask_generator.generate(image)
