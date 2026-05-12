import nibabel as nib
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import random

ORGAN_PRIORS = {
    1: {"name": "liver", "metal_prob": 0.1, "shape": "small_blob"},
    2: {"name": "bladder", "metal_prob": 0.3, "shape": "irregular"},
    4: {"name": "kidney", "metal_prob": 0.6, "shape": "stone"},
    5: {"name": "bone", "metal_prob": 0.9, "shape": "implant"},
}

def load_nii_gz(path):
    """
    Load a .nii or .nii.gz file.

    Returns:
        data (np.ndarray): image volume (float64 by default)
        affine (np.ndarray): 4x4 affine matrix (voxel -> world)
        header (nib.Nifti1Header): metadata
    """
    img = nib.load(path)
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    return data, affine, header


def generate_shape(mask, shape_type):
    """
    Generate a synthetic metal-like region inside the mask.
    """
    h, w = mask.shape
    result = np.zeros((h, w), dtype=np.uint8)

    # Get coordinates inside organ
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return result

    # Pick random center inside organ
    idx = random.randint(0, len(xs) - 1)
    cx, cy = xs[idx], ys[idx]

    if shape_type == "stone":
        radius = random.randint(3, 8)
        cv2.circle(result, (cx, cy), radius, 1, -1)

    elif shape_type == "implant":
        length = random.randint(10, 30)
        angle = random.uniform(0, np.pi)
        x2 = int(cx + length * np.cos(angle))
        y2 = int(cy + length * np.sin(angle))
        cv2.line(result, (cx, cy), (x2, y2), 1, thickness=3)

    elif shape_type == "small_blob":
        radius = random.randint(2, 5)
        cv2.circle(result, (cx, cy), radius, 1, -1)

    elif shape_type == "irregular":
        for _ in range(5):
            rx = cx + random.randint(-10, 10)
            ry = cy + random.randint(-10, 10)
            cv2.circle(result, (rx, ry), random.randint(2, 5), 1, -1)

    # Ensure it's inside the organ
    return result * mask

def generate_guidance(segmentation, priors):
    guidance = np.zeros_like(segmentation, dtype=np.uint8)

    for z in range(segmentation.shape[2]):
        slice_seg = segmentation[:, :, z]
        for label_id, info in priors.items():
            organ_mask = (slice_seg == label_id)

            if organ_mask.sum() == 0:
                continue

            if np.random.rand() < info["metal_prob"]:
                shape_mask = generate_shape(organ_mask, info["shape"])
                guidance[:, :, z] = np.maximum(guidance[:, :, z], shape_mask)

    return guidance

def save_png_slice(volume, filepath, slice_index=None, cmap="gray", title=None):
    if volume.ndim != 3:
        raise ValueError("Expected a 3D volume")
    if slice_index is None:
        slice_index = volume.shape[2] // 2
    slice_img = volume[:, :, slice_index]

    plt.figure(figsize=(8, 8))
    plt.imshow(slice_img, cmap=cmap)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.savefig(filepath, bbox_inches="tight", pad_inches=0)
    plt.close()

def save_overlay_slice(segmentation_slice, guidance_slice, filepath, title=None):
    plt.figure(figsize=(8, 8))
    plt.imshow(segmentation_slice, cmap="tab20")
    plt.imshow(guidance_slice, cmap="Reds", alpha=0.5)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.savefig(filepath, bbox_inches="tight", pad_inches=0)
    plt.close()


if __name__ == "__main__":
    nii_path = "nnUNet_output/nnunet/infer_out/body6_img5001.nii.gz"
    data, affine, header = load_nii_gz(nii_path)

    segmentation = data.astype(np.uint8)
    guidance = generate_guidance(segmentation, ORGAN_PRIORS)

    save_png_slice(segmentation, "output/maggnet/guidance_on_segmentation/segmentation_mid_slice.png", cmap="tab20", title="Segmentation")
    save_png_slice(guidance, "output/maggnet/guidance_on_segmentation/guidance_mid_slice.png", cmap="gray", title="Guidance")
    save_overlay_slice(segmentation[:, :, segmentation.shape[2] // 2], guidance[:, :, guidance.shape[2] // 2], "output/maggnet/guidance_on_segmentation/overlay_mid_slice.png", title="Overlay")

    print("Shape:", data.shape)
    print("Dtype:", data.dtype)
    print("Voxel spacing:", header.get_zooms())
    print("Affine:\n", affine)
    print("Saved segmentation slice to output/maggnet/guidance_on_segmentation/segmentation_mid_slice.png")
    print("Saved guidance slice to output/maggnet/guidance_on_segmentation/guidance_mid_slice.png")
