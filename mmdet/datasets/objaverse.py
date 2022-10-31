from mmdet.datasets.lvis import LVISV1Dataset
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.builder import DATASETS

import copy
import glob
import os
import random
from typing import Any, Dict, List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np

# get the shapely polygonn from the mask
import rasterio
import shapely
from matplotlib.patches import Polygon as PolygonPatch
from PIL import Image
from rasterio import features
from shapely.geometry import MultiPolygon, Polygon

RENDERED_OBJECTS_BASE_DIR = "/net/nfs2.prior/jordis/projects/openvocab/data/views"


@DATASETS.register_module()
class ObjaverseDummyDataset(LVISV1Dataset):
    def __init__(self, ignore_passthrough=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_passthrough = ignore_passthrough
        if not self.ignore_passthrough:
            self.category_name_to_id = {}
            for category in self.coco.dataset["categories"]:
                self.category_name_to_id[category["id"]] = category["name"]
                self.category_name_to_id[category["name"]] = category["id"]

    def lvis_annotation(self, idx):
        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return ann_info

    def parse_ann(self, idx, ann_info):
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_ann_info(self, idx):
        if not self.ignore_passthrough:
            return dict(parser=self, idx=idx)
        else:
            ann_info = super().get_ann_info(idx)
            return ann_info


def colormap(rgb=False):
    color_list = np.array(
        [
            0.000,
            0.447,
            0.741,
            0.850,
            0.325,
            0.098,
            0.929,
            0.694,
            0.125,
            0.494,
            0.184,
            0.556,
            0.466,
            0.674,
            0.188,
            0.301,
            0.745,
            0.933,
            0.635,
            0.078,
            0.184,
            0.300,
            0.300,
            0.300,
            0.600,
            0.600,
            0.600,
            1.000,
            0.000,
            0.000,
            1.000,
            0.500,
            0.000,
            0.749,
            0.749,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.333,
            0.333,
            0.000,
            0.333,
            0.667,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            0.333,
            0.000,
            0.667,
            0.667,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            0.333,
            0.500,
            0.000,
            0.667,
            0.500,
            0.000,
            1.000,
            0.500,
            0.333,
            0.000,
            0.500,
            0.333,
            0.333,
            0.500,
            0.333,
            0.667,
            0.500,
            0.333,
            1.000,
            0.500,
            0.667,
            0.000,
            0.500,
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.143,
            0.143,
            0.143,
            0.286,
            0.286,
            0.286,
            0.429,
            0.429,
            0.429,
            0.571,
            0.571,
            0.571,
            0.714,
            0.714,
            0.714,
            0.857,
            0.857,
            0.857,
            1.000,
            1.000,
            1.000,
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list


def clipped_zoom(img, zoom_factor=0):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of
    the image without changing dimensions
    ------
    Args:
        img : ndarray
            Image array
        zoom_factor : float
            amount of zoom as a ratio [0 to Inf). Default 0.
    ------
    Returns:
        result: ndarray
           numpy ndarray of the same shape of the input img zoomed by the specified factor.
    """
    if zoom_factor == 0:
        return img

    height, width = img.shape[:2]  # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1, x1, y2, x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) // 2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (
        width - resize_width
    ) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0, 0)] * (
        img.ndim - 2
    )

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode="constant")
    assert result.shape[0] == height and result.shape[1] == width
    return result


def mask_to_polygons_layer(mask: np.ndarray) -> Union[Polygon, MultiPolygon]:
    all_polygons = []
    for shape, _ in features.shapes(
        mask.astype(np.int16),
        mask=(mask > 0),
        transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0),
    ):
        all_polygons.append(shapely.geometry.shape(shape))

    all_polygons = shapely.geometry.MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == "Polygon":
            all_polygons = shapely.geometry.MultiPolygon([all_polygons])
    return all_polygons


def paste_image(
    background_image: np.ndarray,
    image_to_paste: Image,
    lvis_annotation: List[dict],
    category_id: int,
) -> np.ndarray:
    foreground_image = np.zeros(
        (background_image.shape[0], background_image.shape[1], 4), dtype=np.uint8
    )
    size = min(background_image.shape[0], background_image.shape[1])

    image_to_paste = image_to_paste.resize((size, size))
    image_to_paste_arr = np.array(image_to_paste)

    # center the image to paste on the foreground_image
    foreground_image[
        (foreground_image.shape[0] - image_to_paste_arr.shape[0])
        // 2 : (foreground_image.shape[0] - image_to_paste_arr.shape[0])
        // 2
        + image_to_paste_arr.shape[0],
        (foreground_image.shape[1] - image_to_paste_arr.shape[1])
        // 2 : (foreground_image.shape[1] - image_to_paste_arr.shape[1])
        // 2
        + image_to_paste_arr.shape[1],
    ] = image_to_paste_arr

    # scale the image to paste
    scale = random.uniform(0.2, 1.1)
    scaled_mask = clipped_zoom(foreground_image, scale)

    # translate the image to paste
    translate_x = random.randint(
        -background_image.shape[1] // 2, background_image.shape[1] // 2
    )
    translate_y = random.randint(
        -background_image.shape[0] // 2, background_image.shape[0] // 2
    )
    translation_matrix = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
    img_translation = cv2.warpAffine(
        scaled_mask,
        translation_matrix,
        (background_image.shape[1], background_image.shape[0]),
    )

    # place the foreground image on the background image
    mask = np.array(img_translation)[:, :, 3] != 0
    mask_3d = np.stack([mask, mask, mask], axis=2)
    overlay = np.where(mask_3d, img_translation[:, :, :3], background_image)

    polygons = mask_to_polygons_layer(mask).simplify(1.5)

    annotations_to_delete = []
    for annotation_i, annotation in enumerate(lvis_annotation):
        new_ann_area = 0
        new_segmentations = []
        for seg in annotation["segmentation"]:
            seg_polygon = Polygon(np.array(seg).reshape(-1, 2))
            # check if polygons instersects with seg_polygon
            if not polygons.intersects(seg_polygon):
                new_segmentations.append(seg)
                new_ann_area += seg_polygon.area
                continue
            # subtract polygons from seg_polygon
            new_seg = seg_polygon.difference(polygons)

            # check if new_seg is empty
            if new_seg.is_empty:
                print(new_seg)
                continue

            if isinstance(new_seg, shapely.geometry.MultiPolygon) or isinstance(new_seg, shapely.geometry.GeometryCollection):
                new_area = 0
                new_seg_parts = []
                for polygon in new_seg.geoms:
                    if polygon.area < 10:
                        # don't add very small polygons
                        continue
                    new_area += polygon.area
                    new_seg_parts.append(
                        np.array(polygon.exterior.coords).reshape(-1).tolist()
                    )
                if new_area < 10 or len(new_seg_parts) == 0:
                    # don't add very small polygons
                    continue
            else:
                new_area = new_seg.area
                if new_area < 10:
                    continue
                new_seg_parts = [np.array(new_seg.exterior.coords).reshape(-1).tolist()]

            new_ann_area += new_area
            new_segmentations.extend(new_seg_parts)
        if len(new_segmentations) == 0 or new_ann_area < 10:
            annotations_to_delete.append(annotation_i)
            continue

        lvis_annotation[annotation_i]["segmentation"] = new_segmentations
        lvis_annotation[annotation_i]["area"] = new_ann_area

        # NOTE: Update the bounding box
        min_x = min([min([x for x in seg[::2]]) for seg in new_segmentations])
        min_y = min([min([y for y in seg[1::2]]) for seg in new_segmentations])
        max_x = max([max([x for x in seg[::2]]) for seg in new_segmentations])
        max_y = max([max([y for y in seg[1::2]]) for seg in new_segmentations])
        lvis_annotation[annotation_i]["bbox"] = [
            min_x,
            min_y,
            max_x - min_x,
            max_y - min_y,
        ]

    # NOTE: delete annotations that are too small
    # print(annotations_to_delete)
    for annotation_i in reversed(annotations_to_delete):
        del lvis_annotation[annotation_i]

    segmentation = (
        [
            np.array(polygon.exterior.coords).reshape(-1).tolist()
            for polygon in polygons.geoms
        ]
        if isinstance(polygons, shapely.geometry.MultiPolygon)
        else ([np.array(polygons.exterior.coords).reshape(-1).tolist()])
    )

    min_x = min([x for pair in segmentation for x in pair[::2]])
    min_y = min([y for pair in segmentation for y in pair[1::2]])
    max_x = max([x for pair in segmentation for x in pair[::2]])
    max_y = max([y for pair in segmentation for y in pair[1::2]])

    annotation = {
        "bbox": [min_x, min_y, max_x - min_x, max_y - min_y],
        "category_id": category_id,
        "image_id": lvis_annotation[0]["image_id"],
        "id": len(lvis_annotation) + 1,
        "segmentation": segmentation,
        "area": polygons.area,
    }
    lvis_annotation.append(annotation)

    return overlay


def visualize_segmentation(
    image: np.ndarray, lvis_annotation: List[dict], dpi: int = 72
) -> plt.Figure:
    fig = plt.figure(frameon=False)
    fig.set_size_inches(image.shape[1] / dpi, image.shape[0] / dpi)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.axis("off")
    fig.add_axes(ax)

    ax.imshow(image)
    for i, annotation in enumerate(
        sorted(lvis_annotation, key=lambda x: x["area"], reverse=True)
    ):
        color_list = colormap(rgb=True) / 255
        color = color_list[i % len(color_list), 0:3]
        for seg in annotation["segmentation"]:
            patch = PolygonPatch(
                np.array(seg).reshape(-1, 2),
                fc=color,
                ec=color,
                alpha=0.5,
                linewidth=3,
                fill=True,
            )
            ax.add_patch(patch)
    return fig


@PIPELINES.register_module()
class ObjaverseAugment:
    def __init__(self, ignore_passthrough=False):
        self.category_paths = glob.glob(os.path.join(RENDERED_OBJECTS_BASE_DIR, "*"))
        self.category_instances = {c: glob.glob(os.path.join(c, "*.png")) for c in self.category_paths}
        self.ignore_passthrough = ignore_passthrough
        self.usable_categories = None
        # self.num_samples = 0

    def sample_image_to_paste(self) -> Tuple[Any, str]:
        # NOTE: randomly choose a category
        category, category_path = None, None
        while category not in self.usable_categories:
            category_path = random.choice(self.category_paths)
            category = os.path.basename(category_path)

        # NOTE: randomly choose an image from the category
        category_instances = self.category_instances[category_path]
        image_to_paste_path = random.choice(category_instances)

        return image_to_paste_path, category

    def add_objects_to_background(
            self,
            background_image: np.ndarray,
            lvis_annotation: List[dict],
            num_objects_to_add: int,
            category_name_to_id: Dict[str, int],
    ) -> np.ndarray:
        for _ in range(num_objects_to_add):
            image_to_paste_path, category = self.sample_image_to_paste()
            image_to_paste = Image.open(image_to_paste_path)

            background_image = paste_image(
                background_image=background_image,
                image_to_paste=image_to_paste,
                lvis_annotation=lvis_annotation,
                category_id=category_name_to_id[category],
            )
        return background_image

    def __call__(self, results):
        if not self.ignore_passthrough:
            background_image = results["img"]
            idx = results["ann_info"]["idx"]
            parser = results["ann_info"]["parser"]

            if self.usable_categories is None:
                self.usable_categories = set(c["name"] for c in parser.coco.dataset["categories"])

            lvis_ann = parser.lvis_annotation(idx)
            # print("before ann boxes", parser.parse_ann(idx, lvis_ann)["bboxes"])  # TODO Only for debug

            try:
                if random.random() > 0.5:
                    num_objects_to_add = random.randint(0, 3)
                    if num_objects_to_add > 0:
                        tmp_ann = copy.deepcopy(lvis_ann)
                        new_img = self.add_objects_to_background(background_image, tmp_ann, num_objects_to_add, parser.category_name_to_id)
                        results["img"] = new_img
                        lvis_ann = tmp_ann
            except:
                print("Failed augment")

            results["ann_info"] = parser.parse_ann(idx, lvis_ann)
            # print("after ann boxes", results["ann_info"]["bboxes"])  # TODO Only for debug

            # if new_img is not None:
            #     cv2.imwrite(f"debug_{self.num_samples}_before.jpg", background_image)
            #     cv2.imwrite(f"debug_{self.num_samples}_after.jpg", new_img)
            #     fig = visualize_segmentation(new_img, lvis_ann)
            #     plt.savefig(f"debug_{self.num_samples}_after_segm.jpg", bbox_inches='tight')
            #     plt.close(fig)
            #     self.num_samples += 1

        return results
