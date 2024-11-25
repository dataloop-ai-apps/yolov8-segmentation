import dtlpy as dl
import json
import logging
import os
import torch
import PIL
import base64
import shutil
import concurrent.futures
import numpy as np

from pathlib import Path
from PIL import Image
from skimage import measure
from io import BytesIO
from ultralytics import YOLO
from ultralytics.yolo.utils import yaml_save
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger('YOLOv8SegmentationAdapter')

# set max image size
PIL.Image.MAX_IMAGE_PIXELS = 933120000

DEFAULT_WEIGHT = 'yolov8l-seg.pt'


@dl.Package.decorators.module(description='Model Adapter for Yolov8 object segmentation',
                              name='model-adapter',
                              init_inputs={'model_entity': dl.Model})
class Adapter(dl.BaseModelAdapter):
    def save(self, local_path, **kwargs):
        self.model_entity.artifacts.upload(os.path.join(local_path, '*'))
        self.configuration.update({'weights_filename': 'weights/best.pt'})

    @staticmethod
    def move_annotation_files(data_path):
        logger.debug(f"Data path: {data_path}")
        path = Path(data_path)
        json_files = (path / 'json').rglob("*.json")
        logger.debug(f"Json files: {json_files}")
        img_extensions = ["jpg", "jpeg", "png", "bmp"]
        item_files = []
        for ext in img_extensions:
            item_files += (path / 'items').rglob(f"*.{ext}")
        for src, dst in zip([json_files, item_files], ['json', 'items']):
            for src_file in src:
                if not os.path.exists(os.path.join(data_path, dst, os.path.basename(src_file))):
                    shutil.move(src_file, os.path.join(data_path, dst, os.path.basename(src_file)))
        for root, dirs, files in os.walk(data_path, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)

    def process_annotation_json_file(self, src_path: str, json_file_path: str, labels_path: str):
        if os.path.isfile(os.path.join(src_path, json_file_path)):
            logger.info(f"Processing {os.path.join(src_path, json_file_path)}")
            with open(os.path.join(src_path, json_file_path), 'r') as json_file:
                item_info = json.load(json_file)
                item_width = item_info.get('metadata', {}).get('system', {}).get('width', 0)
                item_height = item_info.get('metadata', {}).get('system', {}).get('height', 0)
                logger.info(
                    f"Item size for file at {os.path.join(src_path, json_file_path)}: {item_width} x {item_height}")
                annotations = list()
                logger.info(
                    f"Item at {os.path.join(src_path, json_file_path)} contains {len(item_info.get('annotations', []))} "
                    f"annotations")
                annotation_lines = []
                for n, ann in enumerate(item_info.get("annotations", [])):
                    valid = True
                    if ann["type"] not in [dl.AnnotationType.SEGMENTATION, dl.AnnotationType.POLYGON]:
                        logger.debug(f"Annotation {n} of item @ {os.path.join(src_path, json_file_path)} was ignored "
                                     f"because it's of type {ann['type']}")
                        continue
                    coordinates = ann.get("coordinates")
                    if isinstance(coordinates, list) and len(coordinates) > 0:
                        logger.debug(
                            f"Annotation {n} of item @ {os.path.join(src_path, json_file_path)} has a list of "
                            f"coordinates")
                        annotation_lines.append([self.model_entity.label_to_id_map.get(ann.get("label"))])
                        coordinates = coordinates[0]
                        for coordinate in coordinates:
                            annotation_lines[-1].append(coordinate['x'] / item_width)
                            annotation_lines[-1].append(coordinate['y'] / item_height)

                    elif isinstance(coordinates, str):
                        logger.info(
                            f"Annotation {n} of item @ {os.path.join(src_path, json_file_path)} is a string encoding a "
                            f"map")
                        encoded_mask = coordinates.split(",")[1]
                        decoded_mask = base64.b64decode(encoded_mask)
                        image_mask = Image.open(BytesIO(decoded_mask))
                        mask_array = np.array(image_mask)
                        logger.info(
                            f"Annotation {n} of item @ {os.path.join(src_path, json_file_path)} was successfully decoded"
                            f" as a map of dimension {mask_array.shape}")
                        mask = np.sum([mask_array[:, :, -x] for x in range(1, 4)], axis=0)
                        logger.info(
                            f"Annotation {n} of item @ {os.path.join(src_path, json_file_path)} converted to mask with "
                            f"1 channel")
                        contours = measure.find_contours(mask, 128)
                        logger.info(
                            f"Annotation {n} of item @ {os.path.join(src_path, json_file_path)} obtained contours of "
                            f"{len(contours)} objects")
                        for i, contour in enumerate(contours):
                            logger.debug(
                                f"Processing contour {i} of annotation {n} of item @ "
                                f"{os.path.join(src_path, json_file_path)}")
                            annotation_lines.append([self.model_entity.label_to_id_map.get(ann.get("label"))])
                            for obj in contour:
                                annotation_lines[i].append(obj[0] / item_height)
                                annotation_lines[i].append(obj[1] / item_width)
                            logger.debug(
                                f"Contour {i} of annotation {n} of item @ {os.path.join(src_path, json_file_path)} "
                                f"generated.")
                        logger.debug(
                            f"Annotation {n} of item @ {os.path.join(src_path, json_file_path)} generated.")
                    else:
                        logger.error(
                            f"Coordinates of invalid type ({type(coordinates)}) "
                            f"or length ({len(coordinates) if isinstance(coordinates, list) else 'nan'})"
                        )
                        valid = False
                        break
                if valid is True:
                    for annotation_line in annotation_lines:
                        annotations.append(' '.join([str(el) for el in annotation_line]))
            labels_file_path = os.path.join(labels_path, os.path.splitext(json_file_path)[0] + '.txt')
            logger.info(f"Saved annotations of item @ {os.path.join(src_path, json_file_path)} in {labels_file_path}")
            return labels_file_path, annotations
        else:
            logger.warn(f"Annotation json file {os.path.join(src_path, json_file_path)} was not found")
            return None, None

    def convert_from_dtlpy(self, data_path, **kwargs):
        ##############
        # Validation #
        ##############

        subsets = self.model_entity.metadata.get("system", dict()).get("subsets", None)
        if 'train' not in subsets:
            raise ValueError(
                'Couldnt find train set. Yolov8 requires train and validation set for training. '
                'Add a train set DQL filter in the dl.Model metadata'
            )
        if 'validation' not in subsets:
            raise ValueError(
                'Couldnt find validation set. Yolov8 requires train and validation set for training. '
                'Add a validation set DQL filter in the dl.Model metadata'
            )

        for subset, filters_dict in subsets.items():
            filters = dl.Filters(custom_filter=filters_dict)
            filters.add_join('type',
                             [dl.AnnotationType.POLYGON, dl.AnnotationType.SEGMENTATION],
                             operator=dl.FILTERS_OPERATIONS_IN)
            filters.page_size = 0
            pages = self.model_entity.dataset.items.list(filters=filters)
            if pages.items_count == 0:
                raise ValueError(
                    f'Could not find segment annotations in subset {subset}. '
                    f'Cannot train without annotation in the data subsets')

        #########
        # Paths #
        #########
        self.move_annotation_files(os.path.join(data_path, 'train'))
        self.move_annotation_files(os.path.join(data_path, 'validation'))
        train_path = os.path.join(data_path, 'train', 'json')
        validation_path = os.path.join(data_path, 'validation', 'json')
        logger.info(f"Path for training: {train_path} -- path for validation: {validation_path}")

        ###########
        # Convert #
        ###########
        logger.info("Converting dtlp annotations to YOLOv8 format")
        for src_path in [train_path, validation_path]:
            labels_path = os.path.join(data_path, 'train' if src_path == train_path else 'validation', 'json', 'labels')
            os.makedirs(labels_path, exist_ok=True)
            logger.info(f"Creating label path: {labels_path}")
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.process_annotation_json_file, src_path, json_file_path, labels_path)
                           for json_file_path in os.listdir(src_path)]
                completed_futures, _ = concurrent.futures.wait(futures)
            results = [future.result() for future in completed_futures]
            for annotation_path, annotations in results:
                if annotation_path is not None:
                    logger.info(f"Saving annotations at: {annotation_path}")
                    with open(annotation_path, 'w') as annotation_file:
                        annotation_file.write('\n'.join(annotations))

    @staticmethod
    def get_default_weights():
        # Expected to be at /tmp/app/weights/
        default_weights_path = os.path.join('tmp/app/weights', DEFAULT_WEIGHT)
        if not os.path.isfile(default_weights_path):
            logger.warning(f"Default weights file not found at {default_weights_path}. Using default model.")
            default_weights_path = model_filename
        return default_weights_path

    def load(self, local_path, **kwargs):
        model_filename = self.configuration.get('weights_filename', DEFAULT_WEIGHT)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model_filepath = os.path.normpath(os.path.join(local_path, model_filename))

        default_weights_path = self.get_default_weights()
        # Always loading default weights https://github.com/ultralytics/ultralytics/issues/3856
        model = YOLO(default_weights_path)
        if os.path.isfile(model_filepath):
            logger.info(f"Custom weights found, loading from {model_filepath}")
            model = YOLO(model_filepath)
        model.to(device=device)
        logger.info(f"Model loaded successfully, Device: {model.device}")
        self.model = model

    def train(self, data_path, output_path, **kwargs):
        self.model.model.args.update(self.configuration.get('modelArgs', dict()))
        epochs = self.configuration.get('epochs', 50)
        batch_size = self.configuration.get('batch_size', 2)
        imgsz = self.configuration.get('imgsz', 640)
        device = self.configuration.get('device', None)
        augment = self.configuration.get('augment', False)
        yaml_config = self.configuration.get('yaml_config', dict())

        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        project_name = os.path.dirname(output_path)
        name = os.path.basename(output_path)

        # https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#13-organize-directories
        src_images_path_train = os.path.join(data_path, 'train', 'items')
        dst_images_path_train = os.path.join(data_path, 'train', 'images')
        src_images_path_val = os.path.join(data_path, 'validation', 'items')
        dst_images_path_val = os.path.join(data_path, 'validation', 'images')
        src_labels_path_train = os.path.join(data_path, 'train', 'json', 'labels')
        dst_labels_path_train = os.path.join(data_path, 'train', 'labels')
        src_labels_path_val = os.path.join(data_path, 'validation', 'json', 'labels')
        dst_labels_path_val = os.path.join(data_path, 'validation', 'labels')

        if not os.path.exists(dst_images_path_train) and os.path.exists(src_images_path_train):
            os.rename(src_images_path_train, dst_images_path_train)
        if not os.path.exists(dst_images_path_val) and os.path.exists(src_images_path_val):
            os.rename(src_images_path_val, dst_images_path_val)
        if not os.path.exists(dst_labels_path_train) and os.path.exists(src_labels_path_train):
            os.rename(src_labels_path_train, dst_labels_path_train)
        if not os.path.exists(dst_labels_path_val) and os.path.exists(src_labels_path_val):
            os.rename(src_labels_path_val, dst_labels_path_val)

        # yolov8 bug - if there are two directories "images" in the path it fails to get annotations
        paths = [dst_images_path_train, dst_images_path_val, dst_labels_path_train, dst_labels_path_val]
        allowed = [1, 1, 0, 0]
        for path, allow in zip(paths, allowed):
            subfolders = [x[0] for x in os.walk(path)]
            for subfolder in subfolders:
                relpath = os.path.relpath(subfolder, data_path)
                dirs = relpath.split(os.sep)
                c = 0
                for i_dir, dirname in enumerate(dirs):
                    if dirname == 'images':
                        c += 1
                        if c > allow:
                            dirs[i_dir] = 'imagesssss'
                new_subfolder = os.path.join(data_path, *dirs)
                if subfolder != new_subfolder:
                    logger.debug(new_subfolder)
                    os.rename(subfolder, new_subfolder)

        # check if validation exists
        if not os.path.isdir(dst_images_path_val):
            raise ValueError(
                'Couldnt find validation set. Yolov8 requires train and validation set for training. '
                'Add a validation set DQL filter in the dl.Model metadata'
            )
        if len(self.model_entity.labels) == 0:
            raise ValueError(
                'model.labels is empty. Model entity must have labels'
            )

        logger.debug(f"Train dir: {os.listdir(os.path.join(data_path, 'train'))}")
        logger.debug(f"Train dir: {os.listdir(os.path.join(data_path, 'train', 'images'))}")
        logger.debug(f"Train dir: {os.listdir(os.path.join(data_path, 'train', 'labels'))}")
        logger.debug(f"Validation dir:{os.listdir(os.path.join(data_path, 'validation'))}")
        logger.debug(f"Validation dir:{os.listdir(os.path.join(data_path, 'validation', 'images'))}")
        logger.debug(f"Validation dir:{os.listdir(os.path.join(data_path, 'validation', 'labels'))}")

        yaml_config.update(
            {'path': os.path.realpath(data_path),  # must be full path otherwise the train adds "datasets" to it
             'train': 'train',
             'val': 'validation',
             'names': self.model_entity.labels
             })
        data_yaml_filename = os.path.join(data_path, f'{self.model_entity.dataset_id}.yaml')
        yaml_save(file=data_yaml_filename, data=yaml_config)
        faas_callback = kwargs.get('on_epoch_end_callback')

        def on_epoch_end(train_obj):

            self.current_epoch = train_obj.epoch
            metrics = train_obj.metrics
            train_obj.plot_metrics()
            if faas_callback is not None:
                faas_callback(self.current_epoch, epochs)
            samples = list()
            for metric_name, value in metrics.items():
                legend, figure = metric_name.split('/')
                samples.append(dl.PlotSample(figure=figure,
                                             legend=legend,
                                             x=self.current_epoch,
                                             y=value))
            self.model_entity.metrics.create(samples=samples, dataset_id=self.model_entity.dataset_id)

        self.model.add_callback(event='on_fit_epoch_end', func=on_epoch_end)
        self.model.train(data=data_yaml_filename,
                         exist_ok=True,  # this will override the output dir and will not create a new one
                         epochs=epochs,
                         batch=batch_size,
                         device=device,
                         augment=augment,
                         name=name,
                         workers=0,
                         imgsz=imgsz,
                         project=project_name)

    def prepare_item_func(self, item):
        filename = item.download(overwrite=True)
        image = Image.open(filename)
        # Check if the image has EXIF data
        if hasattr(image, '_getexif'):
            exif_data = image._getexif()
            # Get the EXIF orientation tag (if available)
            if exif_data is not None:
                orientation = exif_data.get(0x0112)
                if orientation is not None:
                    # Rotate the image based on the orientation tag
                    if orientation == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation == 6:
                        image = image.rotate(270, expand=True)
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)
        image = image.convert('RGB')
        return image

    def predict(self, batch, **kwargs):
        inference_args = self.configuration.get("inference_args", {})
        inference_conf = inference_args.get("conf", 0.05)
        inference_iou = inference_args.get("iou", 0.7)
        inference_imgsz = inference_args.get("imgsz", 640)
        inference_precision = inference_args.get("half", False)
        inference_device = inference_args.get("device", None)
        inference_max_det = inference_args.get("max_det", 300)
        inference_augment = inference_args.get("augment", False)
        inference_agnostic_nms = inference_args.get("agnostic_nms", False)
        inference_classes = inference_args.get("classes", None)
        inference_retina_masks = inference_args.get("retina_masks", False)
        if inference_device is None:
            inference_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        results = self.model.predict(source=batch,
                                     save=False,
                                     save_txt=False,  # save predictions as labels
                                     conf=inference_conf,
                                     iou=inference_iou,
                                     imgsz=inference_imgsz,
                                     half=inference_precision,
                                     device=inference_device,
                                     max_det=inference_max_det,
                                     augment=inference_augment,
                                     agnostic_nms=inference_agnostic_nms,
                                     classes=inference_classes,
                                     retina_masks=inference_retina_masks
                                     )
        batch_annotations = list()
        for i_img, res in enumerate(results):  # per image
            image_annotations = dl.AnnotationCollection()
            if res.masks:

                for box, mask in zip(reversed(res.boxes), reversed(res.masks)):
                    cls, conf = box.cls.squeeze(), box.conf.squeeze()
                    c = int(cls)
                    label = res.names[c]
                    if label not in list(self.configuration.get("label_to_id_map", {}).keys()):
                        logger.error(f"Predict label {label} is not among the models' labels.")
                    image_annotations.add(annotation_definition=dl.Polygon(geo=mask.xy[0], label=label),
                                          model_info={'name': self.model_entity.name,
                                                      'model_id': self.model_entity.id,
                                                      'confidence': float(conf)})
            batch_annotations.append(image_annotations)
        return batch_annotations
