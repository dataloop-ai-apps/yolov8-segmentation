import dtlpy as dl
import json
import logging
import os
import torch
import PIL

from PIL import Image
from ultralytics import YOLO
from ultralytics.yolo.utils import yaml_save

logger = logging.getLogger('YOLOv8SegmentationAdapter')

# set max image size
PIL.Image.MAX_IMAGE_PIXELS = 933120000


@dl.Package.decorators.module(description='Model Adapter for Yolov8 object segmentation',
                              name='model-adapter',
                              init_inputs={'model_entity': dl.Model})
class Adapter(dl.BaseModelAdapter):
    def save(self, local_path, **kwargs):
        self.model_entity.artifacts.upload(os.path.join(local_path, '*'))
        self.configuration.update({'weights_filename': 'weights/best.pt'})

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
            filters.add_join(field='type', values=['segment', 'polygon'], operator=dl.FILTERS_OPERATIONS_IN)
            filters.page_size = 0
            pages = self.model_entity.dataset.items.list(filters=filters)
            if pages.items_count == 0:
                raise ValueError(
                    f'Could not find segment annotations in subset {subset}. '
                    f'Cannot train without annotation in the data subsets')

        #########
        # Paths #
        #########
        train_dir = self.model_entity.metadata['system']['subsets']['train']['filter']['$and'][-1]['dir'][1:]
        val_dir = self.model_entity.metadata['system']['subsets']['validation']['filter']['$and'][-1]['dir'][1:]
        train_path = os.path.join(data_path, 'train', 'json', train_dir)
        validation_path = os.path.join(data_path, 'validation', 'json', val_dir)

        ###########
        # Convert #
        ###########
        for src_path in [train_path, validation_path]:
            labels_path = os.path.join(data_path, 'train' if src_path == train_path else 'validation', 'json', 'labels')
            os.makedirs(labels_path, exist_ok=True)
            for json_file_path in os.listdir(src_path):
                with open(os.path.join(src_path, json_file_path), 'r') as json_file:
                    item_info = json.load(json_file)
                    item_width = item_info.get('metadata', {}).get('system', {}).get('width', 0)
                    item_height = item_info.get('metadata', {}).get('system', {}).get('height', 0)
                    annotations = list()
                    for ann in item_info.get("annotations", []):
                        valid = True
                        annotation_line = [self.model_entity.label_to_id_map.get(ann.get("label"))]
                        for coordinates in ann.get("coordinates", []):
                            if isinstance(coordinates, list) and len(coordinates) > 0:
                                for coordinate in coordinates:
                                    annotation_line.append(coordinate['x'] / item_width)
                                    annotation_line.append(coordinate['y'] / item_height)
                            else:
                                logger.error(f"Coordinates of invalid type ({type(coordinates)}) "
                                             f"or length ({len(coordinates) if isinstance(coordinates, list) else 'nan'}"
                                             f")")
                                valid = False
                                break
                        if valid is True:
                            annotations.append(' '.join([str(el) for el in annotation_line]))
                labels_file_path = os.path.join(labels_path, os.path.splitext(json_file_path)[0] + '.txt')
                with open(labels_file_path, 'w') as labels_file:
                    labels_file.write('\n'.join(annotations))

    def load(self, local_path, **kwargs):
        model_filename = self.configuration.get('weights_filename', 'yolov8l-seg.pt')
        model_filepath = os.path.join(local_path, model_filename)
        # first load official model -https://github.com/ultralytics/ultralytics/issues/3856
        _ = YOLO('yolov8l-seg.pt')
        if os.path.isfile(model_filepath):
            model = YOLO(model_filepath) 
        else:
            logger.warning(f'Model path ({model_filepath}) not found! loading default model weights')
            model = YOLO('yolov8l-seg.pt') 
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
        train_dir = self.model_entity.metadata['system']['subsets']['train']['filter']['$and'][-1]['dir'][1:]
        val_dir = self.model_entity.metadata['system']['subsets']['validation']['filter']['$and'][-1]['dir'][1:]


        # https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#13-organize-directories
        src_images_path_train = os.path.join(data_path, 'train', 'items', train_dir)
        dst_images_path_train = os.path.join(data_path, 'train', 'images')
        src_images_path_val = os.path.join(data_path, 'validation', 'items', val_dir)
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
                    print(new_subfolder)
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
        results = self.model.predict(source=batch, save=False, save_txt=False)  # save predictions as labels
        batch_annotations = list()
        for i_img, res in enumerate(results):  # per image
            if res.masks:
                image_annotations = dl.AnnotationCollection()
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
