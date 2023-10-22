import json

from ultralytics.models import YOLO
from ultralytics.utils import yaml_save
import dtlpy as dl
from PIL import Image
import cv2
import numpy as np
import tqdm
import logging
import os

logger = logging.getLogger('YOLOv8Adapter')


@dl.Package.decorators.module(description='Model Adapter for Yolov8 object detection',
                              name='model-adapter',
                              init_inputs={'model_entity': dl.Model})
class Adapter(dl.BaseModelAdapter):
    def save(self, local_path, **kwargs):
        self.model_entity.artifacts.upload(os.path.join(local_path, '*'))
        self.configuration.update({'weights_filename': 'weights/best.pt'})

    def convert_from_dtlpy(self, data_path, **kwargs):
        train_dir = self.model_entity.metadata['system']['subsets']['train']['filter']['$and'][-1]['dir'][1:]
        val_dir = self.model_entity.metadata['system']['subsets']['validation']['filter']['$and'][-1]['dir'][1:]
        train_path = os.path.join(data_path, 'train', 'json', train_dir)
        validation_path = os.path.join(data_path, 'validation', 'json', val_dir)

        for src_path in [train_path, validation_path]:
            labels_path = os.path.join(os.path.split(src_path)[0], 'labels')
            os.makedirs(labels_path, exist_ok=True)
            for json_file_path in os.listdir(src_path):
                with open(os.path.join(src_path, json_file_path), 'r') as json_file:
                    item_info = json.load(json_file)
                    item_width = item_info.get('metadata', {}).get('system', {}).get('width', 0)
                    item_height = item_info.get('metadata', {}).get('system', {}).get('height', 0)
                    annotations = list()
                    for ann in item_info.get("annotations", []):
                        valid = True
                        annotation_line = [self.configuration.get("label_to_id_map", {}).get(ann.get("label"))]
                        for coordinates in ann.get("coordinates", []):
                            if isinstance(coordinates, list):
                                for coordinate in coordinates:
                                    annotation_line.append(coordinate['x'] / item_width)
                                    annotation_line.append(coordinate['y'] / item_height)
                            elif isinstance(coordinates, dict) and 'x' in coordinates.keys():
                                valid = False
                                break
                        if valid:
                            annotations.append(' '.join([str(el) for el in annotation_line]))
                labels_file_path = os.path.join(labels_path, os.path.splitext(json_file_path)[0] + '.txt')
                with open(labels_file_path, 'w') as labels_file:
                    labels_file.write('\n'.join(annotations))



    def load(self, local_path, **kwargs):
        model_filename = self.configuration.get('weights_filename', 'yolov8m-seg.pt')
        model_filepath = os.path.join(local_path, model_filename)
        # first load official model -https://github.com/ultralytics/ultralytics/issues/3856
        _ = YOLO('yolov8m-seg.pt')
        if os.path.isfile(model_filepath):
            model = YOLO(model_filepath)  # pass any model type
        else:
            logger.warning(f'Model path ({model_filepath}) not found! loading default model weights')
            model = YOLO(model_filename)  # pass any model type
        self.model = model

    def train(self, data_path, output_path, **kwargs):
        self.model.model.args.update(self.configuration.get('modelArgs', dict()))
        epochs = self.configuration.get('epochs', 50)
        batch_size = self.configuration.get('batch_size', 2)
        imgsz = self.configuration.get('imgsz', 640)
        device = self.configuration.get('device', 'cpu')
        augment = self.configuration.get('augment', False)
        yaml_config = self.configuration.get('yaml_config', dict())

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

        # validation_images_path
        # train_lables_path
        # train_images_path

        yaml_config.update(
            {'path': os.path.realpath(data_path),  # must be full path otherwise the train adds "datasets" to it
             'train': 'train/images',
             'val': 'validation/images',
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

        # self.model.add_callback(event='on_train_epoch_end', func=on_train_epoch_end)
        # self.model.add_callback(event='on_val_end', func=on_train_epoch_end)
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
                    image_annotations.add(annotation_definition=dl.Polygon(geo=mask.xy[0], label=label),
                                          model_info={'name': self.model_entity.name,
                                                      'model_id': self.model_entity.id,
                                                      'confidence': float(conf)})
                batch_annotations.append(image_annotations)
        return batch_annotations

    def export(self):
        model = YOLO("model.pt")
        model.fuse()
        model.info(verbose=True)  # Print model information
        model.export(format='')


def package_creation(project: dl.Project):
    metadata = dl.Package.get_ml_metadata(cls=Adapter,
                                          default_configuration={'weights_filename': 'yolov8l-seg.pt',
                                                                 'epochs': 25,
                                                                 'batch_size': 8,
                                                                 'imgsz': 960,
                                                                 'conf_thres': 0.25,
                                                                 'iou_thres': 0.45,
                                                                 'max_det': 1000,
                                                                 'augment': False,
                                                                 'device': 'cuda:0'},
                                          output_type=dl.AnnotationType.SEGMENTATION,
                                          )
    modules = dl.PackageModule.from_entry_point(entry_point='model_adapter.py')

    package = project.packages.push(package_name='yolov8-seg',
                                    src_path=os.getcwd(),
                                    # description='Global Dataloop Yolo V8 implementation in pytorch',
                                    is_global=False,
                                    package_type='ml',
                                    # codebase=dl.GitCodebase(git_url='https://github.com/dataloop-ai-apps/yolov8.git',
                                    #                         git_tag='v0.1.13'),
                                    modules=[modules],
                                    service_config={
                                        'runtime': dl.KubernetesRuntime(pod_type=dl.INSTANCE_CATALOG_GPU_K80_M,
                                                                        runner_image='ultralytics/ultralytics:8.0.183',
                                                                        autoscaler=dl.KubernetesRabbitmqAutoscaler(
                                                                            min_replicas=0,
                                                                            max_replicas=1),
                                                                        preemptible=False,
                                                                        concurrency=1).to_json(),
                                        'executionTimeout': 10000 * 3600,
                                        'initParams': {'model_entity': None}
                                    },
                                    metadata=metadata)
    return package


def model_creation(package: dl.Package, dataset: dl.Dataset):
    import ultralytics
    labels = ultralytics.YOLO().names
    labels = {i:l.tag for i,l in enumerate(dataset.labels)}

    model = package.models.create(model_name='yolov8large-seg',
                                  description='yolov8 for image segmentation',
                                  tags=['yolov8', 'pretrained', 'segmentation'],
                                  dataset_id=dataset.id,
                                  status='created',
                                  scope='project',
                                  configuration={
                                      'weights_filename': 'yolov8l-seg.pt',
                                      'imgz': 960,
                                      'device': 'cuda:0',
                                      'id_to_label_map': labels,
                                      'label_to_id_map': {v: k for k, v in labels.items()}
                                  },
                                  project_id=package.project.id,
                                  labels=list(labels.values()),
                                  input_type='image',
                                  output_type='segment'
                                  )
    return model


def deploy():
    dl.setenv('prod')
    project_name = '<insert-project-name>'
    project = dl.projects.get(project_name)
    dataset = project.datasets.get("<insert-dataset-name>")
    package = package_creation(project=project)
    print(f'new mode pushed. codebase: {package.codebase}')
    model = model_creation(package=package, dataset=dataset)
    return model


if __name__ == "__main__":
    deploy()
