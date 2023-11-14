import dtlpy as dl
import os
import argparse

from model_adapter import Adapter


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
                                    is_global=False,
                                    package_type='ml',
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
    labels = {i: l.tag for i, l in enumerate(dataset.labels)}

    model = package.models.create(
        model_name='yolov8large-seg',
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


def parse_args():
    arg_parser= argparse.ArgumentParser()
    arg_parser.add_argument("--env", "-e", type=str, help="Environment (prod or rc)")
    arg_parser.add_argument("--project", "-p", type=str, help="Project name")
    arg_parser.add_argument("--dataset", "-d", type=str, help="Dataset name")
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    env = args.env
    project_name = args.project
    dataset_name = args.dataset
    dl.setenv(env)
    project = dl.projects.get(project_name)
    package = package_creation(project)
    dataset = project.datasets.get(dataset_name)
    model = model_creation(package, dataset)
    print(
        f"Model {model.name} created with dataset {dataset.name}"
        f"with package {package.name} in project {project.name}!"
        )
