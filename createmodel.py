import dtlpy as dl
import os

from model_adapter import Adapter


def package_creation(project: dl.Project) -> dl.Package:
    metadata = dl.Package.get_ml_metadata(cls=Adapter,
                                          default_configuration={'weights_filename': 'yolov8l-seg.pt',
                                                                 'epochs': 10,
                                                                 'batch_size': 2,
                                                                 'imgsz': 640,
                                                                 'conf_thres': 0.25,
                                                                 'iou_thres': 0.45,
                                                                 'max_det': 1000,
                                                                 'augment': False},
                                          output_type=dl.AnnotationType.POLYGON,
                                          )
    modules = dl.PackageModule.from_entry_point(entry_point='model_adapter.py')

    package = project.packages.push(package_name='yolov8-seg',
                                    src_path=os.getcwd(),
                                    is_global=True,
                                    package_type='ml',
                                    modules=[modules],
                                    codebase=dl.GitCodebase(git_url='https://github.com/dataloop-ai-apps/yolov8-segmentation.git',
                                                            git_tag='v0.1.3'),
                                    service_config={
                                        'runtime': dl.KubernetesRuntime(pod_type=dl.INSTANCE_CATALOG_REGULAR_M,
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


def model_creation(package: dl.Package, model_name: str = "yolov8seg") -> dl.Model:
    import ultralytics
    label_map = ultralytics.YOLO().names

    model = package.models.create(
        model_name=model_name,
        description='yolov8 for image segmentation',
        tags=['yolov8', 'pretrained', 'segmentation'],
        dataset_id=None,
        status='trained',
        scope='public',
        configuration={
            'weights_filename': 'yolov8l-seg.pt',
            'imgz': 640,
            'id_to_label_map': label_map,
            'label_to_id_map': {v: k for k,v in label_map.items()}
            },
        project_id=package.project.id,
        labels=list(label_map.values()),
        input_type='image',
        output_type='segment'
        )
    return model


if __name__ == "__main__":
    project_name = 'DataloopModels'
    model_name = 'yolov8seg'
    dl.setenv('prod')
    project = dl.projects.get(project_name)
    package = package_creation(project)
    model = model_creation(package, model_name)
    print(
        f"Model {model.name} created with package {package.name} "
        f"in project {project.name}!"
        )
