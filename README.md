# YOLOv8 Segmentation Model Adapter

## Introduction

This repo is a model integration between [Ultralytics Yolov8](https://github.com/ultralytics/ultralytics) segmentation model and [Dataloop](https://dataloop.ai/)  
For the object detection YOLOv8 adapter, check out [this repo](https://github.com/dataloop-ai-apps/yolov8/).

Ultralytics YOLOv8 represents a modernized iteration, refining the successes of prior YOLO models. With added features and improvements, it aims to enhance both performance and versatility. YOLOv8 prioritizes speed, accuracy, and user-friendly design, making it a reliable option for tasks like object detection, tracking, instance segmentation, image classification, and pose estimation. In this repo we implement the integration between YOLOv8 in its segmentation model with our Dataloop platform.

YOLOv8 segmentation achieves state of the art results in the task of identifying objects in images while also outlying its contours, like in the following images that display its results in a variety of tasks:

<img width="1218" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/0dd04299-a163-45fd-a142-c68a3c52e37b">

<img width="1218" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/9832ceba-7737-42fb-ab40-3d706177a6f8">

<img width="1218" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/59f36c0e-0303-4865-acbf-c68d5d13d421">


## Requirements

* dtlpy
* ultralytics==8.0.17
* torch==2.0.0
* pillow>=9.5.0
* An account in the [Dataloop platform](www.console.dataloop.com)

  
## Installation

To install the package and create the YOLOv8 model adapter, you will need a [project](https://developers.dataloop.ai/tutorials/getting_started/sdk_overview/chapter/#to-create-a-new-project) and a [dataset](https://developers.dataloop.ai/tutorials/data_management/manage_datasets/chapter/#create-dataset) in the Dataloop platform. The dataset should have [directories](https://developers.dataloop.ai/tutorials/data_management/manage_datasets/chapter/#create-directory) containing its training and validation subsets.

Once this is setup, run the ```createmodel.py``` script contained in this repo, with the project and dataset names:

```bash
python createmodel.py -p <project-name> -d <dataset-name> [-e <prod-or-rc> -m <model-name>]
```

The ```-p``` option should be followed by the chosen project name, ```-d``` by the dataset name. There are two optional flags: ```-e``` chooses the environent (either production or rc, production by default) and ```-m``` allows to set a custom name for the installed model (it's installed as yolov8seg by default).

## Training

Training YOLOv8 segmentation can either be done via the platform or the SDK. For either purpose, it is necessary to first set the models subsets for training and validation. This can be done via the SDK, once you get the YOLOv8segmentation's model id. Getting the model id can be done either via the platform or the SDK:

```python
project = dl.projects.get('<project-name>')
project.models.list().print()
```

should display a table with all the models in the project along with their id's. Or, via the platform, go to the Model Management page, find the YOLOv8 segmentation model in the list, and click in the three dots in the right end of its row and copy the id:
<img width="143" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/7e18b364-84b0-4257-947c-a5b7405a4da7">

and then, define the subsets:

```python
model_entity = dl.models.get(model_id=<yolov8seg-model-id>)
model_entity.metadata['system'] = dict() if not model_entity.metadata.get('system', False) else model_entity.metadata['system']
model_entity.metadata['system']['subsets'] = {
    'train': {'filter': dl.Filters(field='dir', values='<trainset-directory>')},
    'validation': {'filter': dl.Filters(field='dir', values='<valset-directory>')}
}
model_entity.update(system_metadata=True)
```
where you should fill the ```<trainset-directory>``` and ```<valset-directory>``` with the name of the respective directories for these subsets in your dataset.

### Editing the configuration

To edit configurations via the platform, go to the YOLOv8-segmentation page in the Model Management and edit the json file displayed there:

<img width="945" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/4c4186fd-535e-4968-81a0-730b6696445d">

or, via the SDK, by editing the model configuration:

```python
model_entity = dl.models.get(model_id='<yolov8seg-model-id>')
model_entity.configuration = {...}
model_entity.update()
```

These are the keys that can be configured:

* ```epochs```: number of epochs to train the model (default: 50)
* ```batch_size```: batch size to be used during the training (default: 2)
* ```imgsz```: the size (imgsz x imgsz) to which images are reshaped before going through the model (default: 640)
* ```device```: whether to train on ```cpu``` or ```cuda``` (default to automatic detection of whether the instance has a GPU)
* ```augment```: boolean, ```True``` if you wish to use ultralytics' augmentation techniques on the training data (default: ```False```)
* ```labels```: The labels over which the model will train and predict (defaults to the labels in the model's dataset's recipe)
* ```id_to_label_map```: Dictionary mapping numbers to labels to guide the model outputs
* ```label_to_id_map```: Inverse map from ```id_to_label_map```

### Training with the Platform

### Training with the SDK

To train the model with the SDK, get the model id and define the service configuration for its training:

```python
model_entity = dl.models.get(model_id='<yolov8seg-model-id>')
ex = model_id.train(service_config={
    'runtime': dl.KubernetesRuntime(pod_type=dl.INSTANCE_CATALOG_GPU_K80_S,
                                    autoscaler=dl.KubernetesRabbitmqAutoscaler(
                                        min_replicas=0,
                                        max_replicas=1),
                                    preemptible=False,
                                    concurrency=1).to_json(),
    'executionTimeout': 10000 * 3600
})
ex.logs(follow=True)  # to stream the logs during training
custom_model = dl.models.get(model_id=model_entity.id)
print(custom_model.status)
```

## Deployment

Once the model is trained, it is necessary to deploy it so it can be used for evaluation and prediction.

### Deploying with the Platform

### Deploying with the SDK

To deploy with the default service configuration defined in the package:

```python
model_entity = dl.models.get(model_id='<model-id>')
model_entity.status = 'deployed'
model_entity.update()
```

## Testing

Once the model is deployed, you can test it by going to the Model Management, selecting the YOLOv8-segmentation model and then going to the test tab. Drag and drop or select an image to the image area:

<img width="1218" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/194c0085-88a9-476a-90e4-efdf4e07070e">

click the test button and wait for the prediction to be done:

<img width="1218" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/074e0281-bd7a-4913-8e39-10b4d6998efc">

## Prediction

### Predicting in the Platform

The best way to perform predictions in the platform is to add a Predict Node to a pipeline:

### Predicting with the SDK

```python
model_entity = dl.models.get(model_id='<model-id>')
item_0 = dl.items.get(item_id='<item_0_id>')
item_1 = dl.items.get(item_id='<item_1_id>')
...
item_n = dl.items.get(item_id='<item_n_id>')
results = model_entity.predict_items([item_0, item_1, ..., item_n], upload_annotations=False)
print(results)
```

## Sources and Further Reading

*[Ultralytics documentation](https://docs.ultralytics.com/)
