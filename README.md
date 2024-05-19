# YOLOv8 Segmentation Model Adapter

## Introduction

This repo is a model integration between [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) segmentation model and [Dataloop](https://dataloop.ai/)  
For the object detection YOLOv8 adapter, check out [this repo](https://github.com/dataloop-ai-apps/yolov8/).

Ultralytics YOLOv8 represents a modernized iteration, refining the successes of prior YOLO models. With added features and improvements, it aims to enhance both performance and versatility. YOLOv8 prioritizes speed, accuracy, and user-friendly design, making it a reliable option for tasks like object detection, tracking, instance segmentation, image classification, and pose estimation. In this repo we implement the integration between YOLOv8 in its segmentation model with our Dataloop platform.

YOLOv8 segmentation achieves state-of-the-art results in the task of identifying objects in images while also outlying its contours, like in the following images that display its results in a variety of tasks:

<img width="1454" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/865d76a7-dc53-47d6-b5cf-57f18919f601">

<img width="1454" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/58fa87e8-f398-4e7e-b00e-784a1fff18fc">

<img width="1454" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/4ca3a84b-d12e-4682-b66e-3c4a8f94fa03">


## Requirements

* dtlpy
* ultralytics==8.0.17
* torch==2.0.0
* pillow>=9.5.0
* An account in the [Dataloop platform](https://console.dataloop.ai/)

  
## Installation

To install the package and create the YOLOv8 model adapter, you will need a [project](https://developers.dataloop.ai/tutorials/getting_started/sdk_overview/chapter/#to-create-a-new-project) and a [dataset](https://developers.dataloop.ai/tutorials/data_management/manage_datasets/chapter/#create-dataset) in the Dataloop platform. The dataset should have [directories](https://developers.dataloop.ai/tutorials/data_management/manage_datasets/chapter/#create-directory) containing its training and validation subsets.

### Installing in the Platform

In the model management page, choose the AI Library button in the menu and in the drop-down menu, pick "Public Models" to see all the publicly available models. You will see yolov8-seg in the list and you can create a copy of it by selecting the "create model" option as presented here:

<img width="1413" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/5c2ab8c9-4afd-420e-893d-db25e99539d8">

You will be presented with the options to choose name, artifact locatiion and tags:

<img width="676" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/ca7480f7-4fb5-4602-8fcc-734f7b9b8483">

Then to choose between fine tuning or just choosing the pretrained weights from Ultralytics (trained on the COCO dataset). If you choose the pretrained weights, the model will be created with status ```trained```, otherwise, when choosing fine-tuning, you have to select a dataset, define the DQL filter or folder for the training and validation subsets, and choose a recipe for training. The model will be created with status ```created``` and you will need to run the training for it before it can be used.

<img width="676" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/df467411-de86-4b9c-bbba-6f3c0d98a73c">

Lastly, define the model configurations:

<img width="676" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/c2f4ec39-1707-43d6-aece-6c861804adda">

After this, the model will appear in the list of the proejct models in Model Management with the name you chose. It can be trained, evaluated and deployed.

### Installing via the SDK

To install YOLOv8-segmentation via SDK, all that is necessary is to clone the model from the AI Library to your own project:

```python
import dtlpy as dl
project = dl.projects.get('Yolo Project')
public_model = dl.models.get(model_name="yolov8-seg")
model = project.models.clone(from_model=public_model,
                             model_name='yolov8-seg-clone',
                             project_id=project.id)
```

For more options when installing the model, check this [page](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#finetune-on-a-custom-dataset).

## Training and Fine-tuning

Training YOLOv8 segmentation can either be done via the platform or the SDK. For either purpose, it is necessary to first set the models subsets for training and validation. In the previous step, you saw how to define the train and validation subsets when creating your copy of the model. If you wish to do this via the SDK or modify them, you can follow [these instructions](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#define-dataset-subsets).

**ATTENTION:** To ensure that training will be successful, verify that the items in the dataset are annotated with annotations of type **polygon**. 

### Editing the configuration

To edit configurations via the platform, go to the YOLOv8-segmentation page in the Model Management and edit the json file displayed there or, via the SDK, by editing the model configuration. Click [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#model-configuration) for more information.

The basic configurations included are:

* ```epochs```: number of epochs to train the model (default: 50)
* ```batch_size```: batch size to be used during the training (default: 2)
* ```imgsz```: the size (imgsz x imgsz) to which images are reshaped before going through the model (default: 640)
* ```device```: whether to train on ```cpu``` or ```cuda``` (default to automatic detection of whether the instance has a GPU)
* ```augment```: boolean, ```True``` if you wish to use ultralytics' augmentation techniques on the training data (default: ```False```)
* ```labels```: The labels over which the model will train and predict (defaults to the labels in the model's dataset's recipe)
* ```id_to_label_map```: Dictionary mapping numbers to labels to guide the model outputs
* ```label_to_id_map```: Inverse map from ```id_to_label_map```
* ```inference_args```: Dictionary containing inference-specific arguments, such as:
  * ```conf```: the confidence threshold, below which predictions will be discarded. Default: 0.2
  * ```iou```: the iou threshold, below which inferences will be discarded. Default: 0.7
  * ```imgsz```: image size to which input images will be converted during inference (size will be ```imgsz x imgsz```). Default: 640
  * ```half```: set to ```true``` to use half-precision during inference. Default: false
  * ```device```: Device in which to run inference. Default: cpu
  * ```max_det```: Maximum number of detections per image. Default: 300
  * ```augment```: Whether to use augmentation on inference or not. Default: false
  * ```agnostic_nms```: Whether to use agnostic non-maximum suppression during inference. Default: false
  * ```classes```: Restrict inference just to the set of classes described here, separated by commas. Default: null
  * ```retina_masks```: Use high quality retina masks if available. Default: false

Additional configurations shown in the [Ultralytics documentation](https://docs.ultralytics.com/usage/cfg/#train) can be included in a dictionary under the key ```yaml_config```.

### Training with the Platform

In the Model Management page of your project, find a version of your YOLOv8-segmentation model with the status **created** and click the three dots in the right of the model's row and select the "Train" option:

<img width="1417" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/f9b8b65c-2e45-488e-a1bc-7dde1a47ae22">

Edit the configuration for this specific run of the training, and choose which instance in which it will run:

<img width="677" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/69a0e642-b70d-4063-9756-efdb25b44550">

and select the service fields (more information [here](https://developers.dataloop.ai/tutorials/faas/custom_environment_using_docker/chapter/)):

<img width="677" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/272b45e9-f975-4f43-af56-da6a453172b2">

Now kick back and wait for the training to finish.

### Training with the SDK

To train the model with the SDK, get the model id and define the service configuration for its training:

```python
model_entity = dl.models.get(model_id='<yolov8seg-model-id>')
ex = model_entity.train()
ex.logs(follow=True)  # to stream the logs during training
custom_model = dl.models.get(model_id=model_entity.id)
print(custom_model.status)
```

For more information on how to customize the service configuration that will run the training, check the [documentation](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#train).

## Deployment

After installing the pretrained model or fine-tuning it on your data, it is necessary to deploy it, so it can be used for prediction.

### Deploying with the Platform

In the Model Management page of your project, find a pretrained or fine-tuned version of your YOLOv8-segmentation model and click the three dots in the right of the model's row and select the "Deploy" option:

<img width="1417" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/3d710956-2128-4964-b8f4-f864fbf18112">

Here you can choose the instance, minimum and maximum number of replicas and queue size of the service that will run the deployed model (for more information on these parameters, check [the documentation](https://developers.dataloop.ai/tutorials/faas/advance/chapter/#autoscaler)):

<img width="677" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/871c85de-9fc9-44e0-a5c2-86596975ddb9">

Proceed to the next page and define the service fields (which are explained [here](https://developers.dataloop.ai/tutorials/faas/custom_environment_using_docker/chapter/)).

<img width="677" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/f9b1c3a8-3a7d-4d33-903f-f80a15f27e2d">

After this, your model is deployed and ready to run inference.

### Deploying with the SDK

To deploy with the default service configuration defined in the package:

```python
model_entity = dl.models.get(model_id='<model-id>')
model_entity.deploy()
```

For more information and how to set specific service settings for the deployed model, check the [documentation](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#clone-and-deploy-a-model).

## Testing

Once the model is deployed, you can test it by going to the Model Management, selecting the YOLOv8-segmentation model and then going to the test tab. Drag and drop or select an image to the image area:

<img width="1450" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/ba0c34e5-6a3e-4474-ad98-06d28eb164c3">

click the test button and wait for the prediction to be done:

<img width="1450" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/e000d206-7627-44b2-afa6-c46ab9551d7a">

## Prediction

### Predicting in the Platform

The best way to perform predictions in the platform is to add a "Predict Node" to a pipeline:

<img width="912" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/ae66f60c-5f3b-403e-a52b-540573a0aea9">

Click [here](https://developers.dataloop.ai/onboarding/08_pipelines/) for more information on Dataloop Pipelines.

### Predicting with the SDK

The deployed model can be used to run prediction on batches of images:

```python
model_entity = dl.models.get(model_id='<model-id>')
results = model_entity.predict_items([item_id_0, item_id_1, ..., item_id_n])
print(results)
```

For more information and options, [check the documentation](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#predict-items).

## Sources and Further Reading

* [Ultralytics documentation](https://docs.ultralytics.com/)
