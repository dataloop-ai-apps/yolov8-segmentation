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

### Installing in the Platform

### Installing via the SDK

To install YOLOv8-segmentation via SDK, all that is necessary is to clone the model from the AI Library to your own project:

```python
public_model = dl.models.get(model_name="yolov8-seg")
model = project.models.clone(from_model=public_model,
                             model_name='resnet_50',
                             project_id=project.id)
```

For more options when installing the model, check this [page](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#finetune-on-a-custom-dataset).

## Training and Finetuning

Training YOLOv8 segmentation can either be done via the platform or the SDK. For either purpose, it is necessary to first set the models subsets for training and validation. In the previous step, you saw how to define the train and validation subsets when creating your copy of the model. If you wish to do this via the SDK or modify them, you can follow [these instructions](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#define-dataset-subsets).

**ATTENTION:** To ensure that training will be successful, verify that the items in the dataset are annotated with annotations of type **polygon**. 

### Editing the configuration

To edit configurations via the platform, go to the YOLOv8-segmentation page in the Model Management and edit the json file displayed there or, via the SDK, by editing the model configuration. Click [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#model-configuration) for more information.

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

In the Model Management page of your project, find a version of your YOLOv8-segmentation model with the status **created** and click the three dots in the right of the model's row and select the "Train" option:

<img width="1421" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/42dc335b-bae3-4992-97d9-47030e1f95da">

Edit the configuration for this specific run of the training, and choose which instance in which it will run:

<img width="674" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/a370c8c7-fcdf-4215-8268-b810055fefca">

and select the service fields (more information [here](https://developers.dataloop.ai/tutorials/faas/custom_environment_using_docker/chapter/)):

<img width="674" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/e2cb2587-d85d-4dd3-baec-0c3703cc8da1">

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

After installing the pretrained model or finetuning it on your data, it is necessary to deploy it so it can be used for prediction.

### Deploying with the Platform

In the Model Management page of your project, find a pretrained or finetuned version of your YOLOv8-segmentation model and click the three dots in the right of the model's row and select the "Deploy" option:

<img width="1430" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/413324a5-251f-459c-998b-72022e13c5be">

Here you can choose the instance, minimum and maximum number of replicas and queue size of the service that will run the deployed model (for more information on these parameters, check [the documentation](https://developers.dataloop.ai/tutorials/faas/advance/chapter/#autoscaler)):

<img width="679" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/8d56ee37-94ea-4971-9ceb-511648d58d4f">

Proceed to the next page and define the service fields (which are explained [here](https://developers.dataloop.ai/tutorials/faas/custom_environment_using_docker/chapter/)).

<img width="679" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/71690c05-0049-45fd-b83b-34e30e2d8ff2">

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

<img width="1218" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/194c0085-88a9-476a-90e4-efdf4e07070e">

click the test button and wait for the prediction to be done:

<img width="1218" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/074e0281-bd7a-4913-8e39-10b4d6998efc">

## Prediction

### Predicting in the Platform

The best way to perform predictions in the platform is to add a Predict Node to a pipeline:

<img width="873" alt="image" src="https://github.com/dataloop-ai-apps/yolov8-segmentation/assets/124260926/d52ac35b-4982-472d-8a82-1f5f01b1da89">

Click [here](https://developers.dataloop.ai/onboarding/08_pipelines/) for more information on Dataloop Pipelines.

### Predicting with the SDK

The deployed model can be used to run prediction on batches of images:

```python
model_entity = dl.models.get(model_id='<model-id>')
item_0 = dl.items.get(item_id='<item_0_id>')
item_1 = dl.items.get(item_id='<item_1_id>')
...
item_n = dl.items.get(item_id='<item_n_id>')
results = model_entity.predict_items([item_0, item_1, ..., item_n], upload_annotations=False)
print(results)
```

For more information and options, [check the documentation](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#predict-items).

## Sources and Further Reading

* [Ultralytics documentation](https://docs.ultralytics.com/)
