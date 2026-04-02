---
comments: true
description: Explore the CrowdHuman dataset, a large-scale benchmark for detecting humans in crowded scenes with 470K+ annotated instances from train and validation subsets.
keywords: CrowdHuman, crowd detection, human detection, object detection, pedestrian detection, occlusion, benchmark dataset, deep learning
---

# CrowdHuman Dataset

The [CrowdHuman](https://www.crowdhuman.org/) dataset is a large-scale benchmark designed to better evaluate detectors in crowd scenarios. It is rich in annotations and contains high diversity with an average of 23 persons per image and various kinds of occlusions. CrowdHuman contains 15,000 training, 4,370 validation, and 5,000 testing images with a total of 470K human instances annotated in the train and validation subsets.

Each human instance is annotated with three types of bounding boxes:

- **Head bounding box** (hbox)
- **Human visible-region bounding box** (vbox)
- **Human full-body bounding box** (fbox)

## Dataset Structure

The CrowdHuman dataset is organized with the following structure:

- **Images**: 15,000 training, 4,370 validation, and 5,000 testing high-resolution images containing crowded scenes.
- **Annotations**: Each image has corresponding `.odgt` annotations (one JSON object per line) containing bounding box information for all human instances.

## Applications

The CrowdHuman dataset is widely used for training and evaluating [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models for human detection in crowded environments. The dataset's dense annotations and high diversity make it particularly valuable for developing robust pedestrian detectors that can handle heavy occlusion, which is common in real-world surveillance, autonomous driving, and crowd monitoring applications.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. In the case of the CrowdHuman dataset, the `CrowdHuman.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/CrowdHuman.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/CrowdHuman.yaml).

!!! example "ultralytics/cfg/datasets/CrowdHuman.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/CrowdHuman.yaml"
    ```

## Usage

To train a YOLO26n model on the CrowdHuman dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="CrowdHuman.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=CrowdHuman.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

## Sample Data and Annotations

The CrowdHuman dataset contains a diverse set of crowded scene images with dense human annotations. Here are some key characteristics of the data:

- **High density**: An average of 23 persons per image, making it significantly more crowded than typical pedestrian datasets.
- **Rich annotations**: Three types of bounding boxes per person (head, visible region, full body) enable multi-task training.
- **Occlusion diversity**: Various kinds of occlusions including person-to-person, object-to-person, and partial visibility.

## Citations and Acknowledgments

If you use the CrowdHuman dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{shao2018crowdhuman,
          title={CrowdHuman: A Benchmark for Detecting Human in a Crowd},
          author={Shao, Shuai and Zhao, Zijian and Li, Boxun and Xiao, Tete and Yu, Gang and Zhang, Xiangyu and Sun, Jian},
          journal={arXiv preprint arXiv:1805.00123},
          year={2018}
        }
        ```

We would like to acknowledge the team at Megvii (Face++) for creating and maintaining the CrowdHuman dataset as a valuable resource for the human detection research community. For more information about the CrowdHuman dataset, visit the [CrowdHuman official website](https://www.crowdhuman.org/).

## FAQ

### What is the CrowdHuman dataset and what makes it unique?

The [CrowdHuman](https://www.crowdhuman.org/) dataset is a large-scale benchmark specifically designed for evaluating human detectors in crowded scenes. Key features include:

- **Size**: 15,000 training, 4,370 validation, and 5,000 testing images.
- **Density**: An average of 23 persons per image with 470K+ total human instances.
- **Annotations**: Three bounding box types per person — head, visible region, and full body.
- **Occlusion**: Extensive occlusion diversity for robust detector evaluation.

### How can I train a YOLO26 model on the CrowdHuman dataset?

To train a YOLO26 model on CrowdHuman for 100 epochs with an image size of 640:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolo26n.pt")

        # Train the model
        results = model.train(data="CrowdHuman.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=CrowdHuman.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

For additional configuration options, refer to the model [Training](../../modes/train.md) page.

### What annotation types does CrowdHuman provide?

CrowdHuman provides three types of [bounding box](https://www.ultralytics.com/glossary/bounding-box) annotations for each human instance:

1. **Head bounding box (hbox)**: Covers only the head region.
2. **Visible-region bounding box (vbox)**: Covers the visible parts of the person.
3. **Full-body bounding box (fbox)**: Covers the entire body including occluded parts.

The Ultralytics YAML configuration uses full-body bounding boxes by default for [object detection](https://www.ultralytics.com/glossary/object-detection) training.

### How does CrowdHuman compare to other pedestrian datasets?

CrowdHuman stands out from other pedestrian datasets due to its significantly higher crowd density (23 persons/image average) and comprehensive annotation scheme. While datasets like [COCO](coco.md) include person annotations among many categories, CrowdHuman is specifically focused on human detection in crowded scenarios, making it ideal for training robust pedestrian detectors for surveillance and autonomous driving applications.

### How can I cite the CrowdHuman dataset in my research?

If you use CrowdHuman in your research, please cite it using the following BibTeX:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{shao2018crowdhuman,
          title={CrowdHuman: A Benchmark for Detecting Human in a Crowd},
          author={Shao, Shuai and Zhao, Zijian and Li, Boxun and Xiao, Tete and Yu, Gang and Zhang, Xiangyu and Sun, Jian},
          journal={arXiv preprint arXiv:1805.00123},
          year={2018}
        }
        ```
