# NumpyViT
This project implements a simplified Vision Transformer (ViT), as described in the original ViT paper, specifically tailored for the classification of satellite images from the EuroSAT dataset. The EuroSAT dataset comprises satellite images covering 10 different land use and land cover classes. Our implementation leverages NumPy and Pillow for image processing and model construction, emphasizing educational purposes and transparency in understanding transformer-based models for image classification.

## Project Structure

- `vit.py`: Contains the Vision Transformer model implementation.
- `dataset.py`: A module for loading and preprocessing the EuroSAT dataset.
- `dataloader.py`: Provides a custom data loader for batching and iterating over the dataset.
- `train.py`: Script to train the ViT model on the EuroSAT dataset.
- `utils.py`: Utility functions, including image transformations and loss calculations.
- `requirements.txt`: Lists the project dependencies for easy installation.

## Installation

To set up the project environment:

1. Clone the repository:
```
1. git clone https://github.com/yourgithub/yourprojectname.git
```
2. Navigate to the project directory:
```
cd NumpyViT
```
3. Install the required dependencies:
```
pip install -r requirements.txt
```

## Data preparation

Organize images as such in the data directory of the project (if it does not exist, create it):
```
data/
├── train/
│   ├── cls1/
│   ├── ├── cls1_img1.jpg
│   ├── ├── cls1_img2.jpg
│   ├── ├── cls1_img3.jpg
│   ├── cls2/
│   ├── ├── cls2_img1.jpg
│   ├── ├── cls2_img2.jpg
│   ├── cls3/
│   ├── ├── ...
│   └── ...
├── val/
│   ├── ...
│   └── ...
└── test/
    ├── ...
    └── ...
```

## Usage

To train the Vision Transformer model, run:
```
python train.py
```

## Results

Our Vision Transformer model was trained and evaluated on the EuroSAT dataset, following the original architecture proposed in the ViT paper. We placed 80% of the dataset in the train folder, 10% in the val folder and 10% in the test folder. After 100 training epochs with a batch size of 16, the table below summarizes the classification accuracy achieved on the dataset:

| Split    | Accuracy (%) |
|----------|--------------|
| Training | 98.5         |
| Validation | 89.4         |
| Test     | 89.6         |

## Citation

If you find this project useful in your research, please consider citing the original Vision Transformer paper:
```
@article{dosovitskiy2020image,
title={An image is worth 16x16 words: Transformers for image recognition at scale},
author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
journal={arXiv preprint arXiv:2010.11929},
year={2020}
}
```
To cite this specific implementation:
```
@article{fontaine2024numpyvit,
title={NumpyViT: a numpy implementation of the vision transformer},
author={Fontaine, Hans-Olivier},
year={2024}
}
```

## License

Distributed under the MIT License. See `LICENSE` for more information.


