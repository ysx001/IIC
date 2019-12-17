# Clustering-based unsupervised brain MRI segmentation

We propose an unsupervised approach to brain MRI segmentation based on k-means clustering of pixel features. We investigate and experiment between different classes of methods for pixel feature extraction: heuristic methods based on patches, and learned methods based on neural networks. We evaluate our algorithm on brain images from the Human Connectome Project (HCP) public dataset.

Our best performing model achieves a DICE score of 0.8 on grey and white matter. We also perform some qualitative analysis on the segmentation results and study the effect different hyperparameters have on the characterstics of the segmentation.

## Getting Started

### Project Structure

- `k-means`: contains the python and matlab code for the segmentation pipeline
  - `loaddata.py`, `loadfreesurfer.py` loads the input data and oracle FreeSurfer segmentation data, and re-group the segmentation labels into 5 catagories: Deep Grey Matter, Grey Matter, White Matter, CSF, and Brain Mask.
  - `meanshift.py` performs the meanshift operation on the input data to obtain a baseline for the project
  - `pretrain_features.py` loads the input data and perform feature extraction on different models
  - `mri_dataset.py` and `unet_features.py` loads the input data, using half of them as training data to fine-tune a U-Net
  - `classfication` contains the MATLAB code for performing k-means
  - `matching.py` performs the Hungarian matching, reorder the predictions, and calculate the DICE score and pre-pixel accuracy
- `iic` contains the code for loading MRI data and training them on a VGG based IIC network

### Example data 

The processed dataset is located on a lab server. An example of the data for one subject is at [this link](https://drive.google.com/drive/folders/1nbeTn4zNrVG4dRlF3fLDZNhjUA2UCBRh?usp=sharing)

### CodaLab
The CodaLab worksheet can be found at [this link](https://worksheets.codalab.org/worksheets/0x041a17a82b8249a0bd7c128d51e8bafa)


## Authors

* **Yuxin Hu** - [Github](https://github.com/yuxinhu)
* **Jason Fang** - [Github](https://github.com/jasonf7)
* **Sarah Xu** - [Github](https://github.com/ysx001)
