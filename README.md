<div align="center">
  <h2><b> PartchSTOOD
 </b></h2>
</div>

<div align="center">

</div>

<div align="center">


</div>

<div align="center">

</div>


---

<div align="center">

<img src="./img/model.png">

</div>

## Requirements
- torch==1.11.0
- timm==1.0.12
- scikit_learn==1.0.2
- tqdm==4.67.1
- pandas==1.4.1
- numpy==1.22.3

## Folder Structure

```tex
└── code-and-data
    ├── config                 # Configuration files for different datasets and hyperparameters
    ├── cpt                    # Pre-trained model checkpoints
    ├── data                   # Traffic datasets, adjacency matrices, and metadata files
    ├── lib
    │   |──  utils.py          # Utility functions for data preprocessing and evaluation metrics
    ├── log                    # Training and testing log files
    ├── models
    │   |──  model.py          # Core implementation of the SqLinear architecture
    ├── main.py                # Main script for model training and evaluation
    └── README.md              # Project documentation and usage guide
```


## Quick Demos
1. Download datasets and place them under `./data`
2. We provide pre-trained weights of results in the paper and the detail configurations under the folder `./config`. For example, you can test the SD dataset by:

```
python main.py --config ./config/SD.conf
```

3. If you want to train the model yourself, you can use the code at line 288 of the main file.
