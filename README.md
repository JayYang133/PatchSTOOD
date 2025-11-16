<div align="center">
  <h2><b> SqLinear: A Linear Architecture for Large-Scale Traffic Prediction via Data-Adaptive Square Partitioning
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

## Contributions

**High-Quality Partitioning.** We propose a novel spatial partitioning method that generates balanced, non-overlapping, and geometrically-regular patches without padding requirements, establishing an optimal foundation for large-scale traffic prediction.

**Hierarchical Linear Modling.** We develop a hierarchical linear interaction module that efficiently captures both inter-patch and intra-patch spatio-temporal interactions, reducing the quadratic complexity bottleneck of attention-based approaches to linear complexity while maintaining modeling fidelity.    

**Theoretical Analysis.** We provide rigorous theoretical guarantees for our partitioning method, proving its effectiveness in preserving network topology and ensuring efficiency.

**Extensive Experiments.** An experimental study on 4 large-scale datasets shows that SqLinear achieves the state-of-the-art prediction accuracy while reducing parameter counts by $2\times$ and accelerating training by $3\times$ compared to existing baselines, validating its practical utility for city-scale deployment.

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
