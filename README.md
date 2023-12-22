### Code used for the results in the paper  ["MCLDA: Multi-level Contrastive Learning for Domain Adaptive Semantic Segmentation"](https://dl.acm.org/doi/abs/10.1145/3628797.3628938)

# Description
We introduce MCLDA, a method that employs multi-level contrastive learning to align domains and enhance feature discriminability. Additionally, we introduce an image mixing strategy to address imbalanced data and consider class context. Our proposed method demonstrates comparable performance to the top-performing methods when using the same segmentation architecture, Deeplabv2 (ResNet101).

# Getting started
## Prerequisite
*  CUDA/CUDNN 
*  Python3
*  Packages found in requirements.txt

# Run training and testing

### Example of training a model with unsupervised domain adaptation on GTA5->CityScapes on a single gpu

python3 trainUDA.py --config ./configs/configUDA.json --name UDA

### Example of testing a model with domain adaptation with CityScapes as target domain

python3 evaluateUDA.py --model-path *checkpoint.pth*
