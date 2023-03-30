# Re-implementation of Graph Attention Network for CS6208: Advanced Topics in Artificial Intelligence

This is a pytorch re-implementation from scratch of the [Graph Attention Network (GAT)](https://openreview.net/pdf?id=rJXMpikCZ) published at ICLR 2018. This implementation uses the standard citation network benchmark dataset Cora for transductive learning. The dataset processing part refers to [this repository](https://github.com/Diego999/pyGAT)

# Significant network modifications

In this implementation, I use cross-entropy losses derived from multiple Graph Attentional Layers (GALs) for supervising 
class prediction, i.e., multi-head prediction. For training, these losses are equally weighted, while, the corresponding logits are combined 
together for class prediction during inference. In the original implementation, they only use one loss for supervising 
class prediction. 

[//]: # (I think multiple losses can focus on different patterns &#40;biases&#41; for recognition, and they will perform )
[//]: # (better compared with the one loss version. This mechanism is similar to [Ensemble Deep Learning]&#40;https://www.sciencedirect.com/science/article/abs/pii/S095219762200269X&#41;.)

# Running command

```
python main --epochs 1000 --lr 2e-2 --weight_decay 5e-4 --val-interval 20 --num_gals_in 8 --num_gals_predict 2 --gpu-ids 0
```

# Performances

The training on a Tesla V100 (32G memory) takes about 37s for 1000 epochs. The final accuracy is 84.70%.

# Requirements

The implementation relies on the following main packages:
Python 3.7,
Pytorch 1.13.1,
Scipy 1.7.3.
