# MLOH

The Matlab codes of image retrieval using the MLOH descriptor!

## Project Structure

```
MLOH/
├── test_main.m                 # Main entry point
├── layers/                     # Custom network layers
│   ├── GeM_lxp.m              # GeM pooling layer
│   ├── LayerNorm_.m            # Layer normalization
│   ├── Lit.m                   # Iterative multi-layer integration
│   └── l2NormalizationLayer.m  # L2 normalization
├── utils/                      # Utility functions
│   ├── features_process_.m     # Feature extraction pipeline
│   ├── gfa_.m                  # Gabor filter attention
│   ├── get_Lw.m                # Whitening parameter computation
│   ├── whitenapply.m           # Apply whitening
│   ├── whitenlearn.m           # Learn whitening
│   ├── mergeBatchNormRes50.m   # BN fusion for ResNet50
│   ├── mergeBatchNormRes101.m  # BN fusion for ResNet101
│   ├── crop_qim.m              # Query image cropping
│   ├── imresizemaxd.m          # Image resizing
│   └── cid2filename.m          # CID to filename mapping
├── eval/                       # Evaluation
│   ├── test_net_.m             # Network testing
│   ├── configdataset.m         # Dataset configuration
│   └── compute_map.m           # mAP computation
├── dataset/                    # Image datasets
└── ims/                        # Whitening training images
```

## Data Preparation

Oxford5k: [Oxford5k](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings)
Paris6k: [Paris6k](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings)
Annotations: [Annotations](http://cmp.felk.cvut.cz/revisitop/)

```
 └ dataset
   └ oxford5k
     └ oxford5k_images
       └ ...
   └ paris6k
     └ paris6k_images
       └ ...
```

## Pretrained Models

You can download our pretrained models and parameters from [Baidu Netdisk](https://pan.baidu.com/s/1GsWLCJZSb5FMNVsPm8bY_A?pwd=7mkq).

## Evaluation

To evaluate on Roxford5k, Rparis6k, oxford5k, and paris6k run:
```
test_main.m
```

## Acknowledgement

Our work builds upon the foundation laid by [Filip Radenovic](https://github.com/filipradenovic/revisitop), and we would like to express our gratitude for his contributions to the image retrieval community.
