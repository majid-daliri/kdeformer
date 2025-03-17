# KDEformer: Accelerating Transformers via Kernel Density Estimation

**KDEformer** can approximate the attention in sub-quadratic time with provable spectral norm bounds. On BigGAN image generation, we achieve better generative scores than the exact computation with over $4\times$ speedup. For ImageNet classification with T2T-ViT, KDEformer shows over $18\times$ speedup while the accuracy drop is less than $0.5\%$.

![fig_biggan_image_generation_10](https://github.com/majid-daliri/kdeformer/assets/112280828/8835c59b-0473-4f41-a637-aeae396dc883)

## How the algorithm works 

The KDEformer leverages the body of research on efficient Kernel Density Estimation (KDE). In the KDE problem, we aim to compute the kernel density for an arbitrary query point, `q`. This needs to be estimated to a relative error in a time complexity proportional to `d` divided by a lower bound on the kernel density, `$µ_X(q)$`.
Our technique transforms the problem of finding the sampling matrix and diagonal scaling that satisfy the attention approximation into a generalization of the KDE problem.

We use an efficient KDE procedure for estimating the exponential kernel density to compute a scaling that satisfies the spectral guarantee of the attention approximation. We also design an efficient sampling matrix that satisfies the approximation with a small number of rows. The sampling probabilities need to be proportional to the column norms of the softmax matrix.

Having a generalized KDE procedure for efficiently evaluating the weighted exponential kernel density enables us to approximate the attention mechanism as per our approximation formula. This approach translates the original problem into a Gaussian KDE problem, which allows us to leverage the significant recent progress in this area.

In this paper, we present a novel Locality-Sensitive Hashing (LSH) algorithm tailored for GPU usage, with a comprehensive explanation included within the main body of the text. To provide an overview of its operation, the algorithm initially segregates the entire space utilizing cosine LSH. This leads to the creation of numerous buckets of diverse sizes, the majority of which are typically on the smaller end of the spectrum.

<p align="center">
<img width="436" alt="Screen Shot 1401-11-13 at 14 30 31" src="https://user-images.githubusercontent.com/112280828/216431091-3b69481b-14c3-4909-acec-26503ee142f0.png">
</p>

Following this, the algorithm arranges all the buckets in an order dictated by their Hamming distance, ensuring that neighboring elements have a Hamming distance of less than one. This arrangement ensures that elements bearing strong similarities are positioned adjacent to each other. Post-sorting, the elements are distributed across several blocks. Diagonal blocks are then isolated and multiplied as separate units.

<p align="center">
<img width="436" alt="Screen Shot 1401-11-13 at 14 30 50" src="https://user-images.githubusercontent.com/112280828/216432181-c6e52de9-59ce-4e7a-9e68-221a0a82b670.png">
</p>

-----
## Experiments

- The codes include three experiments; (1) attention approximation on GloVe dataset, (2) BigGAN image geneartion and (3) ImageNet classification
- The proposed algorithm is implemented in ``KDEformer.py``.
- For running ScatterBrain, install the pacakge from https://github.com/HazyResearch/fly.

### 1. Attention Approximation on GloVe Dataset

- For GloVe experiment, download and preparse dataset:
```bash
wget https://nlp.stanford.edu/data/glove.twitter.27B.zip
unzip glove.twitter.27B.zip
```
and run 
```bash
python demo_glove.py
```

### 2. Image Generation with BigGAN

- For BigGAN image generation, run 
```bash
python demo_biggan.py --attention METHOD
```
where METHOD can be one of ``exact`` (default), ``kdeformer``, ``performer``, ``reformer``, ``sblocal``. If ScatterBrain is not installed, the option with ``sblocal`` would not run.

### 3. ImageNet Classification with T2T-ViT model

For ImageNet experiment, first download imagenet validation dataset: https://www.image-net.org/download.php. We will have to login and submit term of access. The total size would be 6.3 GB. Then, set the variable ``DATASETPATH`` in demo_imagenet.py file with your path.

Second download the pretrained T2T-ViT model from the [T2T-ViT repo](https://github.com/yitu-opensource/T2T-ViT/releases). We choose ``82.6_T2T_ViTt_24`` model which can be downloaded by

```sh
wget https://github.com/yitu-opensource/T2T-ViT/releases/download/main/82.6_T2T_ViTt_24.pth.tar
```
and set the variable ``MODELPATH`` in demo_imagenet.py file with your path.

To run the code with the exact attention,
```python
python demo_imagenet.py --attn_method kdeformer --num_samples1 98
```
The exact attention can be tested by replacing argument with ``--attn_method full``.


-----

## Citation
