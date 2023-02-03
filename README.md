# KDEformer: Accelerating Transformers via Kernel Density Estimation

**KDEformer** can approximate the attention in sub-quadratic time with provable spectral norm bounds. On BigGAN image generation, we achieve better generative scores than the exact computation with over $4\times$ speedup. For ImageNet classification with T2T-ViT, KDEformer shows over $18\times$ speedup while the accuracy drop is less than $0.5\%$.

[ADD BIGGAN RESULT IMAGES HERE]

## How the algorithm works 

Also, along with the theory, we designed a practical method suitable for it, which works well in practice on the mentioned datasets. The big picture technique used has a very simple structure. First, by using LSH, it samples the elements that have the greatest impact on the final output, which is a sparse estimator. Also, among the rest of the elements, we will sample randomly, so that we will have an unbiased estimator that works perfectly empirically.

Also, we use a new special LSH in this paper which works on GPU and its description is given in detail in the paper. But in order to have a general intuition of how it works, we will give a general description of it. First, it partitions the entire space using LSH cosine. Therefore, we will have several buckets with various sizes, most of which are generally small in size.

<img width="436" alt="Screen Shot 1401-11-13 at 14 30 31" src="https://user-images.githubusercontent.com/112280828/216431091-3b69481b-14c3-4909-acec-26503ee142f0.png">

Then that algorithm sorts all buckets based on Hamming distance in such a way that both adjacent elements have hashing with Hamming distance less than one. In this way, two elements are placed next to each other if they are very similar to each other. After sorting, we divide the elements into different blocks and multiply the diagonal blocks as a separate family.

<img width="436" alt="Screen Shot 1401-11-13 at 14 30 50" src="https://user-images.githubusercontent.com/112280828/216432181-c6e52de9-59ce-4e7a-9e68-221a0a82b670.png">

-----
## Experiments

- The codes include two experiments; (1) attention approximation on GloVe dataset and (2) BigGAN image geneartion.
- The proposed algorithm is implemented in ``KDEformer.py``.
- For running ScatterBrain, install the pacakge from https://github.com/HazyResearch/fly.
- For GloVe experiment, download and preparse dataset:
```bash
wget https://nlp.stanford.edu/data/glove.twitter.27B.zip
unzip glove.twitter.27B.zip
```
and run 
```bash
python demo_glove.py
```
- For BigGAN image generation, run 
```bash
python demo_biggan.py --attention METHOD
```
where METHOD can be one of ``exact`` (default), ``kdeformer``, ``performer``, ``reformer``, ``sblocal``. If ScatterBrain is not installed, the option with ``sblocal`` would not run.

-----

## Citation
