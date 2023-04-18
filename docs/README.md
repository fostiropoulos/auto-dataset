
<h1 style="text-align:center"> Batch Model Consolidation: A Multi-Task Model Consolidation Framework </h1>
<hr style="border:2px solid gray">

<h3 style="text-align:center"> Iordanis Fostiropoulos &nbsp;&nbsp;&nbsp; Jiaye Zhu &nbsp;&nbsp;&nbsp; Laurent Itti</h3>
<p style="text-align:center"> University of Southern California</p>

## Abstract

In Continual Learning (CL), a model is required to learn a stream of tasks sequentially 
without significant performance degradation on previously learned tasks. 
Current approaches fail for a long sequence of tasks from diverse domains and difficulties. 
Many of the existing CL approaches are difficult to apply in practice due to excessive memory 
cost or training time, or are tightly coupled to a single device. With the intuition 
derived from the widely applied mini-batch training, we propose Batch Model Consolidation 
(**BMC**) to support more realistic CL under conditions where multiple agents are 
exposed to a range of tasks. During a _regularization_ phase, BMC trains multiple 
_expert models_ in parallel on a set of disjoint tasks. Each expert maintains weight 
similarity to a _base model_ through a _stability loss_, and constructs a 
_buffer_ from a fraction of the task's data. During the _consolidation_ phase, 
combine the learned knowledge on `batches' of _expert models_ using a 
_batched consolidation loss_ in _memory_ data that aggregates all buffers. 
We thoroughly evaluate each component of our method in an ablation study and demonstrate 
the effectiveness on standardized benchmark datasets Split-CIFAR-100, Tiny-ImageNet, 
and the Stream dataset composed of 71 image classification tasks from diverse domains 
and difficulties. Our method outperforms the next best CL approach by 70% and is the 
only approach that can maintain performance at the end of 71 tasks.

## Overview

![The intuition of our work](https://drive.google.com/file/d/1ZgwGy1Ta2u9Wim0D010uf7cSGw07qts9/view?usp=share_link)

![A single incremental step of BMC](https://drive.google.com/file/d/1nG4kD2PCP0sMZxBRD3LN8fZjzYvQrpTJ/view?usp=share_link)

![Paralleled multi-expert training framework](https://drive.google.com/file/d/1NAswFVQtiNn6xkilUig42guGfvi-babV/view?usp=share_link)

## The Stream Dataset

Stream dataset implements the logic for processing and managing a large sequence of datasets, 
and provides a method to train on interdisciplinary tasks by projecting all datasets on the same dimension,
by extracting features from pre-trained models.

See [the repository](https://github.com/fostiropoulos/stream/tree/cvpr_release) for Stream dataset installation and usages.

## Class-Incremental Learning



## Citation
