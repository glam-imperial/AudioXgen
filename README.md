# AudioXgen
Audio Explanation Synthesis with Generative Foundation Models


## Contents:

**[audio_explanation_generation](audio_explanation_generation.py)** --> Inference script for generating audio explanations using AudioXgen.\
**[fidelity_performance](fidelity_performance.py)** --> Script to measure the fidelity score of explanations on the Speech Commands and TESS datasets.\
**[models](models/)** --> Folder containing classification models that predict on encoded datasets using EnCodec.\
**[sample_explanations](sample_explanations/)** --> Sample audio explanations generated by AudioXgen.

## [EnCodec](https://github.com/facebookresearch/encodec) Installation
Encodec has now been added to Transformers. For more information, please refer to [Transformers' Encodec docs](https://huggingface.co/docs/transformers/main/en/model_doc/encodec).

You can find both the [24KHz](https://huggingface.co/facebook/encodec_24khz) and [48KHz](https://huggingface.co/facebook/encodec_48khz) checkpoints on the 🤗 Hub.

## Citation

Please cite [our paper](https://arxiv.org/pdf/2410.07530).
```
@misc{akman2024audioexplanationsynthesisgenerative,
      title={Audio Explanation Synthesis with Generative Foundation Models}, 
      author={Alican Akman and Qiyang Sun and Björn W. Schuller},
      year={2024},
      eprint={2410.07530},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2410.07530}, 
}
```
