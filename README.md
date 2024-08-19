# Evaluation of Speech Embedding Methods

This repository contains the codes used in the paper: [From Raw Speech to Fixed Representations: A Comprehensive Evaluation of Speech Embedding Techniques](https://doi.org/10.1109/TASLP.2024.3426301)

The small-scale models are:

* Audio word2vec
* Speech2vec
* LEE
* Siamese

In the root directory are the scripts for pre-training them, while in the task-specific directories (gender_id and emotion_id) are the codes for fine-tuning them. These models are implemented in Pytorch.

The task-specific directories also contain scripts for fine-tuning the self-supervised models used in this study. We only provide the codes for the base Wav2vec2 model and Whisper because the code is almost the same for the rest of the models.
To use another pre-trained, self-supervised model, modify this line `wav2vec2_hub: facebook/wav2vec2-base` in the `hyperparams.yaml` file with the link to the model. These models are implemented using the [SpeechBrain toolkit](https://github.com/speechbrain/speechbrain).

The repository also contained scripts for running the Integrated Gradients algorithm to select the 10% of most important dimensions.

The following self-supervised models are used in the study:

1. English:
   * [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base)
   * [facebook/wav2vec2-large](https://huggingface.co/facebook/wav2vec2-large)
   * [facebook/wav2vec2-large-960h-lv60-self](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self)
   * [facebook/hubert-base-ls960](https://huggingface.co/facebook/hubert-base-ls960)
   * [facebook/hubert-large-ll60k](https://huggingface.co/facebook/hubert-large-ll60k)
   * [facebook/hubert-large-ls960-ft](https://huggingface.co/facebook/hubert-large-ls960-ft)
   * [microsoft/wavlm-base-plus](https://huggingface.co/microsoft/wavlm-base-plus)
   * [microsoft/wavlm-large](https://huggingface.co/microsoft/wavlm-large)
   * [patrickvonplaten/wavlm-libri-clean-100h-large](https://huggingface.co/patrickvonplaten/wavlm-libri-clean-100h-large)
   * [openai/whisper-small.en](https://huggingface.co/openai/whisper-small.en)

2. Finnish:
   * [facebook/wav2vec2-large-uralic-voxpopuli-v2](https://huggingface.co/facebook/wav2vec2-large-uralic-voxpopuli-v2)
   * [Finnish-NLP/wav2vec2-xlsr-300m-finnish-lm](https://huggingface.co/Finnish-NLP/wav2vec2-xlsr-300m-finnish-lm)
   * [openai/whisper-small](https://huggingface.co/openai/whisper-small)

3. French:
   * [facebook/wav2vec2-base-fr-voxpopuli-v2](https://huggingface.co/facebook/wav2vec2-base-fr-voxpopuli-v2)
   * [facebook/wav2vec2-large-fr-voxpopuli](https://huggingface.co/facebook/wav2vec2-large-fr-voxpopuli)
   * [jonatasgrosman/wav2vec2-large-xlsr-53-french](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-french)
   * [jonatasgrosman/exp_w2v2t_fr_hubert_s990](https://huggingface.co/jonatasgrosman/exp_w2v2t_fr_hubert_s990)
   * [jonatasgrosman/exp_w2v2t_fr_wavlm_s929](https://huggingface.co/jonatasgrosman/exp_w2v2t_fr_wavlm_s929)
   * [openai/whisper-small](https://huggingface.co/openai/whisper-small)
