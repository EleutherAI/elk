---
title: 'elk: A Python package to elicit latent knowledge from LLMs'
tags:
  - python
  - machine leaarning
  - interpretability
  - ai alignment
  - honest AI
authors:
  - name: Nora Belrose
    affiliation: 1 
  - name: Walter Laurito
    corresponding: true
    affiliation: 2
  - name: Alex Mallen
    affiliation: 1
  - name: Fabien Roger
    affiliation: 3
  - name: Kay Kozaronek
    affiliation: 4
  - name: Christy Koh
    affiliation: 5
  - name: Jonathan NG
    affiliation: 6
  - name: Reagan Lee
    affiliation: 7
  - name: Alex Wan
    affiliation: 8
affiliations:
 - name: EleutherAI
   index: 1
 - name: FZI Research Center for Information Technology, Germany
   index: 2
 - name: Redwood Research
   index: 3
date: 31 July 2023
bibliography: paper.bib

---

# Summary

`elk` is a library designed to elicit latent knowledge ([elk](`https://docs.google.com/document/d/1WwsnJQstPq91_Yh-Ch2XRL8H_EpsnjrC1dwZXR37PC8/`) [@author:elk]) from language models. It includes implementations of both the original and an enhanced version of the CSS method, as well as an approach based on the CRC method [@author:burns]. Designed for researchers, `elk` offers features such as multi-GPU support, integration with Huggingface, and continuous improvement by a dedicated group of people. The Eleuther AI Discord's `elk` channel provides a platform for collaboration and discussion related to the library and associated research.

# Statement of need

Language models are proficient at predicting successive tokens in a sequence of text. However, they often inadvertently mirror human errors and misconceptions, even when equipped with the capability to "know better." This behavior becomes particularly concerning when models are trained to generate text that is highly rated by human evaluators, leading to the potential output of erroneous statements that may go undetected. Our solution is to directly elicit latent knowledge (([elk](`https://docs.google.com/document/d/1WwsnJQstPq91_Yh-Ch2XRL8H_EpsnjrC1dwZXR37PC8/edit`) [@author:elk]) from within the activations of a language model to mitigate this challenge.

`elk` is a specialized library developed to provide both the original and an enhanced version of the CSS methodology. Described in the paper "Discovering Latent Knowledge in Language Models Without Supervision" by Burns et al. [@author:burns]. In addition, we have implemented an approach based on the Contrastive Representation Clustering (CRC) method (2022) from the same paper.

`elk` serves as a tool for those seeking to investigate the veracity of model output and explore the underlying beliefs embedded within the model. The library offers:

- Multi-GPU Support: Efficient extraction, training, and evaluation through parallel processing.
- Integration with Huggingface: Easy utilization of models and datasets from a popular source.
- Active Development and Support: Continuous improvement by a dedicated team of researchers and engineers.

For collaboration, discussion, and support, the [Eleuther AI Discord's elk channel](https://discord.com/channels/729741769192767510/1070194752785489991) provides a platform for engaging with others interested in the library or related research projects.

# Acknowledgements
We would like to thank [SERI MATS](https://www.serimats.org/) and [EleutherAI](https://www.eleuther.ai/) for supporting our work.