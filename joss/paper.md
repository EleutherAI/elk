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
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 2
  - name: Alex Mallen
    affiliation: 1
  - name: Author with no affiliation
    affiliation: 3
affiliations:
 - name: EleutherAI
   index: 1
 - name: FZI Research Center for Information Technology, Germany
   index: 2
 - name: EleutherAI
   index: 3
date: 13 August 2017
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

`elk` is a library designed to elicit latent knowledge ([ELK](`https://docs.google.com/document/d/1WwsnJQstPq91_Yh-Ch2XRL8H_EpsnjrC1dwZXR37PC8/edit`) [@author:elk]) from language models. It includes implementations of both the original and an enhanced version of the CSS method, as well as an approach based on the CRC method. Designed for researchers, `elk` offers features such as multi-GPU support, integration with Huggingface, and continuous improvement by a dedicated team. The Eleuther AI Discord's `elk` channel provides a platform for collaboration and discussion related to the library and associated research.

# Statement of need

Language models are proficient at predicting successive tokens in a sequence of text. However, they often inadvertently mirror human errors and misconceptions, even when equipped with the capability to "know better." This behavior becomes particularly concerning when models are trained to generate text that is highly rated by human evaluators, leading to the potential output of erroneous statements that may go undetected. Our solution is to directly Elicit Latent Knowledge (ELK) from within the activations of a language model to mitigate this challenge.

`elk` is a specialized library developed to provide both the original and an enhanced version of the CSS methodology. Described in the paper "Discovering Latent Knowledge in Language Models Without Supervision" by Burns et al. [@author:burns], the CSS method has been instrumental in our understanding of language models. In addition, we have implemented an approach based on the Contrastive Representation Clustering (CRC) method (2022) from the same paper. The CRC technique allows for the discovery of features in the hidden states of a language model that adhere to specific logical consistency requirements. Interestingly, these features have proven to be highly effective for question-answering and text classification tasks, even when trained without labels.

Designed with the research community in mind, elk serves as a powerful tool for those seeking to investigate the veracity of model output and explore the underlying beliefs embedded within the model. The library offers:

Multi-GPU Support: Efficient extraction, training, and evaluation through parallel processing.
Integration with Huggingface: Easy utilization of models and datasets from a popular source.
Active Development and Support: Continuous improvement by a dedicated team of researchers and engineers.

For collaboration, discussion, and support, the [Eleuther AI Discord's elk channel](https://discord.com/channels/729741769192767510/1070194752785489991) provides a platform for engaging with others interested in the library or related research projects.


# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"


# Acknowledgements


# References