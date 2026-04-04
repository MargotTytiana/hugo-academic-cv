---
title: "Master Thesis Working Paper"
authors:
- me
date: "2026-03-31T00:00:00Z"

# Schedule page publish date (NOT publication's date).
publishDate: "2026-03-31T00:00:00Z"

# Publication type.
# Accepts a single type but formatted as a YAML list (for Hugo requirements).
# Enter a publication type from the CSL standard.
publication_types: ["article"]

# Publication name and optional abbreviated publication name.
publication: ""
publication_short: ""

abstract: Most speaker recognition systems rely on spectral features such as MFCCs, which treat speech production as a linear process. Under clean acoustic conditions, this assumption holds, but the spectral envelope that these features measure is directly disrupted by additive noise, causing recognition performance to degrade sharply in real-world environments. This thesis proposes ChaoNet, a speaker recognition framework based on the premise that a nonlinear, chaotic system generates speech. Instead of extracting spectral features, ChaoNet reconstructs the phase space of each speech frame and extracts chaotic features through Multi-scale Lyapunov Spectrum Analysis (MLSA) and Recurrence Quantification Analysis (RQA). These features describe the dynamic behavior of the vocal tract, which remains more stable under noise than the spectral envelope. The extracted features drive a chaotic embedding layer based on controlled Lorenz neurons, in which speaker-specific information is encoded in the geometry of strange attractors. A bifurcation control mechanism allows the system to adapt to different dynamical regimes for different speakers. Experiments on LibriSpeech show that ChaoNet achieves 94–97% accuracy under clean conditions with 26 speakers and 92.75% with 251 speakers. The main advantage appears under noisy conditions - at 20 dB SNR, ChaoNet retains an average of 82.6% of its clean performance across four noise types, compared to 43.6% for MFCC-MLP and 34.8% for Mel-MLP. This advantage comes at a cost - ChaoNet has about 8 times as many parameters and takes about 7 times longer to train than the baseline. Ablation studies show that attractor pooling contributes the most to performance, while the phase synchronization loss, despite being theoretically motivated, consistently hurts accuracy. This failure reveals a fundamental mismatch - chaotic systems amplify input differences through sensitive dependence on initial conditions, so different utterances from the same speaker produce divergent trajectories regardless of speaker identity. This finding indicates that chaos synchronization theory cannot be directly transferred to feedforward architectures without fundamental redesign.

# Summary. An optional shortened abstract.
summary: A novel speaker recognition framework integrating chaos theory with deep neural networks.

tags:
- Chaos Theory
- Speaker Recognition
- Audio Processing
- Signal Processing
- MLSA & RQA
- Chaotic Features
- Chaotic Neural Networks

featured: true

hugoblox:
  ids:
    arxiv: 1512.04133v1

links:
- type: working paper
  provider: Tampere University
  id: 1512.04133v1
- type: code
  url: https://github.com/MargotTytiana
- type: slides
  url: https://drive.google.com/file/d/1RqtpSIna04B5tagm4lh_uFQe-_QmSvxd/view?usp=drive_link
- type: dataset
  url: https://www.openslr.org/12/
- type: poster
  url: "#"
- type: source
  url: "#"
- type: video
  url: https://youtube.com
- type: custom
  label: Custom Link
  url: http://example.org

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
image:
  caption: 'RQA Comparison'
  focal_point: ""
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects:
- internal-project

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
slides: ""
---

This work is driven by the results in my [previous paper](/publications/conference-paper/) on LLMs.

> [!NOTE]
> Create your slides in Markdown - click the *Slides* button to check out the example.

Add the publication's **full text** or **supplementary notes** here. You can use rich formatting such as including [code, math, and images](https://docs.hugoblox.com/content/writing-markdown-latex/).
