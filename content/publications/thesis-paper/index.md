<!-- ---
title: 'Thesis Paper for Review'

# Authors
# If you created a profile for a user (e.g. the default `me` user), write the username (folder name) here
# and it will be replaced with their full name and linked to their profile.
authors:
  - me

date: '2026-04-01T00:00:00Z'

# Schedule page publish date (NOT publication's date).
publishDate: '2026-04-01T00:00:00Z'

# Publication type.
# Accepts a single type but formatted as a YAML list (for Hugo requirements).
# Enter a publication type from the CSL standard.
publication_types: ['thesis']

# Publication name and optional abbreviated publication name.
publication: In *Tampere University*
publication_short: In *TUNI*

abstract: Most speaker recognition systems rely on spectral features such as MFCCs, which treat speech production as a linear process. Under clean acoustic conditions, this assumption holds, but the spectral envelope that these features measure is directly disrupted by additive noise, causing recognition performance to degrade sharply in real-world environments. This thesis proposes ChaoNet, a speaker recognition framework based on the premise that a nonlinear, chaotic system generates speech. Instead of extracting spectral features, ChaoNet reconstructs the phase space of each speech frame and extracts chaotic features through Multi-scale Lyapunov Spectrum Analysis (MLSA) and Recurrence Quantification Analysis (RQA). These features describe the dynamic behavior of the vocal tract, which remains more stable under noise than the spectral envelope. The extracted features drive a chaotic embedding layer based on controlled Lorenz neurons, in which speaker-specific information is encoded in the geometry of strange attractors. A bifurcation control mechanism allows the system to adapt to different dynamical regimes for different speakers. Experiments on LibriSpeech show that ChaoNet achieves 94–97% accuracy under clean conditions with 26 speakers and 92.75% with 251 speakers. The main advantage appears under noisy conditions - at 20 dB SNR, ChaoNet retains an average of 82.6% of its clean performance across four noise types, compared to 43.6% for MFCC-MLP and 34.8% for Mel-MLP. This advantage comes at a cost - ChaoNet has about 8 times as many parameters and takes about 7 times longer to train than the baseline. Ablation studies show that attractor pooling contributes the most to performance, while the phase synchronization loss, despite being theoretically motivated, consistently hurts accuracy. This failure reveals a fundamental mismatch - chaotic systems amplify input differences through sensitive dependence on initial conditions, so different utterances from the same speaker produce divergent trajectories regardless of speaker identity. This finding indicates that chaos synchronization theory cannot be directly transferred to feedforward architectures without fundamental redesign.

# Summary. An optional shortened abstract.
summary: This page for reviewing the paper. ChaoNet - A novel speaker recognition framework integrating chaos theory with deep neural networks.

tags:
  - Chaos Theory
  - Speaker Recognition
  - Audio Processing
  - Signal Processing
  - MLSA & RQA
  - Chaotic Features
  - Chaotic Neural Networks

# Display this page in the Featured widget?
featured: true

# Standard identifiers for auto-linking
hugoblox:
  ids:
    doi: 10.5555/123456

# Custom links
links:
  - type: pdf
    url: /uploads/thesis-paper.pdf
  - type: code
    url: https://github.com/HugoBlox/kit
  - type: dataset
    url: https://github.com/HugoBlox/kit
  - type: slides
    url: https://www.slideshare.net/
  - type: source
    url: https://github.com/HugoBlox/kit
  - type: video
    url: https://youtube.com

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
image:
  caption: 'RQA Comparison'
  focal_point: 'Smart'
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects:
  - example

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
slides: ""
---
 -->
