---
title: 'Projects'
date: 2024-05-19
type: landing

design:
  # Section spacing
  spacing: '5rem'

# Page sections
sections:
  - block: collection
    content:
      title: Selected Projects
      text: |
              I build things — sometimes to solve a problem, sometimes to understand one.

              My work sits at the intersection of machine learning and signal intelligence,
              with a recurring interest in the places where conventional models break down
              and something more principled is needed. I'm drawn to research that requires
              both mathematical depth and engineering discipline: designing systems that don't
              just perform well on benchmarks, but actually make sense.

              The projects here span audio, vision, data, and beyond —
              connected less by domain than by a shared instinct:
              that the most interesting problems are the ones nobody has cleanly solved yet.
      filters:
        folders:
          - projects
    design:
        view: article-grid
        fill_image: false
        columns: 3
        show_date: false
        show_read_time: false
        show_read_more: false

  - block: collection
    content:
      title: Audio & Speech
      text: >
        Representation learning, speaker modeling, and generative systems for audio.
        From classical signal processing pipelines to end-to-end neural architectures —
        with a particular interest in what happens when the two are forced to coexist.
      filters:
        folders:
          - projects
    design:
        view: article-grid
        fill_image: false
        columns: 3
        show_date: false
        show_read_time: false
        show_read_more: false

  - block: collection
    content:
      title: Computer Vision
      text: >
        Visual understanding across modalities — image classification, segmentation,
        and multi-sensor fusion. Projects here tend to ask: what does a model actually
        learn, and is it learning the right thing?
      filters:
        folders:
          - projects
    design:
        view: article-grid
        fill_image: false
        columns: 3
        show_date: false
        show_read_time: false
        show_read_more: false

  - block: collection
    content:
      title: Sequence & Language Models
      text: >
        Transformers, attention, and everything downstream. Work in this area explores
        both the capabilities and the failure modes of large-scale sequential models —
        including retrieval-augmented systems and domain-specific fine-tuning.
      filters:
        folders:
          - projects
    design:
        view: article-grid
        fill_image: false
        columns: 3
        show_date: false
        show_read_time: false
        show_read_more: false

  - block: collection
    content:
      title: Multimodal Learning
      text: >
        Systems that reason across more than one modality simultaneously.
        Audio-visual correspondence, cross-modal retrieval, and the open question
        of how to align representations that were never designed to talk to each other.
      filters:
        folders:
          - projects
    design:
        view: article-grid
        fill_image: false
        columns: 3
        show_date: false
        show_read_time: false
        show_read_more: false

  - block: collection
    content:
      title: ML Systems & Infrastructure
      text: >
        The part nobody talks about until it breaks. Efficient training pipelines,
        experiment tracking, and deployment-aware model design.
        Because a model that can't run in production isn't a model — it's a paper.
      filters:
        folders:
          - projects
    design:
        view: article-grid
        fill_image: false
        columns: 3
        show_date: false
        show_read_time: false
        show_read_more: false



---

