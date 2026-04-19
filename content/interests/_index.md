---
title: Interests
summary: My Interests
type: landing

cascade:
  - target:
      path: '{/interests/*/**}'
    type: docs
    params:
      show_breadcrumb: true

sections:
  - block: collection
    id: interests
    content:
      title: Interests
      text: |
        Here are the things that I care about.
      filters:
        folders:
          - interests
        tags: Interests
        kinds:
          - section
    design:
        view: article-grid
        fill_image: false
        columns: 3
        show_date: false
        show_read_time: false
        show_read_more: false
---
