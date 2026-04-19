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
      filters:
        folders:
          - interests
        tags: Interests
        kinds:
          - section
    design:
      view: article-grid
      show_read_time: false
      show_date: false
      show_read_more: false
      columns: 1
---
