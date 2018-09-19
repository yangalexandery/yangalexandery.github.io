---
layout: page
title: Workbook
permalink: /workbook
---

Every so often, I write-up things I learn and post them here. This can be considered a 'blog' of sorts, but I prefer the term 'workbook' since this is done more for my own understanding than for the reader's.

{% for post in site.posts %}
  {% if post.visible %}
  * {{ post.date | date_to_string }} &raquo; [ {{ post.title }} ]({{ post.url }})
  {% endif %}
{% endfor %}

<!-- I sometimes post on <a href="https://medium.com/@alex_yang">Medium</a>, more posts will be added soon. -->
