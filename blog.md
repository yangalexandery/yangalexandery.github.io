---
layout: page
title: Workbook
permalink: /workbook
---

In the summer of 2018, I decided to write up a few things I learned to further my understanding of them, which you can view here.

{% for post in site.posts %}
  {% if post.visible %}
  * {{ post.date | date_to_string }} &raquo; [ {{ post.title }} ]({{ post.url }})
  {% endif %}
{% endfor %}

<!-- I sometimes post on <a href="https://medium.com/@alex_yang">Medium</a>, more posts will be added soon. -->
