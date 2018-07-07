---
layout: page
title: Blog
permalink: /blog
---

More coming soon, hopefully.

{% for post in site.posts %}
  {% if post.visible %}
  * {{ post.date | date_to_string }} &raquo; [ {{ post.title }} ]({{ post.url }})
  {% endif %}
{% endfor %}

<!-- I sometimes post on <a href="https://medium.com/@alex_yang">Medium</a>, more posts will be added soon. -->
