---
title: "Skills"
layout: archive
permalink: /skills
---


{% assign posts = site.categories.Skills %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %}
