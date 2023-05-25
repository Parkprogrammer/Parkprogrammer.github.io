---
title: "Introduction"
layout: archive
permalink: /introduction
---

# Basic 
### Born in **2000 December** in **Seoul**
### Currently living in **Suwon**

# Favorite quote
>> What doesn't kill you makes you stronger

{% assign posts = site.categories.introduction %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %}
