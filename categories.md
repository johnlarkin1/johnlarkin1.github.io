---
layout: page
title: Categories
permalink: /categories/
---

<div>
  {% assign categories = site.categories | sort %}
  {% assign favorites_category = site.categories["⭐️ Favorites"] %}
  {% assign other_categories = categories | where_exp: "category", "category[0] != '⭐️ Favorites'" %}

{% if favorites_category %}

<div class="archive-group">
<div id="#{{ "⭐️ Favorites" | slugize }}"></div>
<p></p>
<h3 class="category-head favorites-category">⭐️ Favorites</h3>
<a name="{{ "⭐️ Favorites" | slugize }}"></a>
{% for post in favorites_category %}
<article class="archive-item">
<h4><a href="{{ site.baseurl }}{{ post.url }}">{{post.title}}</a></h4>
</article>
{% endfor %}
</div>
{% endif %}

{% for category in other_categories %}

  <div class="archive-group">
    {% capture category_name %}{{ category | first }}{% endcapture %}
    <div id="#{{ category_name | slugize }}"></div>
    <p></p>
    <h3 class="category-head">{{ category_name }}</h3>
    <a name="{{ category_name | slugize }}"></a>
    {% for post in site.categories[category_name] %}
    <article class="archive-item">
      <h4><a href="{{ site.baseurl }}{{ post.url }}">{{post.title}}</a></h4>
    </article>
    {% endfor %}
  </div>
  {% endfor %}
</div>
