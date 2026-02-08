---
layout: page
title: Categories
featured-img: namibia
featured-img-caption: Floating over the Namib Desert in May 2024 with my BIL, sis, and Dad. First hot air balloon ride!
permalink: /categories/
---

{% assign categories = site.categories | sort %}

<div class="categories-grid" id="categories-grid">

  {% comment %} ── Favorites row first ── {% endcomment %}
  {% assign fav_posts = site.categories["⭐️ Favorites"] %}
  {% if fav_posts %}
  {% assign fav_latest = fav_posts.first %}
  {% assign fav_oldest = fav_posts.last %}
  <div class="categories-grid__row categories-grid__row--favorites"
       tabindex="0" role="button">
    <div class="categories-grid__row-header" aria-expanded="false">
      <span class="categories-grid__row-icon">⭐️</span>
      <h3 class="categories-grid__row-name">Favorites</h3>
      <div class="categories-grid__row-meta">
        <span class="categories-grid__row-stat">{{ fav_posts.size }} post{% if fav_posts.size != 1 %}s{% endif %}</span>
        <span class="categories-grid__row-separator">&middot;</span>
        <span class="categories-grid__row-stat categories-grid__row-stat--first">{{ fav_oldest.date | date: "%b %Y" }}</span>
        <span class="categories-grid__row-separator">&middot;</span>
        <span class="categories-grid__row-stat categories-grid__row-stat--latest">{{ fav_latest.date | date: "%b %Y" }}</span>
      </div>
      <svg class="categories-grid__chevron" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true"><path fill-rule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z" clip-rule="evenodd"/></svg>
    </div>
    <div class="categories-grid__posts" aria-hidden="true">
      <ul class="categories-grid__post-list">
        {% for post in fav_posts %}
        <li class="categories-grid__post-item">
          <a class="categories-grid__post-link" href="{{ site.baseurl }}{{ post.url }}">
            <span class="categories-grid__post-title">{{ post.title }}</span>
            <time class="categories-grid__post-date">{{ post.date | date: "%b %d, %Y" }}</time>
          </a>
        </li>
        {% endfor %}
      </ul>
    </div>
  </div>
  {% endif %}

  {% comment %} ── Year in Review row second ── {% endcomment %}
  {% assign yir_posts = site.categories["🎉 Year in Review"] %}
  {% if yir_posts %}
  {% assign yir_latest = yir_posts.first %}
  {% assign yir_oldest = yir_posts.last %}
  <div class="categories-grid__row categories-grid__row--year-in-review"
       tabindex="0" role="button">
    <div class="categories-grid__row-header" aria-expanded="false">
      <span class="categories-grid__row-icon">🎉</span>
      <h3 class="categories-grid__row-name">Year in Review</h3>
      <div class="categories-grid__row-meta">
        <span class="categories-grid__row-stat">{{ yir_posts.size }} post{% if yir_posts.size != 1 %}s{% endif %}</span>
        <span class="categories-grid__row-separator">&middot;</span>
        <span class="categories-grid__row-stat categories-grid__row-stat--first">{{ yir_oldest.date | date: "%b %Y" }}</span>
        <span class="categories-grid__row-separator">&middot;</span>
        <span class="categories-grid__row-stat categories-grid__row-stat--latest">{{ yir_latest.date | date: "%b %Y" }}</span>
      </div>
      <svg class="categories-grid__chevron" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true"><path fill-rule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z" clip-rule="evenodd"/></svg>
    </div>
    <div class="categories-grid__posts" aria-hidden="true">
      <ul class="categories-grid__post-list">
        {% for post in yir_posts %}
        <li class="categories-grid__post-item">
          <a class="categories-grid__post-link" href="{{ site.baseurl }}{{ post.url }}">
            <span class="categories-grid__post-title">{{ post.title }}</span>
            <time class="categories-grid__post-date">{{ post.date | date: "%b %d, %Y" }}</time>
          </a>
        </li>
        {% endfor %}
      </ul>
    </div>
  </div>
  {% endif %}

  {% comment %} ── All other categories alphabetically ── {% endcomment %}
  {% for category in categories %}
    {% assign cat_name = category | first %}
    {% if cat_name == "⭐️ Favorites" or cat_name == "🎉 Year in Review" %}
      {% continue %}
    {% endif %}
    {% assign cat_posts = category | last %}
    {% assign cat_latest = cat_posts.first %}
    {% assign cat_oldest = cat_posts.last %}
    <div class="categories-grid__row"
         tabindex="0" role="button">
      <div class="categories-grid__row-header" aria-expanded="false">
        <h3 class="categories-grid__row-name">{{ cat_name }}</h3>
        <div class="categories-grid__row-meta">
          <span class="categories-grid__row-stat">{{ cat_posts.size }} post{% if cat_posts.size != 1 %}s{% endif %}</span>
          <span class="categories-grid__row-separator">&middot;</span>
          <span class="categories-grid__row-stat categories-grid__row-stat--first">{{ cat_oldest.date | date: "%b %Y" }}</span>
          <span class="categories-grid__row-separator">&middot;</span>
          <span class="categories-grid__row-stat categories-grid__row-stat--latest">{{ cat_latest.date | date: "%b %Y" }}</span>
        </div>
        <svg class="categories-grid__chevron" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true"><path fill-rule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z" clip-rule="evenodd"/></svg>
      </div>
      <div class="categories-grid__posts" aria-hidden="true">
        <ul class="categories-grid__post-list">
          {% for post in cat_posts %}
          <li class="categories-grid__post-item">
            <a class="categories-grid__post-link" href="{{ site.baseurl }}{{ post.url }}">
              <span class="categories-grid__post-title">{{ post.title }}</span>
              <time class="categories-grid__post-date">{{ post.date | date: "%b %d, %Y" }}</time>
            </a>
          </li>
          {% endfor %}
        </ul>
      </div>
    </div>
  {% endfor %}

</div>
