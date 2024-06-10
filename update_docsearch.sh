#!/bin/bash

# Set environment variables
export APPLICATION_ID=S9NC0RYCHF
export API_KEY=842757a059db8a232231828803688f96

# Create config.json content as a string
CONFIG=$(cat <<EOF
{
  "index_name": "docusaurus-algolia",
  "start_urls": ["https://docsaid.org/"],
  "sitemap_urls": ["https://docsaid.org/sitemap.xml"],
  "sitemap_alternate_links": true,
  "stop_urls": [
    "/tests",
    "/search",
    ".*/#.*"
  ],
  "selectors": {
    "lvl0": {
      "selector": "(//ul[contains(@class,'menu__list')]//a[contains(@class, 'menu__link menu__link--sublist menu__link--active')]/text() | //nav[contains(@class, 'navbar')]//a[contains(@class, 'navbar__link--active')]/text())[last()]",
      "type": "xpath",
      "global": true,
      "default_value": "Documentation"
    },
    "lvl1": "header h1, article h1",
    "lvl2": "article h2",
    "lvl3": "article h3",
    "lvl4": "article h4",
    "lvl5": "article h5, article td:first-child",
    "lvl6": "article h6",
    "text": "article p, article li, article td:last-child"
  },
  "custom_settings": {
    "separatorsToIndex": "_",
    "attributesForFaceting": [
      "filterOnly(language)",
      "filterOnly(version)",
      "filterOnly(type)",
      "filterOnly(docusaurus_tag)"
    ],
    "attributesToRetrieve": [
      "hierarchy",
      "content",
      "anchor",
      "url",
      "url_without_anchor",
      "type"
    ]
  },
  "searchableAttributes": [
    "hierarchy.lvl0",
    "hierarchy.lvl1",
    "hierarchy.lvl2",
    "hierarchy.lvl3",
    "hierarchy.lvl4",
    "hierarchy.lvl5",
    "hierarchy.lvl6",
    "content"
  ]
}
EOF
)

# Run the Docker command
docker run -it --env APPLICATION_ID --env API_KEY -e "CONFIG=$CONFIG" algolia/docsearch-scraper
