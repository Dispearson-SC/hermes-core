---
name: malformed-skill
description: [this is not valid yaml: because of a stray colon: in a list
---

# Malformed skill

This body should never reach the index -- only the discovery scan must survive
parsing this frontmatter without crashing.
