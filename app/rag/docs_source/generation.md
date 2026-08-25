# Smart Form Generation (יצירת טפסים חכמה)

## What smart generation does

Statistiloto can generate lottery forms (six-number combinations) using the
lottery-tree algorithm. The generation considers historical frequency patterns
to produce combinations.

## How it works

1. The lottery-tree algorithm builds a frequency tree from the selected archive.
2. It ranks numbers and groups by historical frequency.
3. It generates combinations that are **non-winning** — i.e. combinations that
   have not appeared as a winning draw in the historical archive.
4. You can include "lucky numbers" (will_be) that must appear in every generated
   form.

## Parameters

- `how_many`: number of forms to generate.
- `form_type`: the systematic form size (default 6 = regular lotto).
- `will_be`: lucky numbers to include in every form.
- `strength`: whether to bias toward frequent (hot) or infrequent (cold) numbers.
- `window`: optional archive window for the frequency tree.

## Important note

Generated forms are based on historical patterns but, like all statistics in
Statistiloto, they do **not** have a higher probability of winning. Each draw is
independent and random. The generation feature is an exploration tool, not a
prediction.

## Strong number

Generated forms include a separate strong number (מספר חזק) when applicable.
The strong number is not part of the six-number group frequency analysis.
