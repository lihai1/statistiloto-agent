# Hot and Cold Numbers (מספרים חמים וקרים)

## What "hot" and "cold" mean

In Statistiloto, "hot" and "cold" describe how often a number or group of numbers
has appeared in past draws within a chosen archive window.

- **Hot (חם)**: numbers or groups that appeared **more frequently** in the selected
  historical period. In the system this maps to `strength=STRONG`.
- **Cold (קר)**: numbers or groups that appeared **less frequently** in the selected
  historical period. In the system this maps to `strength=WEAK`.

## Important disclaimer

Historical frequency is **not** winning probability. A number that appeared often in
the past has the same chance of being drawn next time as any other number, because
each draw is independent.

- Hebrew: אלה נתוני עבר בלבד; הם לא מעידים שלקבוצה מסוימת יש סיכוי גבוה יותר בהגרלה הבאה.
- English: These are historical observations only and do not imply a higher probability
  in the next draw.

## How to use hot/cold

Hot/cold is a **view** on top of the generalized group-size statistics. You can look at:

- Hot single numbers (group_size=1, strength=hot)
- Cold single numbers (group_size=1, strength=cold)
- Hot pairs (group_size=2, strength=hot)
- Cold pairs (group_size=2, strength=cold)
- Hot triples (group_size=3, strength=hot)
- Cold triples (group_size=3, strength=cold)
- ...and so on up to group_size=6.

The archive window (טווח הגרלות) controls which historical draws are included.
A narrower window (e.g. last 100 draws) may show different hot/cold patterns than
the full archive.
