# Group Sizes (גדלי קבוצות)

Statistiloto analyzes lottery numbers by **group size** — how many numbers appear
together in a combination. This is a single generalized concept, not separate
features.

## Group size reference

| Group size | Hebrew | English | Description |
|---|---|---|---|
| 1 | מספרים בודדים | singles / numbers | Individual number frequency |
| 2 | זוגות | pairs | Two numbers appearing together |
| 3 | שלשות / שלישיות | triples | Three numbers appearing together |
| 4 | רביעיות | quads / quadruples | Four numbers appearing together |
| 5 | חמישיות | quints / quintuples | Five numbers appearing together |
| 6 | שישיות | six-number groups | Full six-number combination (excluding strong) |

## The strong number

The **strong number** (מספר חזק) is always **separate** from the six-number group.
It is never mixed into group-size analysis of the regular numbers. When you analyze
a form with 7 numbers (6 regular + 1 strong), the system strips the strong number
and analyzes only the 6 regular numbers.

## How group size is used

- **Statistics page**: choose group size 1-6 to see frequency rankings.
- **Analyze page**: enter your numbers; the system returns frequency groups for all
  subset sizes 1-6 of your selection.
- **Generate page**: the system generates six-number combinations (group_size=6).

## API mapping

In the system, `group_size` maps to the `form_type` parameter in the Go lottery
service. Both refer to the same concept: the size of the number group being analyzed.
