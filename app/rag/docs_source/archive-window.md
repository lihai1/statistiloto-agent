# Archive Window (טווח הגרלות)

## What the archive window is

The archive window defines **which historical draws** are included in a statistics
or analysis query. By default, the entire archive is used (from 2004-02-12 to today).

## How to set the window

- **From date** (מ-): only include draws on or after this date.
- **To date** (עד): only include draws on or before this date.

Both are optional. You can set just "from" (e.g. "last 2 years"), just "to"
(e.g. "everything before 2020"), or both.

## Why the window matters

Hot/cold patterns can change depending on the window:

- A number that was "hot" in 2005-2010 might be "cold" in 2020-2025.
- A narrow window (e.g. last 50 draws) shows recent trends.
- A wide window (e.g. full archive) shows long-term patterns.

## Common windows

| Question | Window |
|---|---|
| "All-time hot pairs" | Full archive (no window) |
| "Hot pairs in the last year" | from = one year ago, to = today |
| "Hot pairs in the last 100 draws" | from = date of 100th-most-recent draw |
| "Patterns before 2015" | to = 2014-12-31 |

## API mapping

In the system, the archive window maps to the `DateWindow` proto field with
`from` and `to` timestamps. Both the Statistics and Analyze tools accept it.
