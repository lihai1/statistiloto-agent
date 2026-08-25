# Analyze vs Statistics (ניתוח מול סטטיסטיקה)

Statistiloto has two related but distinct capabilities for exploring number
frequencies. Understanding the difference helps you ask the right question.

## Statistics (סטטיסטיקה)

**What it does**: Ranks the most or least frequent number groups of a given size
across the entire archive (or a date window).

**Use when**: You want to know "What are the hottest pairs in the last 100 draws?"
or "What are the coldest triples overall?"

**Parameters**:
- `group_size` (1-6): the size of the groups to rank
- `strength` (hot/cold): whether to show most or least frequent
- `how_many`: how many top/bottom results to return
- `window` (optional): date range to filter the archive

**Returns**: A ranked list of groups with their appearance counts.

## Analyze (ניתוח)

**What it does**: Takes **your** selected numbers and shows how often every subset
of those specific numbers has appeared together historically.

**Use when**: You want to know "I picked 7, 11, 17, 24, 31, 36 — how often has this
combination or parts of it appeared?" or "What stands out in my numbers?"

**Parameters**:
- `form`: your selected numbers (1-6 regular numbers)
- `window` (optional): date range to filter the archive

**Returns**: Frequency groups for all subset sizes 1-6 of your numbers, plus the
total archive size used.

## Key difference

- **Statistics** answers: "What's common across ALL draws?"
- **Analyze** answers: "How do MY specific numbers look historically?"

## When to use which

| Question | Use |
|---|---|
| "What are the hot pairs?" | Statistics (group_size=2, hot) |
| "What are the cold triples?" | Statistics (group_size=3, cold) |
| "Analyze my numbers 7,11,17,24,31,36" | Analyze |
| "What stands out in 3,15,22,29?" | Analyze |
| "Most frequent single numbers" | Statistics (group_size=1, hot) |
| "Has this six-number group appeared?" | Analyze |
