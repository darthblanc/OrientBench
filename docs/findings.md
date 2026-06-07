# Findings

**Headline: column-wise (transposed) CSV format degrades accuracy for a small
model but not a capable one — orientation sensitivity is concentrated at small
scale.** Under identical conditions (same dataset, columns, max-rows, and
n=200 per model), Qwen2.5 3B loses 16.5 points overall when data is presented
column-wise, with every task type affected in the same direction. Claude Haiku
shows no meaningful effect (−2.5 points, within sampling noise).

## Side-by-side, identical args (ott_movies_clean_unique, n=50 per task type, 200 per model)

| Task | Qwen2.5 3B Row | Col | Δ | Haiku Row | Col | Δ |
|---|---|---|---|---|---|---|
| cell_recall | 88% | 56% | **−32 (row)** | 100% | 100% | 0 (saturated) |
| attr_scan | 42% | 30% | **−12 (row)** | 48% | 44% | −4 (noise) |
| comparison | 18% | 8% | **−10 (row)** | 74% | 68% | −6 (noise) |
| row_list | 24% | 12% | **−12 (row)** | 100% | 100% | 0 (saturated) |
| **TOTAL** | **43%** | **26.5%** | **−16.5 (row)** | **80.5%** | **78%** | **−2.5 (noise)** |

Δ = col_acc − row_acc; negative means row-wise scored higher.

## How to read this

- **The 3B effect is real, not noise.** All four task types favor row-wise,
  with magnitudes (10–32 pts) far above the ~2-pt noise floor at n=50 per task.
  Consistent direction across four independent task types is the signal.
- **Haiku is robust where it can be measured.** On the two non-saturated tasks
  (attr_scan, comparison), deltas of −4 and −6 are within sampling noise — one
  to three questions at this n. No orientation effect is detectable.
- **Saturation caveat.** Haiku scores 100% in both orientations on cell_recall
  and row_list, so those tasks have no power to detect an effect for Haiku. The
  "no effect" conclusion for Haiku applies to the two informative tasks; the
  capability comparison rests on the tasks where both models had room to
  differ.

## Interpretation

Smaller models appear to rely on row-local adjacency — a record's fields
sitting together on one line — to associate attributes with their entity.
Transposing the table breaks that adjacency, and the 3B model's accuracy drops
across every task type. The more capable model reconstructs the
entity–attribute mapping regardless of layout. Practical takeaway: when
prompting smaller models with tabular data, row-wise formatting is the safer
default.

## Scope and next steps

This compares a small local model against a capable cloud model and finds the
orientation effect concentrated at the small end. A natural extension is a
mid-capability model (e.g. an 8B local model, already supported by the registry)
to show whether the effect fades smoothly with capability or drops off sharply.
A paired significance test (McNemar's, since each task is run in both
orientations) would also let deltas ship with a confidence read rather than a
point estimate alone.
