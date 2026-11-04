---
name: poster-critique
description: How to inspect a rendered poster - deterministic probes first, then what a small local vision model can and cannot judge - and which typed fix each flaw maps to.
priority: 4
use: [critic]
---
# Poster critique

## Order of inspection <!-- core -->
1. Deterministic measurements first: OCR of every placed text, contrast at the worst pixels, margins, logo rules, dead space, hierarchy
   ratio. These are facts and decide pass or fail.
2. Glance test with the local vision model (LFM3-VL-2B): it reads the title and date from a 300 px thumbnail; the reading is compared with
   the brief. Unreadable means the hierarchy failed.
3. Pairwise preference between two designs of the same brief: the model sees both orders; only an answer that survives the swap counts,
   otherwise the pair is a tie. Used to rank variants.
A small model is not asked "is this good?": tested on plain and designed posters alike it answered yes to everything.

## Flaw to typed fix <!-- core -->
| Flaw | Symptom | Fix (typed, allow-listed) |
|---|---|---|
| title too small | title not read at 300 px | text.resize title up (limit 15 %), or change the plan to a BigTitle |
| centre void | large empty band | move content block (flow_valign), enlarge hero or title |
| low contrast | text under 4.5:1 | recolour to the brand colour that reads; add a soft scrim behind the block |
| colour clash | more than one accent hue | recolour to the structural tone or the single accent |
| margin overflow | text within the safe margin | text.resize down |
| flat page | looks like a text document | add structure (band, tab, columns, icon row) or art |

## Honesty
Report what was measured or read. A reading the model could not make is "unknown", never a pass. No invented scores.
