---
name: typography-pairing
description: Type scale, pairing, weight, tracking and case rules, and the five type_style presets this agent offers.
priority: 3
use: [plan]
digest: "One or two families; scale first, then contrast, then weight. Display leading 1.05-1.2, body 1.3-1.5; wide tracking only on small uppercase labels; all caps only for short lines; never break a number across lines. type_style: classic, bold, elegant, editorial, condensed. BigTitle font_roles gives a two-weight title."
---
# Typography

## Rules <!-- core -->
- One or two families per poster: a display voice for the title and one neutral voice for everything else. Never mix three.
- Scale first, then contrast, then weight: make the title big before making it bold. Sizes follow a ratio (about 1.5 for dramatic display
  type, 1.25-1.33 between body levels), not arbitrary steps. Tested rule of thumb, not a standard.
- Leading: tight for display (about 1.05-1.2), looser for body (about 1.3-1.5). Tracking: wide (150-300) for small uppercase labels, normal
  or slightly tight for big titles.
- All-caps only for short lines (a title, a label, a venue); sentence case for anything that reads as a sentence.
- Never let a number or phone number break across lines. Keep the longest line of a block no more than about 40 characters on a poster.

## The type_style presets <!-- core -->
- `classic`: Montserrat Bold titles, regular body. Safe, formal.
- `bold`: heavy uppercase titles and headings, tracked labels. Recruitment, sports, hackathons.
- `elegant`: light weight, tracked uppercase. Cultural events, commemorations, art.
- `editorial`: spaced uppercase labels, light body, sentence-case titles. Talks, workshops, institutional news.
- `condensed`: tall condensed uppercase (Barlow Condensed). Dense content, long titles, sports.
Two-weight titles: BigTitle accepts `font_roles` per line, e.g. a light first word over a heavy second word, for a wordmark feel.

## Hierarchy by weight and size
The title carries size; the date carries weight; the rest is regular. Never use the same size for title and department/organiser lines:
the organiser is secondary.
