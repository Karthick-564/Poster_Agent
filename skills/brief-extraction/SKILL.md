---
name: brief-extraction
description: Turning a free-text requirement into verbatim brief fields without inventing anything.
priority: 8
use: [ingest]
---
# Brief extraction

## Rules <!-- core -->
1. Every string on the poster comes from the brief or the brand profile, verbatim. Tag its provenance.
2. Never invent emails, hashtags, dates, times, venues, prices or contact numbers. If a key fact is missing, list it as missing and say so.
3. Accept bullet dashes, asterisks, numbered lists and `Key: Value` lines.
4. A registration or application URL in the brief means a QR block and a call to action belong on the poster.
5. Date, time and venue, when supplied, are always shown.
