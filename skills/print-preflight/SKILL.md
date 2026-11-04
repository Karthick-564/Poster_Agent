---
name: print-preflight
description: Rules for print-ready output (A4, A3 and larger) - bleed, safe margin, effective resolution per image.
priority: 9
use: [plan]
---
# Print preflight

## Rules <!-- formats: a3, a4 -->
- Effective resolution of each raster = its pixels / its printed size in inches. Report it per image; never certify an upscaled
  low-resolution image as 300 dpi.
- 3 mm bleed on all four sides; critical text and logos inside a 6 mm safe margin.
- PDF carries vector type (fonts embedded or outlined).
- Digital formats (story, 4:5, square) need none of this; the 4.5 % safe margin is enough.
