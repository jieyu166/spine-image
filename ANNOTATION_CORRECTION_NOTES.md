# Annotation Correction Notes

Date: 2026-05-22

Case context: lumbar lateral X-ray annotated in `spinal-annotation-web.html`.

## 2026-05-22 Doctor Correction vs Codex Initial Annotation

Doctor corrected the initial Codex annotation substantially inward. The main error pattern was that Codex placed endplate points too far outside the true vertebral body margins, producing oversized vertebral polygons.

Key differences:

- Overall: Codex boxes were too large, especially in the anterior-posterior width. Doctor points hug the visible cortical/endplate margins more tightly.
- Posterior points: Codex placed many posterior-superior and posterior-inferior points too far posterior/right, extending into the pedicle/facet/posterior soft-tissue overlap rather than stopping at the posterior vertebral body wall.
- Anterior points: Codex often placed anterior points on osteophyte margins or outside the vertebral body contour. Doctor correction uses the vertebral body cortical margin/endplate corner, not the projecting osteophyte tip.
- Midpoints: Codex midpoints were approximately centered between overly wide endpoints, so they also drifted outside the true endplate line. Doctor midpoints follow the actual endplate surface, including mild concavity/tilt.
- T12 lower endplate: Codex placed the T12 lower line too high and too wide. Doctor correction places it on the visible lower endplate line, shorter and more horizontal.
- S1 upper endplate: Codex placed the S1 upper line too low/wide posteriorly. Doctor correction is shorter and aligned with the true S1 superior endplate.
- L5: Codex substantially overestimated the posterior and inferior extent. Doctor correction shows L5 as a more trapezoid/wedge-shaped vertebral body with the posterior border much more anterior than Codex marked.
- L1-L4: Codex rectangles were generally inflated; doctor correction reduced height and width, with posterior borders moved anteriorly and endplate lines adjusted to visible cortical boundaries.

Practical rules for future computer-use annotation:

- Do not include osteophytes in vertebral body corners.
- Do not use the outer posterior element shadow as the posterior vertebral body border.
- Prefer the dense cortical vertebral body wall and endplate line over faint overlapping soft-tissue/bowel/pedicle shadows.
- For each endplate, place anterior, middle, posterior points on the actual endplate surface, not on an enclosing rectangle.
- When uncertain, err slightly inward on the vertebral body margin rather than outward.
- Use smaller, tighter polygons; the corrected annotations are consistently smaller than the initial Codex estimate.
- For L5/S1 and T12/L1 boundary levels, zoom in and confirm with the doctor or with a screenshot before final export because these levels are most prone to overextension.

Pending:

- Doctor will export the corrected JSON from the annotation tool.
- After export, compare the corrected JSON numerically against the initial Codex coordinates if both files are available.

## 2026-05-22 Second Patient Correction

Case context: second lumbar lateral X-ray pasted into `spinal-annotation-web.html`; doctor corrected the Codex trial annotation.

Observed correction pattern:

- Codex still overestimated vertebral width, especially the posterior/right border. Doctor moved posterior points markedly anterior/left for L1-L5.
- Codex used near-rectangular vertebral boxes. Doctor corrected toward smaller trapezoids following the actual vertebral body cortex.
- Doctor marked several levels as biconcave/wedged by placing the middle endplate points on the depressed/curved endplate surface rather than on a straight line between endpoints.
- L1-L4: doctor moved posterior-superior and posterior-inferior points inward and shortened the endplate lines. The corrected bodies are much narrower than Codex's first estimate.
- L5: doctor made the body much smaller and wedge-shaped; Codex had extended L5 posteriorly/inferiorly too far.
- S1 upper endplate: doctor shortened the line and placed it along the true superior sacral endplate rather than a broader sacral slope.
- T12 lower endplate: doctor shortened and angled the line with the visible lower endplate; avoid extending into overlapping rib/facet shadows.
- The corrected annotation triggered biconcave compression warnings for L1-L4, which is a useful clue: the middle points should reflect true endplate depression rather than be smoothed away.

Rules added for the next annotation attempt:

- Be even more conservative on posterior body margins: stop at the posterior vertebral body cortex, not at pedicle/facet overlap.
- Draw smaller than seems intuitive from the raw lateral image; prior Codex attempts remain too large even after initial correction.
- For biconcave or concave endplates, deliberately place the middle point on the concavity/depression. Do not average the anterior/posterior endpoint line.
- Prefer short endplate segments that match the dense visible endplate, not the full projected shadow.
- For L5, expect a shorter wedge/trapezoid and avoid extending posteriorly into lumbosacral overlap.
- For S1 and T12 boundary levels, use short boundary lines only on the visible endplate surface.
- Before batch-clicking a full case, mark one vertebra and compare visually against these doctor-corrected examples; adjust inward if the polygon resembles a large rectangle.

## 2026-05-22 Coordinate Check and User Feedback

After another annotation attempt on the same second-patient image, the physician/user noted that the annotations were obviously too posterior and still looked enlarged, raising concern for a coordinate-conversion or zoom issue.

Coordinate check in the browser:

- `canvas.width = 782`, `canvas.height = 2038`
- displayed canvas rect was also `782 x 2038`
- `zoom = 1`
- `panX = 0`, `panY = 0`
- stored points matched requested click coordinates within about 1 px

Conclusion: no meaningful browser/canvas coordinate-conversion error was found for this attempt. The error is mainly landmark selection: Codex placed the landmarks too posterior and too large.

Additional correction for future attempts:

- Treat the current failure as a systematic posterior bias, not only an over-wide-box problem.
- Shift the whole vertebral annotation anteriorly when the anterior vertebral cortex is visible left of the Codex anterior points.
- Do not anchor the posterior border on the posterior shadow/facet/pedicle overlap; this makes the box appear enlarged and shifted backward.
- Use the anterior cortical body wall as the first anchor. Then choose a short posterior endpoint on the true posterior vertebral body cortex, not the outer posterior projection.
- Before completing a full batch, annotate one mid-lumbar vertebra and verify it is not posterior-shifted relative to the visible anterior cortex.
- If points appear to have a "zoomed/enlarged" feel, first assume landmark selection bias; verify `zoom/pan/canvas rect`, then shrink and shift anterior rather than blaming coordinate conversion.

## 2026-05-22 Numeric Comparison After Doctor Correction

Comparison target: Codex reannotation after anterior-shift attempt vs physician-corrected annotation on the same image.

Browser coordinate state remained normal (`zoom=1`, `panX=0`, `panY=0`, canvas display size matched image size). The correction was therefore treated as a landmark-selection correction, not a coordinate transform correction.

Mean physician correction by level, relative to Codex:

- S1: x -46.7 px, y -128.0 px; width -145.6 px.
- L5: x -94.4 px, y -58.0 px; width -44.0 px; height -24.5 px.
- L4: x -116.1 px, y -9.4 px; width -79.2 px; height -43.4 px.
- L3: x -123.7 px, y +39.3 px; width -51.7 px; height -31.6 px.
- L2: x -125.2 px, y +88.7 px; width -22.2 px; height -17.5 px.
- L1: x -109.2 px, y +144.7 px; width -31.4 px; height -29.8 px.
- T12: x -90.3 px, y +221.7 px; width -41.4 px.

Interpretation:

- The dominant error remains a large posterior/right bias: physician moved nearly all vertebral points left/anterior by about 90-125 px from L1-L5.
- Codex still overestimates width; physician narrowed every vertebral body, most strongly at S1 and L4.
- Codex over-stretched the vertical distribution: lower boundary S1 was moved superiorly, while upper boundary T12/L1 were moved inferiorly.
- For future attempts, apply a much stronger anterior shift than previously attempted, and constrain each vertebra to a smaller, shorter body before clicking.

## 2026-05-22 Next Patient Doctor Correction

Comparison target: Codex first-pass annotation vs physician-corrected annotation on the next lumbar lateral image.

Mean physician correction by level, relative to Codex:

- S1: x -219.3 px, y -166.4 px; height -54.9 px; width +80.2 px.
- L5: x -207.4 px, y -111.1 px; width +15.8 px; height +19.8 px.
- L4: x -143.1 px, y -70.3 px; width -8.4 px; height +38.3 px.
- L3: x -96.3 px, y -87.0 px; width +24.9 px; height +63.9 px.
- L2: x -60.3 px, y -54.2 px; width about unchanged; height +29.2 px.
- L1: x -55.4 px, y +34.6 px; width -45.6 px.
- T12: x -69.1 px, y +142.7 px; width -11.6 px.

Interpretation:

- Lumbosacral levels were still badly misplaced: physician moved S1/L5 far anterior/left and superior/up. Codex was too posterior and too low at the bottom of the image.
- Codex's L5/S1 geometry produced false or exaggerated listhesis/disc narrowing alerts. Do not trust automated metric warnings from a poor first-pass annotation.
- Upper levels required less x correction but still needed level-specific y correction; T12 was too high in Codex and should be anchored lower on the visible inferior endplate.
- Physician did not simply shrink every level. Some corrected vertebrae became wider/taller while moving to the true vertebral body. The main failure is wrong anchoring, especially in the lumbosacral region, not just scale.

Rules added:

- For the next first pass, anchor S1 and L5 first using the clearly visible sacral/lumbosacral cortical line. Do not extrapolate from posterior shadows.
- At L5/S1, if the annotation creates dramatic retrolisthesis or disc narrowing, suspect point placement before accepting the metric.
- Lower lumbar points should be shifted much farther anterior/left than intuition from the full shadow suggests.
- T12 lower endplate must be anchored on the actual visible lower endplate; if uncertain, defer T12 until after L1-L5 are placed.
