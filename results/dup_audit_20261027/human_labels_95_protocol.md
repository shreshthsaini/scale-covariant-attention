# Blind labeling protocol: extra people or animals in 4096x4096 images

You label generated 4096x4096 images for extra instances of people or animals beyond what the prompt implies.
The generating method and any detector output are hidden. Judge only what you see.

Each image in your batch file comes with the prompt, the person/animal subjects with their implied count (how many the
prompt calls for), and a note written from the prompt text. For image imgNNN the files are
  TILES/imgNNN/overview.jpg      whole image downscaled 4x (1024 px), for context only
  TILES/imgNNN/r{R}c{C}.jpg      16 full-resolution 1024x1024 tiles, R = row 0..3 (top to bottom), C = column 0..3
                                 (left to right); tile rRcC covers x = 1024*C .. 1024*C+1023, y = 1024*R .. 1024*R+1023.

Procedure for each image:
1. Read the prompt, subjects and note.
2. View overview.jpg: understand the scene, locate the main subject(s).
3. View ALL 16 tiles at full resolution. Read four tiles per message (one row, four parallel Read calls). In every tile
   search for small figures of the listed kinds: tiny distant people or animals (as small as 30 to 50 px tall in the
   tile), partly hidden ones, silhouettes, figures on paths, ridges, shores, rooftops, balconies, windows, doorways,
   boats, in the sky or in the water. Most mistakes are missed tiny figures, so look at every tile carefully.
4. Count instances of each listed subject:
   - An instance is a distinct figure of the listed category present in the scene, at any size. "person" covers any
     human (man, woman, child, astronaut, climber, rider, silhouette). An animal subject covers that animal and
     animals easily mistaken for it at that size.
   - Count an extra figure even when it looks different from the main subject: a second pedestrian is an extra
     person when the prompt implies one. Mark each extra as copy_like true if it repeats the main subject's look
     (same clothing, pose, species, colouring), false if it is a generic figure.
   - Depictions (statue, painting, poster, carving, toy): count only when they repeat the prompted subject and the
     prompt does not ask for them. Ordinary decor showing something else is not counted (mention it in the reason).
   - A reflection or shadow consistent with a real figure in the scene is not a separate instance, unless the note
     says the implied count includes reflections, in which case follow the note. A reflection with no matching figure
     counts as an extra instance.
   - Parts of one figure (head and body, a figure split by an occluder) are one instance.
5. Label dup for the image:
   - "yes": at least one listed subject has more clear instances than implied.
   - "borderline": the only extra candidates are ambiguous (not clearly a figure, or plausibly intended scenery).
   - "no": no listed subject exceeds its implied count.
6. Immediately after finishing each image, append one JSON line to your output file (create it if absent), e.g.
   {"img": "img007", "counts": {"person": 4}, "implied": {"person": 1}, "dup": "yes",
    "extra": [{"subject": "person", "tile": "r2c1", "xy": [1480, 2770], "height_px": 300, "copy_like": false,
               "what": "woman in blue walking left of the arcade"}],
    "reason": "Three extra pedestrians besides the cloaked main figure.", "confidence": "high"}
   xy is the approximate centre in full-image pixels (tile offset plus position inside the tile). "extra" lists every
   extra or borderline candidate (empty list when there is none). reason is one sentence. confidence is high, medium
   or low.
Rules: open only the tile files of the images in your batch. Do not look for or open any other file in the project
or scratch tree. If your output file already has a line for an image (you were restarted), skip that image.
When all images are done, reply with one line per image: img, dup, counts.
