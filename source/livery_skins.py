"""Make the 2026 car's extra skins from its race livery.

The race livery is the kn5's own CarpaintExterior_Base_color.png (made from
source/texture/CarpaintExterior_Base_color.png), so a skin replaces it with a
file of the same name. This writes, into mitsuoka_viewt_lemons_2026/skins/:

- lost_in_translation_red: the livery with the rising sun's cream rays
  painted red. The cream race numbers stay.
- one skin per road car colour (burgundy, silver, ...): the livery's green,
  red and cream all in that colour, as plain road paint. The door roundels,
  sponsor stickers and number plates stay; the cream race numbers go.

Both kinds paint over the three small stickers that carry rising-sun rays and
keep the front plate's cream lettering. The road colours are the flat body
colours of the road car's skins (mitsuoka_viewt/skins/<colour>/), and their
ui_skin.json and livery.png are copied from there.

Run from the repository root:  python source/livery_skins.py
"""
import json
import os
import shutil

import numpy as np
from PIL import Image
from scipy import ndimage

SRC = 'source/texture/CarpaintExterior_Base_color.png'
ROAD_BASE = 'source/texture/CarpaintExterior_Base_color_base.png'
ROAD_SKINS = 'mitsuoka_viewt/skins'
SKINS = 'mitsuoka_viewt_lemons_2026/skins'
TEXTURE = 'CarpaintExterior_Base_color.png'

CREAM = np.array([226, 224, 210], float)
RED = np.array([192, 2, 2], float)
GREEN = np.array([27, 46, 37], float)
MIN_AREA = 800  # smaller cream specks are left alone (bits of sticker artwork)
# The cream race numbers: centre points (x, y) in the 4096 px texture
NUMBERS = [(495, 1510), (500, 1805), (1414, 2890), (1626, 2885)]
# The front number plate's cream lettering and stripes, kept (x0, y0, x1, y1)
PLATE = (940, 1280, 1090, 1700)
# Stickers with rising-sun rays (x0, y0, x1, y1)
STICKERS = [(3180, 2850, 3345, 2930), (3365, 2922, 3525, 3000), (3520, 2815, 3690, 2900)]
ROAD_COLOURS = ['cream_ivory', 'silver', 'burgundy', 'bottle_green', 'slate_blue', 'pale_blue']


def seg_dist(p, a, b):
    """Distance of each pixel from segment a-b, and how far along it (0 at a)."""
    d = b - a
    t = np.clip(((p - a) @ d) / (d @ d), 0, 1)
    return np.linalg.norm(p - (a + t[..., None] * d), axis=-1), t


def save(img, skin):
    os.makedirs(f'{SKINS}/{skin}', exist_ok=True)
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(f'{SKINS}/{skin}/{TEXTURE}', optimize=True)
    print('wrote', skin)


a = np.array(Image.open(SRC).convert('RGB')).astype(float)
cream = np.linalg.norm(a - CREAM, axis=-1) < 30
lab, n = ndimage.label(cream)
area = ndimage.sum(cream, lab, index=np.arange(1, n + 1))
numbers = {lab[y, x] for x, y in NUMBERS}
assert 0 not in numbers, 'a number centre is not on cream'
x0, y0, x1, y1 = PLATE
plate = {i for i, sl in enumerate(ndimage.find_objects(lab), 1)
         if sl[1].start >= x0 and sl[1].stop <= x1 and sl[0].start >= y0 and sl[0].stop <= y1}
plate_px = np.isin(lab, list(plate))

# --- Red: the rising sun's cream rays turned red ---
paint = [i for i in 1 + np.flatnonzero(area >= MIN_AREA) if i not in numbers | plate]
rays = np.isin(lab, paint)
near = ndimage.binary_dilation(rays, iterations=3)
out = a.copy()
for other in (RED, GREEN):
    # Edge pixels blend cream with red or green: swap the cream share for red
    dist, t = seg_dist(a, other, CREAM)
    m = near & (dist < 20)
    out[m] = a[m] + t[m][:, None] * (RED - CREAM)
out[rays] = RED
for sx0, sy0, sx1, sy1 in STICKERS:
    out[sy0:sy1, sx0:sx1] = RED
save(out, 'lost_in_translation_red')
red_meta = json.load(open(f'{SKINS}/lost_in_translation/ui_skin.json'))
red_meta.update(skinname='Lost in Translation (red)', priority=2)
json.dump(red_meta, open(f'{SKINS}/lost_in_translation_red/ui_skin.json', 'w'), indent=2)
shutil.copy(f'{SKINS}/lost_in_translation/livery.png', f'{SKINS}/lost_in_translation_red/livery.png')

# --- Road colours: the large patches of the livery's green, red and cream
# (numbers included) become the one colour, and so do the pixels round their
# edges that blend two livery colours. Decals, which only share colours with
# the livery in small patches, stay ---
pure = ((np.linalg.norm(a - GREEN, axis=-1) < 15) | (np.linalg.norm(a - RED, axis=-1) < 30)
        | (np.linalg.norm(a - CREAM, axis=-1) < 25))
plab, pn = ndimage.label(pure)
parea = ndimage.sum(pure, plab, index=np.arange(1, pn + 1))
livery = np.isin(plab, 1 + np.flatnonzero(parea >= MIN_AREA)) & ~plate_px
# Shaded and smeared reds (pinker or darker than RED) where they touch the
# livery; the red in sponsor stickers is ringed by the sticker, so it doesn't
reddish = (a[..., 0] > 140) & (a[..., 0] - a[..., 1] > 80) & (a[..., 0] - a[..., 2] > 80) & ~livery
rlab, rn = ndimage.label(reddish)
touching = np.unique(rlab[ndimage.binary_dilation(livery, iterations=2) & reddish])
livery |= np.isin(rlab, touching[touching > 0])
blend = np.zeros_like(livery)
for c1, c2 in ((GREEN, RED), (GREEN, CREAM), (RED, CREAM)):
    blend |= seg_dist(a, c1, c2)[0] < 25
livery |= ndimage.binary_dilation(livery, iterations=4) & blend & ~plate_px
# Stray edge pixels left round the old numbers: a pixel with at least 6 of its
# 8 neighbours painted is painted too
for _ in range(3):
    neighbours = ndimage.convolve(livery.astype(int), np.ones((3, 3), int), mode='constant') - livery
    livery |= (neighbours >= 6) & ~plate_px
road_base = np.array(Image.open(ROAD_BASE).convert('RGB')).astype(int)
for skin in ROAD_COLOURS:
    road = np.array(Image.open(f'{ROAD_SKINS}/{skin}/CarpaintExterior_Base_color_base.png').convert('RGB')).astype(int)
    colour = np.median(road[np.abs(road - road_base).sum(-1) > 30], axis=0)
    out = a.copy()
    out[livery] = colour
    for sx0, sy0, sx1, sy1 in STICKERS:
        out[sy0:sy1, sx0:sx1] = colour
    save(out, skin)
    meta = json.load(open(f'{ROAD_SKINS}/{skin}/ui_skin.json'))
    meta.update(team='Lost in Translation', number='67', priority=1)
    json.dump(meta, open(f'{SKINS}/{skin}/ui_skin.json', 'w'), indent=2)
    shutil.copy(f'{ROAD_SKINS}/{skin}/livery.png', f'{SKINS}/{skin}/livery.png')
