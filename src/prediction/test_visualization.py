import rasterio
import numpy as np
import matplotlib.pyplot as plt

with rasterio.open("antioquia_ndvi_2004_2008.tif") as src:
    data = src.read()   # (5, H, W)

band_names = ["Red", "NIR", "SWIR1", "SWIR2", "NDVI"]

# 1. All bands
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, ax in enumerate(axes):
    ax.imshow(data[i], cmap="gray", vmin=0, vmax=0.3)
    ax.set_title(band_names[i])
plt.show()

# 2. True color (R, G, B) — no Green band so approximate with Red
rgb = np.stack([data[0], data[0]*0.8, data[0]*0.5], axis=-1)
rgb = np.clip(rgb / 0.2, 0, 1)
plt.imshow(rgb)
plt.title("Approximate True Color")
plt.show()

# 3. False color (NIR, Red, SWIR1) — forest = bright green
false = np.stack([data[1], data[0], data[2]], axis=-1)
false = np.clip(false / 0.3, 0, 1)
plt.imshow(false)
plt.title("False Color — forest is bright green")
plt.show()

# 4. NDVI
plt.imshow(data[4], cmap="RdYlGn", vmin=-0.2, vmax=0.8)
plt.colorbar(label="NDVI")
plt.title("NDVI")
plt.show()

# 5. Random 512x512 patch
h, w = data.shape[1], data.shape[2]
r, c = np.random.randint(0, h-512), np.random.randint(0, w-512)
patch = data[4, r:r+512, c:c+512]   # NDVI patch
plt.imshow(patch, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
plt.title(f"Random NDVI patch at row={r} col={c}")
plt.show()