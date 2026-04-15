import requests
from datetime import datetime, timezone, timedelta
from io import BytesIO
from astropy.io import fits
import numpy as np
from PIL import Image
import traceback
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor
from aiapy.calibrate import degradation
from astropy.time import Time
import astropy.units as u

debug_dir = Path("d:/media/Research/OffTopic/SolarObservatory/SolarObservatory/debug_images")
debug_dir.mkdir(exist_ok=True, parents=True)

wavelengths = {131: 9, 171: 10, 193: 11, 304: 13, 1700: 16}
now = datetime.now(timezone.utc)

def process_jp2(response_content, output_filename, wl):
    from PIL import Image
    import sunpy.visualization.colormaps as cm
    
    # Load JP2 directly with PIL
    img = Image.open(BytesIO(response_content))
    data = np.array(img).astype(np.float32) / 255.0
    
    # Helioviewer applies a specific brightness scaling factor dynamically for 304.
    # An MAE and pixel ratio analysis against HV PNGs confirms a ~1.75x factor.
    if wl == 304:
        data = np.clip(data * 1.75, 0.0, 1.0)
    
    # Map the normalized values (0-1) through the official SDO colormap
    cmap = cm.cmlist.get(f'sdoaia{wl}')
    if cmap is not None:
        rgba = cmap(data)
        data_8bit = (rgba[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(data_8bit).convert("RGB")
    else:
        # Fallback if no colormap
        img = img.convert("RGB")

    # Downscale it to 2k
    img = img.resize((2048, 2048), Image.Resampling.LANCZOS)
    
    img.save(output_filename)

def process_wl(wl):
    print(f"Starting {wl}...")
    try:
        # Check today first, then yesterday if today is empty (in case it's right after midnight UTC)
        for days_back in [0, 1]:
            h = now - timedelta(days=days_back)
            base_url = f"http://jsoc.stanford.edu/data/aia/images/{h.year}/{h.month:02d}/{h.day:02d}/{wl}/"
            r = requests.get(base_url, timeout=10)
            
            if r.status_code == 200:
                pattern = r'href="([^"]+\.jp2)"'
                matches = re.findall(pattern, r.text)
                if matches:
                    # JSOC orders alphabetically, so the last match is the latest chronological image
                    best_url = base_url + matches[-1]
                    
                    print(f"[{wl}] Found latest 4K JSOC JP2: {best_url}")
                    r_nrt = requests.get(best_url, timeout=60)
                    if r_nrt.status_code == 200:
                        nrt_path = str(debug_dir / f"{wl}_nrt.png")
                        process_jp2(r_nrt.content, nrt_path, wl)
                        print(f"[{wl}] Perfectly processed 4K JSOC data and saved _nrt.png")
                        return

        print(f"[{wl}] Error: Could not find any JSOC JP2 files for today or yesterday.")
                
    except Exception as e:
        print(f"[{wl}] Error: {e}")

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(process_wl, wl) for wl in wavelengths.keys()]
    for f in futures:
        f.result()
print("All done!")
