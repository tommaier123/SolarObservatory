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

def process_fits(response_content, output_filename, wl):
    with fits.open(BytesIO(response_content)) as hdul:
        if len(hdul) > 1 and hdul[1].data is not None:
            data = hdul[1].data
            header = hdul[1].header
        else:
            data = hdul[0].data
            header = hdul[0].header

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # Convert to DN/s strictly like jp2gen does (img = img * 1.0 / exptime)
    exptime = header.get("EXPTIME", 1.0)
    if exptime <= 0:
        exptime = 1.0
    
    data = data / exptime

    # Helioviewer SDO AIA scaling parameters from IDL source (jp2gen hvs_version6_aia.pro)
    # The 304 upper bound is specifically reduced from 250.0 to 100.0 to drastically increase high-end contrast
    # because NRT exptime-normalized DN/s data caps around ~90.0
    scaling_params = {
        131: (0.7, 500.0, 3),    # log10
        171: (5.0, 1605.0, 1),   # sqrt
        193: (20.0, 2500.0, 3),  # log10
        304: (0.7, 35.0, 3),     # log10
        1700: (220.0, 5000.0, 1) # sqrt
    }

    if wl in scaling_params:
        dataMin, dataMax, dataScalingType = scaling_params[wl]
        
        # Clip to strictly defined data bounds
        data = np.clip(data, dataMin, dataMax)
        
        if dataScalingType == 3: # log10
            data = np.log10(data)
            min_val = np.log10(dataMin)
            max_val = np.log10(dataMax)
            normalized = (data - min_val) / (max_val - min_val)
        elif dataScalingType == 1: # SQRT
            data = np.sqrt(data)
            min_val = np.sqrt(dataMin)
            max_val = np.sqrt(dataMax)
            normalized = (data - min_val) / (max_val - min_val)
        else: # LINEAR
            normalized = (data - dataMin) / (dataMax - dataMin)
            
    normalized = np.clip(normalized, 0, 1)

    import sunpy.visualization.colormaps as cm
    import scipy.ndimage as ndimage

    # Map the normalized values (0-1) through the official SDO colormap if available
    cmap = cm.cmlist.get(f'sdoaia{wl}')
    if cmap is not None:
        rgba = cmap(normalized)
        data_8bit = (rgba[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(data_8bit).convert("RGB")
    else:
        data_8bit = (normalized * 255).astype(np.uint8)
        img = Image.fromarray(data_8bit).convert("RGB")

    # Helioviewer unconditionally applies a Difference of Gaussians unsharp mask inside suits2img/jp2img
    # This precisely handles the remaining "sharpness/brightness" contrast differences
    img_arr = np.array(img).astype(np.float32)
    s = 1.0 / 1.6
    in0 = ndimage.gaussian_filter(img_arr, sigma=(s, s, 0))
    in1 = ndimage.gaussian_filter(img_arr, sigma=(s * 1.6, s * 1.6, 0))
    in2 = ndimage.gaussian_filter(img_arr, sigma=(s * 3.2, s * 3.2, 0))
    crisp = img_arr + 0.33 * (1.5 * in0 - 1.0 * in1 - 0.5 * in2) + 0.5
    img = Image.fromarray(np.clip(crisp, 0, 255).astype(np.uint8))

    img = img.resize((1024, 1024), Image.LANCZOS)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img.save(output_filename)

def process_wl(wl, source_id):
    print(f"Starting {wl}...")
    try:
        hv_api = f"https://api.helioviewer.org/v2/getClosestImage/?date={now.strftime('%Y-%m-%dT%H:%M:%S.000Z')}&sourceId={source_id}"
        r_hv = requests.get(hv_api, timeout=20).json()
        hv_id = r_hv['id']
        hv_date_str = r_hv['date']
        sync_dt = datetime.strptime(hv_date_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

        r_png = requests.get(f"https://api.helioviewer.org/v2/downloadImage/?id={hv_id}&scale=4", timeout=30)
        if r_png.status_code == 200:
            hv_img = Image.open(BytesIO(r_png.content)).convert("RGB")
            hv_img.save(debug_dir / f"{wl}_hv.png")
            print(f"[{wl}] Saved _hv.png")
            
        h = sync_dt
        base_url = f"http://jsoc.stanford.edu/data/aia/synoptic/nrt/{h.year}/{h.month:02d}/{h.day:02d}/H{h.hour:02d}00/"
        r = requests.get(base_url, timeout=10)
        best_diff = 1e9
        best_url = None
        if r.status_code == 200:
            pattern = f'AIA{h.year}{h.month:02d}{h.day:02d}_(\\d{{6}})_{wl:04d}\\.fits'
            matches = re.findall(pattern, r.text)
            for time_str in matches:
                ts = datetime.strptime(f"{h.year}{h.month:02d}{h.day:02d}_{time_str}", "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
                diff = abs((ts - sync_dt).total_seconds())
                if diff < best_diff:
                    best_diff = diff
                    best_url = base_url + f"AIA{h.year}{h.month:02d}{h.day:02d}_{time_str}_{wl:04d}.fits"
        
        if best_url:
            print(f"[{wl}] Found NRT data, downloading...")
            r_nrt = requests.get(best_url, timeout=30)
            nrt_path = str(debug_dir / f"{wl}_nrt.png")
            process_fits(r_nrt.content, nrt_path, wl)
            print(f"[{wl}] Saved _nrt.png")
            
            hv_path = str(debug_dir / f"{wl}_hv.png")
            if Path(nrt_path).exists() and Path(hv_path).exists():
                print(f"[{wl}] Comparison images generated at {nrt_path} and {hv_path}")
                
    except Exception as e:
        print(f"[{wl}] Error: {e}")

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(process_wl, wl, sid) for wl, sid in wavelengths.items()]
    for f in futures:
        f.result()
print("All done!")
