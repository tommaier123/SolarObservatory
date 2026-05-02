#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import struct
import io
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import requests
import traceback
from PIL import Image
 

warnings.filterwarnings('ignore')

# Set this to False to skip saving local debug images and save processing time
SAVE_DEBUG_IMAGES = False





def list_hmi_candidates(max_days=1):
    """Return list of (ts, url) for HMI `_M_4k.jpg` over -max_days..+max_days around now."""
    now = datetime.now(timezone.utc)
    out = []
    for day_offset in range(-max_days, max_days + 1):
        d = now + timedelta(days=day_offset)
        base_url = f'https://jsoc1.stanford.edu/data/hmi/images/{d.year}/{d.month:02d}/{d.day:02d}/'
        try:
            r = requests.get(base_url, timeout=10)
            if r.status_code != 200:
                continue
            matches = re.findall(r'href="([0-9_]+_M_4k\.jpg)"', r.text)
            for m in matches:
                try:
                    parts = m.split('_')
                    ts = datetime.strptime(f"{parts[0]}_{parts[1]}", '%Y%m%d_%H%M%S').replace(tzinfo=timezone.utc)
                except Exception:
                    continue
                out.append((ts, base_url + m))
        except requests.RequestException:
            continue
    out.sort(key=lambda x: x[0])
    return out


def list_aia_candidates(wavelength, max_days=1):
    """Return list of (ts, url) for AIA JP2s over -max_days..+max_days around now."""
    now = datetime.now(timezone.utc)
    out = []
    for day_offset in range(-max_days, max_days + 1):
        h = now + timedelta(days=day_offset)
        base_url = f"http://jsoc.stanford.edu/data/aia/images/{h.year}/{h.month:02d}/{h.day:02d}/{wavelength}/"
        try:
            r = requests.get(base_url, timeout=10)
            if r.status_code != 200:
                continue
            matches = re.findall(r'href="([^\"]+\.jp2)"', r.text)
            for m in matches:
                parts = m.split('__')
                if len(parts) < 2:
                    continue
                try:
                    date_str = parts[0].replace('_', '-')
                    time_str = parts[1][:8].replace('_', ':')
                    ts = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                except Exception:
                    continue
                out.append((ts, base_url + m))
        except requests.RequestException:
            continue
    out.sort(key=lambda x: x[0])
    return out


def download_aia_by_url(url, wavelength):
    """Download a JP2 by URL and process to 2048 grayscale array, returning (flattened, timestamp, wl, w, h)."""
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        from io import BytesIO
        img = Image.open(BytesIO(r.content))

        # attempt to parse timestamp from filename
        fn = url.rsplit('/', 1)[-1]
        parts = fn.split('__')
        if len(parts) >= 2:
            date_str = parts[0].replace('_', '-')
            time_str = parts[1][:8].replace('_', ':')
            try:
                ts = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            except Exception:
                ts = datetime.now(timezone.utc)
        else:
            ts = datetime.now(timezone.utc)

        data = np.array(img).astype(np.float32) / 255.0
        if wavelength == 304:
            data = np.clip(data * 1.75, 0.0, 1.0)

        import sunpy.visualization.colormaps as cm
        cmap = cm.cmlist.get(f'sdoaia{wavelength}')
        if cmap is not None:
            rgba = cmap(data)
            data_8bit = (rgba[:, :, :3] * 255).astype(np.uint8)
            img_out = Image.fromarray(data_8bit).convert('L')
        else:
            img_out = img.convert('L')

        img_out = img_out.resize((2048, 2048), Image.Resampling.LANCZOS)
        arr = np.array(img_out)

        if SAVE_DEBUG_IMAGES:
            debug_dir = Path(__file__).parent / 'debug_images'
            debug_dir.mkdir(exist_ok=True)
            Image.fromarray(arr).save(debug_dir / f'AIA_{wavelength}_{ts.strftime("%Y%m%d_%H%M%S")}_byurl.png')

        return arr.flatten(), ts, wavelength, 2048, 2048
    except Exception as e:
        print(f"Error downloading AIA by URL {url}: {e}")
        traceback.print_exc()
        return None, None, wavelength, 0, 0


def download_hmi_by_url(url):
    """Download HMI jpg by URL and return flattened array and timestamp."""
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
        # parse timestamp
        fn = url.rsplit('/', 1)[-1]
        parts = fn.split('_')
        try:
            ts = datetime.strptime(f"{parts[0]}_{parts[1]}", '%Y%m%d_%H%M%S').replace(tzinfo=timezone.utc)
        except Exception:
            ts = datetime.now(timezone.utc)

        img = img.resize((2048, 2048), Image.Resampling.LANCZOS)
        arr = np.array(img)
        if arr.ndim == 3:
            arr = arr[:, :, 0]

        if SAVE_DEBUG_IMAGES:
            debug_dir = Path(__file__).parent / 'debug_images'
            debug_dir.mkdir(exist_ok=True)
            Image.fromarray(arr).save(debug_dir / f'HMI_{ts.strftime("%Y%m%d_%H%M%S")}_byurl.png')

        return arr.flatten(), ts, 2048, 2048
    except Exception as e:
        print(f"Error downloading HMI by URL {url}: {e}")
        traceback.print_exc()
        return None, None, 0, 0





def create_rgb_image(r_channel, g_channel, b_channel):
    if len(r_channel) != len(g_channel) or len(g_channel) != len(b_channel):
        raise ValueError("All channels must have the same length")
    
    pixel_count = len(r_channel)
    rgb_data = np.empty(pixel_count * 3, dtype=np.uint8)
    
    rgb_data[0::3] = r_channel
    rgb_data[1::3] = g_channel
    rgb_data[2::3] = b_channel
    
    return rgb_data.tobytes()


def create_container_file(timestamp, downloads):
    container_path = Path(__file__).parent / 'solar.dat'
    successful = [d for d in downloads if d[0] is not None]
    wavelength_order = [131, 171, 193, 304, 1700, 'HMI']
    download_dict = {d[2]: d for d in successful}
    
    ordered_downloads = []
    for wl in wavelength_order:
        if wl in download_dict:
            ordered_downloads.append(download_dict[wl])
        else:
            black_data = np.zeros(2048 * 2048, dtype=np.uint8)
            ordered_downloads.append((black_data, timestamp, wl, 2048, 2048))
    
    width, height = 2048, 2048
    rgb_image_0 = create_rgb_image(
        ordered_downloads[0][0],
        ordered_downloads[1][0],
        ordered_downloads[2][0]
    )
    
    rgb_image_1 = create_rgb_image(
        ordered_downloads[3][0],
        ordered_downloads[4][0],
        ordered_downloads[5][0]
    )
    
    with open(container_path, 'wb') as f:
        f.write(struct.pack('B', 2))
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        f.write(timestamp_str.encode('ascii'))
        
        f.write(struct.pack('<H', width))
        f.write(struct.pack('<H', height))
        f.write(rgb_image_0)
        
        f.write(struct.pack('<H', width))
        f.write(struct.pack('<H', height))
        f.write(rgb_image_1)


def main():
    # Build candidate lists in parallel for HMI and AIA wavelengths
    aia_wavelengths = [131, 171, 193, 304, 1700]
    lists = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(list_aia_candidates, wl, 1): ('aia', wl) for wl in aia_wavelengths}
        futures[executor.submit(list_hmi_candidates, 1)] = ('hmi', None)
        for fut, meta in futures.items():
            kind, wl = meta
            try:
                res = fut.result()
            except Exception:
                res = []
            if kind == 'hmi':
                lists['hmi'] = res
            else:
                lists[wl] = res

    # collect candidate reference times: take latest N from each list
    REF_CANDIDATES = set()
    N = 3
    for k, lst in lists.items():
        for ts, url in lst[-N:]:
            REF_CANDIDATES.add(ts)

    if not REF_CANDIDATES:
        raise Exception('No candidates found for HMI/AIA')

    # For each reference candidate, compute nearest file per channel and the max deviation
    def nearest_in_list(lst, ref_ts):
        if not lst:
            return None, None, float('inf')
        best = min(lst, key=lambda x: abs((x[0] - ref_ts).total_seconds()))
        return best[0], best[1], abs((best[0] - ref_ts).total_seconds())

    best_ref = None
    best_selection = None
    best_maxdiff = float('inf')

    for ref in sorted(REF_CANDIDATES, reverse=True):
        selection = {}
        maxdiff = 0.0
        # HMI
        h_ts, h_url, h_diff = nearest_in_list(lists.get('hmi', []), ref)
        selection['HMI'] = (h_ts, h_url)
        maxdiff = max(maxdiff, h_diff)
        # AIA channels
        for wl in aia_wavelengths:
            ts, url, diff = nearest_in_list(lists.get(wl, []), ref)
            selection[wl] = (ts, url)
            maxdiff = max(maxdiff, diff)

        # choose smaller maxdiff, tie-breaker newer ref
        if (maxdiff < best_maxdiff) or (maxdiff == best_maxdiff and (best_ref is None or ref > best_ref)):
            best_maxdiff = maxdiff
            best_ref = ref
            best_selection = selection

    if best_selection is None:
        raise Exception('Failed to choose selection')

    # Download selected images in parallel
    downloads = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_map = {}
        # schedule AIA downloads
        for wl in aia_wavelengths:
            ts, url = best_selection[wl]
            if url:
                future_map[executor.submit(download_aia_by_url, url, wl)] = ('aia', wl)
        # schedule HMI
        h_ts, h_url = best_selection['HMI']
        if h_url:
            future_map[executor.submit(download_hmi_by_url, h_url)] = ('hmi', None)

        for fut, meta in future_map.items():
            kind, wl = meta
            try:
                res = fut.result()
            except Exception:
                res = (None, None, wl, 0, 0)
            if kind == 'aia':
                downloads.append(res)
            else:
                # download_hmi_by_url returns (flattened, ts, w, h)
                arr, ts, w, h = res
                downloads.append((arr, ts, 'HMI', w, h))

    all_downloads = downloads
    # Use the average timestamp of the successfully downloaded images as the container timestamp
    successful_items = [d for d in all_downloads if d[0] is not None and d[1] is not None]
    if successful_items:
        secs = [d[1].timestamp() for d in successful_items]
        avg = sum(secs) / len(secs)
        container_ts = datetime.fromtimestamp(avg, tz=timezone.utc)
    else:
        container_ts = datetime.now(timezone.utc)

    create_container_file(container_ts, all_downloads)
    timestamp_path = Path(__file__).parent / 'timestamp.txt'
    with open(timestamp_path, 'w') as f:
        f.write(container_ts.strftime('%Y-%m-%d %H:%M:%S'))
    
    successful = [d for d in all_downloads if d[0] is not None]
    container_path = Path(__file__).parent / 'solar.dat'
    file_size = container_path.stat().st_size
    
    print("\n" + "="*60)
    print("TIMESTAMP SUMMARY")
    print("="*60)
    print(f"Current time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    # Print age of the average timestamp compared to now (UTC)
    now_utc = datetime.now(timezone.utc)
    age_delta = now_utc - container_ts
    age_seconds = int(age_delta.total_seconds())
    hrs, rem = divmod(age_seconds, 3600)
    mins, secs = divmod(rem, 60)
    if hrs > 0:
        age_human = f"{hrs}h{mins}m{secs}s"
    elif mins > 0:
        age_human = f"{mins}m{secs}s"
    else:
        age_human = f"{secs}s"
    print(f"Data age: {age_human} ({age_seconds/60:.1f} min)")
    print("="*60)
    for wavelength in [131, 171, 193, 304, 1700, 'HMI']:
        download = next((d for d in successful if d[2] == wavelength), None)
        if download:
            ts = download[1]
            delta = (ts - container_ts).total_seconds() / 60
            print(f"{str(wavelength).rjust(5)}: {ts.strftime('%Y-%m-%d %H:%M:%S %Z')} (Δ {delta:+.1f} min)")
    print("="*60)
    print(f"Container: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    print("="*60)


if __name__ == '__main__':
    main()



