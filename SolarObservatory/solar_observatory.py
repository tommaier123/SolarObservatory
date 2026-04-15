#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import struct
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import drms
from astropy.io import fits
import requests
import traceback
from PIL import Image
import time

warnings.filterwarnings('ignore')

# Set this to False to skip saving local debug images and save processing time
SAVE_DEBUG_IMAGES = False


def download_hmi_nrt():
    try:
        client = drms.Client()
        series = 'hmi.m_720s_nrt'
        
        now = datetime.now(timezone.utc)
        query_start = (now - timedelta(hours=24)).strftime('%Y.%m.%d_%H:%M:%S_TAI')
        query_end = now.strftime('%Y.%m.%d_%H:%M:%S_TAI')
        
        result = None
        for attempt in range(5):
            try:
                result = client.query(f'{series}[{query_start}-{query_end}]', key='T_REC', seg='magnetogram')
                break
            except Exception as e:
                print(f"JSOC Query failed (attempt {attempt + 1}/5): {e}. Retrying in 10s...")
                if attempt == 4:
                    raise
                time.sleep(10)
        
        if isinstance(result, tuple):
            keys, segments = result
        else:
            keys = result
            segments = None
        
        if len(keys) == 0:
            return None, None, 0, 0
        
        latest_rec = keys.iloc[-1]['T_REC']
        
        if segments is not None and 'magnetogram' in segments.columns:
            magnetogram_path = segments.iloc[-1]['magnetogram']
            base_url = "http://jsoc.stanford.edu"
            fits_url = f"{base_url}{magnetogram_path}"
            
            response = requests.get(fits_url, timeout=60)
            response.raise_for_status()
            
            from io import BytesIO
            with fits.open(BytesIO(response.content)) as hdul:
                if len(hdul) > 1 and hdul[1].data is not None:
                    data = hdul[1].data
                    header = hdul[1].header
                else:
                    data = hdul[0].data
                    header = hdul[0].header
            
            time_str = latest_rec.replace('_TAI', '')
            try:
                actual_timestamp = datetime.strptime(time_str, '%Y.%m.%d_%H:%M:%S.%f')
            except ValueError:
                actual_timestamp = datetime.strptime(time_str, '%Y.%m.%d_%H:%M:%S')
            
            if actual_timestamp.tzinfo is None:
                actual_timestamp = actual_timestamp.replace(tzinfo=timezone.utc)
            
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            vmin, vmax = -1500, 1500
            normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
            data_8bit = (normalized * 255).astype(np.uint8)
            
            img = Image.fromarray(data_8bit)
            img = img.resize((2048, 2048), Image.LANCZOS)
            resized = np.array(img)
            resized = np.fliplr(resized)
            
            if SAVE_DEBUG_IMAGES:
                debug_dir = Path(__file__).parent / 'debug_images'
                debug_dir.mkdir(exist_ok=True)
                Image.fromarray(resized).save(debug_dir / f'HMI_{actual_timestamp.strftime("%Y%m%d_%H%M%S")}.png')
            
            return resized.flatten(), actual_timestamp, 2048, 2048
        
    except Exception as e:
        print(f"Error downloading HMI NRT data: {e}")
        traceback.print_exc()
        return None, None, 0, 0


def download_and_process(timestamp, wavelength):
    import re
    all_matches = []
    
    # Check yesterday, today, and tomorrow to ensure we find the absolute closest image even near midnight
    for day_offset in [-1, 0, 1]:
        h = timestamp + timedelta(days=day_offset)
        base_url = f"http://jsoc.stanford.edu/data/aia/images/{h.year}/{h.month:02d}/{h.day:02d}/{wavelength}/"
        try:
            r = requests.get(base_url, timeout=10)
            if r.status_code == 200:
                pattern = r'href="([^"]+\.jp2)"'
                matches = re.findall(pattern, r.text)
                for m in matches:
                    all_matches.append((base_url, m))
        except requests.exceptions.RequestException:
            pass
            
    if not all_matches:
        raise Exception(f"[{wavelength}] Error: No JP2 files found around {timestamp}")
        
    best_match_url = None
    min_diff = None
    actual_timestamp = timestamp
    
    for base_url, filename in all_matches:
        try:
            # Parse filenames like 2026_04_15__00_00_09_348__SDO_AIA_AIA_171.jp2
            parts = filename.split('__')
            if len(parts) >= 2:
                date_str = parts[0].replace('_', '-')
                time_str = parts[1][:8].replace('_', ':')
                file_time = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
                file_time = file_time.replace(tzinfo=timezone.utc)
                
                diff = abs((file_time - timestamp).total_seconds())
                if min_diff is None or diff < min_diff:
                    min_diff = diff
                    best_match_url = base_url + filename
                    actual_timestamp = file_time
        except Exception:
            continue
            
    if not best_match_url:
        # Fallback to the last available file if parsing somehow completely failed
        best_match_url = all_matches[-1][0] + all_matches[-1][1]
    
    print(f"[{wavelength}] Selected closest image: {best_match_url.split('/')[-1]} (diff: {min_diff}s)")
    
    r_nrt = requests.get(best_match_url, timeout=60)
    r_nrt.raise_for_status()
    
    from io import BytesIO
    # Load JP2 directly with PIL
    img = Image.open(BytesIO(r_nrt.content))
    data = np.array(img).astype(np.float32) / 255.0
    
    if wavelength == 304:
        data = np.clip(data * 1.75, 0.0, 1.0)
    
    import sunpy.visualization.colormaps as cm
    cmap = cm.cmlist.get(f'sdoaia{wavelength}')
    if cmap is not None:
        rgba = cmap(data)
        data_8bit = (rgba[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(data_8bit).convert("L")
    else:
        img = img.convert("L")
    
    img = img.resize((2048, 2048), Image.Resampling.LANCZOS)
    data_8bit = np.array(img)
    
    if SAVE_DEBUG_IMAGES:
        debug_dir = Path(__file__).parent / 'debug_images'
        debug_dir.mkdir(exist_ok=True)
        Image.fromarray(data_8bit).save(debug_dir / f'AIA_{wavelength}_{actual_timestamp.strftime("%Y%m%d_%H%M%S")}.png')
    
    return data_8bit.flatten(), actual_timestamp, wavelength, 2048, 2048


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
    hmi_data, hmi_timestamp, hmi_width, hmi_height = download_hmi_nrt()
    
    if hmi_data is None:
        raise Exception("Failed to download HMI NRT data")
    
    aia_wavelengths = [131, 171, 193, 304, 1700]
    aia_downloads = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(download_and_process, hmi_timestamp, wl): wl 
            for wl in aia_wavelengths
        }
        
        for future in as_completed(futures):
            result = future.result()
            aia_downloads.append(result)
    
    all_downloads = aia_downloads + [(hmi_data, hmi_timestamp, 'HMI', hmi_width, hmi_height)]
    create_container_file(hmi_timestamp, all_downloads)
    
    timestamp_path = Path(__file__).parent / 'timestamp.txt'
    with open(timestamp_path, 'w') as f:
        f.write(hmi_timestamp.strftime('%Y-%m-%d %H:%M:%S'))
    
    successful = [d for d in all_downloads if d[0] is not None]
    container_path = Path(__file__).parent / 'solar.dat'
    file_size = container_path.stat().st_size
    
    print("\n" + "="*60)
    print("TIMESTAMP SUMMARY")
    print("="*60)
    print(f"Current time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("="*60)
    for wavelength in [131, 171, 193, 304, 1700, 'HMI']:
        download = next((d for d in successful if d[2] == wavelength), None)
        if download:
            ts = download[1]
            delta = (ts - hmi_timestamp).total_seconds() / 60
            print(f"{str(wavelength).rjust(5)}: {ts.strftime('%Y-%m-%d %H:%M:%S %Z')} (Δ {delta:+.1f} min)")
    print("="*60)
    print(f"Container: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    print("="*60)


if __name__ == '__main__':
    main()



