#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import struct
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
from sunpy.net import Fido, attrs as a
from sunpy.map import Map
import astropy.units as u
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from astropy.time import Time, TimeDelta
from scipy.ndimage import zoom
import drms
from astropy.io import fits
import requests
import traceback
from PIL import Image

warnings.filterwarnings('ignore')


def download_hmi_nrt():
    try:
        client = drms.Client()
        series = 'hmi.m_720s_nrt'
        
        now = datetime.now(timezone.utc)
        query_start = (now - timedelta(hours=24)).strftime('%Y.%m.%d_%H:%M:%S_TAI')
        query_end = now.strftime('%Y.%m.%d_%H:%M:%S_TAI')
        
        result = client.query(f'{series}[{query_start}-{query_end}]', key='T_REC', seg='magnetogram')
        
        if isinstance(result, tuple):
            keys, segments = result
        else:
            keys = result
            segments = None
        
        if len(keys) == 0:
            return None, None, 0, 0
        
        # Get the most recent record
        latest_rec = keys.iloc[-1]['T_REC']
        
        # NRT data is directly accessible via URL without export
        # Get magnetogram segment path from segments dataframe
        if segments is not None and 'magnetogram' in segments.columns:
            magnetogram_path = segments.iloc[-1]['magnetogram']
            base_url = "http://jsoc.stanford.edu"
            fits_url = f"{base_url}{magnetogram_path}"
            
            response = requests.get(fits_url, timeout=60)
            response.raise_for_status()
            
            temp_file = f"temp_nrt_{latest_rec.replace('.', '_').replace(':', '_')}.fits"
            with open(temp_file, 'wb') as f:
                f.write(response.content)
            
            with fits.open(temp_file) as hdul:
                if len(hdul) > 1 and hdul[1].data is not None:
                    data = hdul[1].data
                    header = hdul[1].header
                else:
                    data = hdul[0].data
                    header = hdul[0].header
            
            actual_timestamp = None
            
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
            
            current_shape = data_8bit.shape
            zoom_factors = (2048 / current_shape[0], 2048 / current_shape[1])
            resized = zoom(data_8bit, zoom_factors, order=3)
            resized = resized[:2048, :2048]
            
            debug_dir = Path(__file__).parent / 'debug_images'
            debug_dir.mkdir(exist_ok=True)
            Image.fromarray(resized).save(debug_dir / f'HMI_{actual_timestamp.strftime("%Y%m%d_%H%M%S")}.png')
            
            return resized.flatten(), actual_timestamp, resized.shape[1], resized.shape[0]
        
    except Exception as e:
        print(f"Error downloading HMI NRT data: {e}")
        traceback.print_exc()
        return None, None, 0, 0


def download_and_process(timestamp, wavelength, detector='AIA'):
    try:
        end_time = Time(timestamp) + TimeDelta(30 * u.minute)
        start_time = Time(timestamp) - TimeDelta(30 * u.minute)
        
        result = Fido.search(
            a.Time(start_time, end_time),
            a.Instrument('AIA'),
            a.Wavelength(wavelength * u.angstrom),
            a.Sample(12 * u.minute)
        )
        
        if len(result) == 0 or len(result[0]) == 0:
            return None, timestamp, wavelength, 0, 0
        
        index = len(result[0]) // 2
        downloaded_files = Fido.fetch(result[0, index], path='./{file}', progress=False)
        
        solar_map = Map(downloaded_files[0])
        actual_timestamp = solar_map.date.to_datetime()
        
        if actual_timestamp.tzinfo is None:
            actual_timestamp = actual_timestamp.replace(tzinfo=timezone.utc)
        
        aia_ranges = {131: (7, 1200), 171: (10, 6000), 193: (120, 6000), 304: (50, 2000), 1700: (200, 2500)}
        
        data = solar_map.data
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        if data is None or data.size == 0:
            return None, timestamp, wavelength, 0, 0
        
        vmin = data.min()
        vmax = data.max()
        data_normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
        data_8bit = (data_normalized * 255).astype(np.uint8)
        
        current_shape = data_8bit.shape
        zoom_factors = (2048 / current_shape[0], 2048 / current_shape[1])
        resized = zoom(data_8bit, zoom_factors, order=3)
        resized = resized[:2048, :2048]
        
        debug_dir = Path(__file__).parent / 'debug_images'
        debug_dir.mkdir(exist_ok=True)
        Image.fromarray(resized).save(debug_dir / f'AIA_{wavelength}_{actual_timestamp.strftime("%Y%m%d_%H%M%S")}.png')
        
        return resized.flatten(), actual_timestamp, wavelength, resized.shape[1], resized.shape[0]
        
    except Exception as e:
        print(f"Error downloading {wavelength}A: {e}")
        return None, timestamp, wavelength, 0, 0, 0, 0


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
    """Main entry point."""
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



