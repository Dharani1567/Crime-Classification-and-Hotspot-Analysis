"""
download_datasets.py
Programmatically download raw crime datasets for:
1. Chicago (data.cityofchicago.org - endpoint: ijzp-q8t2)
2. Los Angeles (data.lacity.gov - endpoint: 2nrs-mtv8)
3. San Francisco (data.sfgov.org - endpoint: wg3w-h783)
4. UK (data.police.uk - street-level crime endpoint: crimes-street/all-crime)

Handles Socrata limit/offset pagination, rate limiting, and saves unadulterated raw data to data/raw/.
"""

import os
import json
import time
import requests

HEADERS = {
    'User-Agent': 'Crime-ML-Research-Project/1.0 (academic research)'
}

def download_socrata(base_url, output_file, target_records=100000, page_size=50000):
    print(f"\n--- Starting download for {base_url} ---")
    if os.path.exists(output_file) and os.path.getsize(output_file) > 100000:
        print(f"File {output_file} already exists ({os.path.getsize(output_file):,} bytes). Skipping.")
        return

    all_records = []
    offset = 0
    while len(all_records) < target_records:
        fetch_limit = min(page_size, target_records - len(all_records))
        url = f"{base_url}?$limit={fetch_limit}&$offset={offset}"
        print(f"Fetching offset {offset:,} with limit {fetch_limit:,}...")
        
        retries = 3
        while retries > 0:
            try:
                res = requests.get(url, headers=HEADERS, timeout=45)
                if res.status_code == 200:
                    records = res.json()
                    break
                else:
                    print(f"Warning: HTTP {res.status_code}. Retrying...")
                    retries -= 1
                    time.sleep(3)
            except Exception as e:
                print(f"Error fetching {url}: {e}. Retrying...")
                retries -= 1
                time.sleep(3)
        else:
            print("Failed after maximum retries.")
            break

        if not records:
            print("No more records returned from API.")
            break

        all_records.extend(records)
        print(f"Downloaded {len(records):,} records. Total accumulated: {len(all_records):,}")
        offset += len(records)
        if len(records) < fetch_limit:
            print("Reached end of data stream.")
            break
        time.sleep(1)

    print(f"Writing {len(all_records):,} raw records to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_records, f, indent=2)
    print(f"Saved {output_file} successfully ({os.path.getsize(output_file):,} bytes).")

def download_uk_crimes(output_file):
    print("\n--- Starting download for UK Police Data (data.police.uk) ---")
    if os.path.exists(output_file) and os.path.getsize(output_file) > 100000:
        print(f"File {output_file} already exists ({os.path.getsize(output_file):,} bytes). Skipping.")
        return

    # High-density urban key points across major UK police force jurisdictions
    locations = [
        ('London_Central', 51.5074, -0.1278),
        ('London_Westminster', 51.4975, -0.1357),
        ('London_Southwark', 51.5033, -0.0895),
        ('London_Camden', 51.5290, -0.1255),
        ('London_Hackney', 51.5450, -0.0553),
        ('London_Croydon', 51.3762, -0.0982),
        ('Birmingham_City', 52.4862, -1.8904),
        ('Birmingham_South', 52.4450, -1.9300),
        ('Leeds_City', 53.8008, -1.5491),
        ('Leeds_Headingley', 53.8180, -1.5790),
        ('Liverpool_City', 53.4084, -2.9916),
        ('Liverpool_North', 53.4400, -2.9700),
        ('Bristol_City', 51.4545, -2.5879),
        ('Bristol_Clifton', 51.4580, -2.6180),
        ('Newcastle_City', 54.9783, -1.6178),
        ('Sheffield_City', 53.3811, -1.4701),
        ('Nottingham_City', 52.9548, -1.1581),
        ('Leicester_City', 52.6369, -1.1398),
        ('Southampton_City', 50.9097, -1.4044),
        ('Cardiff_City', 51.4816, -3.1791)
    ]
    
    # 2024 calendar months
    months = ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06']
    
    seen_ids = set()
    all_uk_records = []
    
    total_queries = len(locations) * len(months)
    query_count = 0
    
    for loc_name, lat, lng in locations:
        for month in months:
            query_count += 1
            url = f"https://data.police.uk/api/crimes-street/all-crime?lat={lat}&lng={lng}&date={month}"
            retries = 3
            batch = []
            while retries > 0:
                try:
                    res = requests.get(url, headers=HEADERS, timeout=30)
                    if res.status_code == 200:
                        batch = res.json()
                        break
                    elif res.status_code == 503: # Rate limit
                        print(f"503 Rate limit encountered on {loc_name} {month}. Backing off...")
                        time.sleep(5)
                        retries -= 1
                    else:
                        print(f"HTTP {res.status_code} on {loc_name} {month}. Retrying...")
                        time.sleep(2)
                        retries -= 1
                except Exception as e:
                    print(f"Exception on {loc_name} {month}: {e}")
                    time.sleep(2)
                    retries -= 1
            
            new_in_batch = 0
            for item in batch:
                rec_id = item.get('id')
                if rec_id and rec_id not in seen_ids:
                    seen_ids.add(rec_id)
                    all_uk_records.append(item)
                    new_in_batch += 1
                elif not rec_id:
                    # Some crimes lack explicit ID, keep if valid
                    all_uk_records.append(item)
                    new_in_batch += 1

            print(f"[{query_count}/{total_queries}] {loc_name} ({month}): retrieved {len(batch)} crimes ({new_in_batch} unique). Total: {len(all_uk_records):,}")
            time.sleep(0.4)

    print(f"Writing {len(all_uk_records):,} raw UK records to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_uk_records, f, indent=2)
    print(f"Saved {output_file} successfully ({os.path.getsize(output_file):,} bytes).")

def main():
    print("="*60)
    print("STARTING PROGRAMMATIC CRIME DATASET ACQUISITION")
    print("="*60)
    
    # 1. Chicago Crime Dataset
    download_socrata(
        base_url="https://data.cityofchicago.org/resource/ijzp-q8t2.json",
        output_file="data/raw/chicago/chicago_crimes_raw.json",
        target_records=100000,
        page_size=50000
    )
    
    # 2. Los Angeles Crime Dataset
    download_socrata(
        base_url="https://data.lacity.gov/resource/2nrs-mtv8.json",
        output_file="data/raw/los_angeles/los_angeles_crimes_raw.json",
        target_records=100000,
        page_size=50000
    )
    
    # 3. San Francisco Crime Dataset
    download_socrata(
        base_url="https://data.sfgov.org/resource/wg3w-h783.json",
        output_file="data/raw/san_francisco/san_francisco_crimes_raw.json",
        target_records=100000,
        page_size=50000
    )
    
    # 4. UK Crime Dataset
    download_uk_crimes(
        output_file="data/raw/uk/uk_crimes_raw.json"
    )
    
    print("\nALL FOUR DATASETS SUCCESSFULLY ACQUIRED AND SAVED IN data/raw/")

if __name__ == '__main__':
    main()
