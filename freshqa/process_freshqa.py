import os
import glob
import pandas as pd
import requests
import json
from urllib.parse import urlparse

INPUT_DIR = "freshqa_archive"
OUTPUT_METADATA_FILE = "freshqa_metadata_with_views.csv"
OUTPUT_BENCHMARK_FILE = "freshqa_benchmark.jsonl"
    

def get_latest_file(directory):
    file_pattern = os.path.join(directory, "*.csv")
    files = glob.glob(file_pattern)
    if not files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    files.sort()
    return files[-1]

def extract_wiki_title(url):
    if pd.isna(url) or 'wikipedia.org/wiki/' not in str(url):
        return None
    try:
        title_part = str(url).split('/wiki/')[-1].split('#')[0].split('?')[0]
        return title_part
    except:
        return None

def extract_source_domain(url):
    if not url or pd.isna(url):
        return None
    try:
        if "://" not in str(url):
            url = "http://" + str(url)
        parsed = urlparse(url)
        return parsed.hostname
    except:
        return None

def get_pageviews(page_title, start_date="2025010100", end_date="2025123100"):
    access_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiIwMDFkMTFmNmQ2MzVmMGY4YmI3MDlkNWViN2ZhNDRlYiIsImp0aSI6IjMzOGQ0Mzc0YzNmZjE5NjBlZDkzNjIwNTdiYjMwYjExOWYzZTY2MzVkZjM3NmY3NDcyZjczMDcyMjNiYzU4ODFjODBkOTliOTZmMjAzZGNkIiwiaWF0IjoxNzEyNjEwMTg0LjY4OTIyNywibmJmIjoxNzEyNjEwMTg0LjY4OTIzLCJleHAiOjMzMjY5NTE4OTg0LjY4NzY1Mywic3ViIjoiNzUzODczODIiLCJpc3MiOiJodHRwczovL21ldGEud2lraW1lZGlhLm9yZyIsInJhdGVsaW1pdCI6eyJyZXF1ZXN0c19wZXJfdW5pdCI6NTAwMCwidW5pdCI6IkhPVVIifSwic2NvcGVzIjpbImJhc2ljIl19.YN0ZvSzsBuYe3Mg-r0C63cWxDXPU3GOCyspUqg4mMv27Qw1FJq9F9H6JKJAUMrqQxB-xyWZqpu8mekvMoxb3Ha5S2fpPbuM4gMB0JketqG2obaDd4QqgtJjg8KDYKwR8ieKoPRLDSHv3Tv4NcvIL-EvzjkRybqrukzQwttwuBUwxmlY8vhC1BZed7URt_-KhMYPsnNfJLSBeWivYJOmrqF2S04AOS0Egjul8Pz_yXAQ7q7aqpIwg6X2jod0ZN5h1gnmAvZmoLB7mKSAxrHEUL2zaQ8BVERWostWVA9ek556cuUJe5NusQ0XW7pcsYIi0YpFjKOBuq-tXzuOlbxFhlbwrp6xkhE_grQGNs1IxyT-w_sjQc2gI48FDe0ldDrTg6ZmgLELsjJM8xOxBy1ng1fY73p-QnaDdxX4hqRw2ZBDlZ1E2j84lvVrv62x_SHPiBNAeywEPcOqDRV_XbU6ArOyJ7QTZXRu9UOT0XDQ-Fx3maCRGb35W4aOtLSWL-SSXYLI8ZuOQ2BwKQQYYbEDMp0W7NjHWzh8YPv6Y2wDaMzsAqaxk2c36pNvTToiTc_P6_a56lydQwoT8ACx1kzzw5lTNPKPEPxPGNiMgtsL3VqtxJWMR7Lgq-ZKwI7cwQ5FTp2YriQDBYuvoaDQeG_eVh8BlNlyg26OYojtYbNos3os"
    client_id = "001d11f6d635f0f8bb709d5eb7fa44eb"
    client_secret = "630b434daa4c8f6cce03b1c294b59574c1ce9431"  # Example client secret
    headers = {
        'Authorization': f'Bearer {access_token}',
        'User-Agent': 'wikipagerank',
    }

    # Construct the API URL with the appropriate parameters
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{page_title}/daily/{start_date}/{end_date}"
    # Make the HTTP GET request to the API
    response = requests.get(url, headers=headers)
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        # Extract the pageview data
        print('retrieved for ', page_title)
        views = sum(item['views'] for item in data['items'])
        return views
    else:
        print(f"Failed to retrieve pageviews data for {page_title}. Status code: {response.status_code}")
        return 0

def parse_dataset(raw_file, output_benchmark_file="freshqa_benchmark.jsonl", output_metadata_file="freshqa_metadata_with_views.csv"):
    print(f"Processing file: {raw_file}")
    
    df = pd.read_csv(raw_file)
    
    df = df[df['split'] == "TEST"].copy()
    
    answer_cols = [f"answer_{i}" for i in range(10)]
    
    df["target_aliases"] = df[answer_cols].apply(lambda row: [v for v in row if pd.notna(v)], axis=1)
    df["target_type"] = df["target_aliases"].apply(lambda aliases: ", ".join(aliases))
    df["target"] = df["answer_0"]
    
    dataset_name = os.path.splitext(os.path.basename(raw_file))[0]
    df["question_id"] = dataset_name + "_" + df["id"].astype(str)
    
    df["question_source"] = df["source"]
    df["question_source_domain"] = df["source"].apply(extract_source_domain)
    df["wiki_title"] = df["source"].apply(extract_wiki_title)
    
    print("Fetching page views...")
    df["page_view"] = df["wiki_title"].apply(get_pageviews)
    
    metadata_cols = [
        "question_id", "question", "wiki_title", "page_view", 
        "fact_type", "split", "false_premise", "effective_year"
    ]
    df[metadata_cols].to_csv(output_metadata_file, index=False)
    
    jsonl_cols = [
        "question", 
        "question_id", 
        "question_source", 
        "target", 
        "target_aliases", 
        "target_type"
    ]
    
    records = df[jsonl_cols].to_dict(orient="records")
    
    with open(output_benchmark_file, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    latest_file = get_latest_file(INPUT_DIR)
    parse_dataset(latest_file)