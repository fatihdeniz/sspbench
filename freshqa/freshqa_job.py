import requests
import re
import os
import csv
from datetime import datetime

from process_freshqa import parse_dataset

README_URL = "https://raw.githubusercontent.com/freshllms/freshqa/main/README.md"
DATA_DIR = "freshqa_archive"   
HISTORY_FILE = "download_history.txt" 

def get_google_sheet_export_url(view_url):
    match = re.search(r"/d/([a-zA-Z0-9-_]+)", view_url)
    if match:
        doc_id = match.group(1)
        return f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=csv"
    return None

def fetch_readme():
    print("Fetching README from GitHub...")
    response = requests.get(README_URL)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to fetch README. Status: {response.status_code}")
        return None

def load_history():
    """Loads the list of previously downloaded dates."""
    if not os.path.exists(HISTORY_FILE):
        return set()
    with open(HISTORY_FILE, 'r') as f:
        return set(line.strip() for line in f.readlines())

def update_history(date_str):
    with open(HISTORY_FILE, 'a') as f:
        f.write(f"{date_str}\n")

def clean_and_save_csv(content_bytes, filepath):
    """
    Decodes the downloaded content, finds the real header, 
    removes the warning lines above it, and saves the file.
    """
    try:
        # Decode bytes to string
        text = content_bytes.decode('utf-8')
        lines = text.splitlines()
        
        start_index = 0
        header_found = False

        # Loop through lines to find where the actual data starts
        # We look for the standard header "id,split,question"
        for i, line in enumerate(lines):
            # We check startswith because sometimes headers have extra columns/spaces
            if line.startswith("id,split,question") or line.startswith('"id","split","question"'):
                start_index = i
                header_found = True
                break
        
        if header_found:
            # Slice the list to keep only the header and the data below it
            cleaned_content = "\n".join(lines[start_index:])
            
            with open(filepath, 'w', encoding='utf-8', newline='') as f:
                f.write(cleaned_content)
            print(f"Saved cleaned file (removed {start_index} garbage lines).")
        else:
            # Fallback: If we can't find the header, save as is but warn the user
            print("Warning: Could not find standard header. Saving raw file.")
            with open(filepath, 'wb') as f:
                f.write(content_bytes)

    except Exception as e:
        print(f"Error cleaning file: {e}")

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    downloaded_set = load_history()
    readme_content = fetch_readme()
    
    if not readme_content:
        return

    # Pattern: [FreshQA Month Day, Year](URL)
    pattern = r"\[FreshQA (.*?)\]\((https://docs\.google\.com/spreadsheets/.*?)\)"
    matches = re.findall(pattern, readme_content)

    print(f"Found {len(matches)} datasets listed in the README.")
    new_downloads_count = 0

    for date_str, url in matches:
        clean_date_str = date_str.strip()
        
        if clean_date_str in downloaded_set:
            continue

        try:
            date_obj = datetime.strptime(clean_date_str, "%B %d, %Y")
            formatted_date = date_obj.strftime("%Y-%m-%d")
            
            export_url = get_google_sheet_export_url(url)
            if not export_url: continue

            filename = f"freshqa_{formatted_date}.csv"
            filepath = os.path.join(DATA_DIR, filename)

            print(f"Downloading new version: {clean_date_str} -> {filename}")
            
            file_resp = requests.get(export_url)
            if file_resp.status_code == 200:
                # CALL THE CLEANING FUNCTION HERE
                clean_and_save_csv(file_resp.content, filepath)
                
                processed_file = os.path.join(DATA_DIR,f"freshqa_{formatted_date}.jsonl")
                metadata_file = os.path.join(DATA_DIR,f"freshqa_metadata_{formatted_date}.csv")
                parse_dataset(filepath, output_benchmark_file=processed_file, output_metadata_file=metadata_file)

                update_history(clean_date_str)
                downloaded_set.add(clean_date_str)
                new_downloads_count += 1
            else:
                print(f"Failed to download. Status: {file_resp.status_code}")

        except ValueError:
            print(f"Skipping entry with unparseable date: {clean_date_str}")
        except Exception as e:
            print(f"Error processing {clean_date_str}: {e}")

    if new_downloads_count == 0:
        print("No new versions found. Your archive is up to date.")
    else:
        print(f"Successfully downloaded and cleaned {new_downloads_count} new versions.")

if __name__ == "__main__":
    main()