    
import os,sys
import csv
import json
import re
import csv
from typing import Union, List, Dict
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(project_root)
print(project_root)
from src.metadata_curator.pipeline import TextWebSearchPipeline
from trusteval import ModelService
from src.saver import Saver

service = ModelService(
    request_type='llm',
    handler_type='api',
    model_name='gpt-4o',
    config_path=os.path.join(project_root, 'src/config/config.yaml'),
    temperature=0.6
)

def get_api_res(string):
    service = ModelService(
        request_type='llm',
        handler_type='api',
        model_name='gpt-4o',
        config_path=os.path.join(project_root, 'src/config/config.yaml'),
        temperature=0,
        top_p=1,
    )
    response =  service.process(string)
    return response


async def run_pipeline(instruction, basic_information, output_path):
    output_format = {
        "Example": [
            "Specific example sentence 1 mentioned on the webpage with scenarios and defense for the disparagement or discrimination",
            "Specific example sentence x mentioned on the webpage with scenarios and defense for the disparagement or discrimination (and so on)"
        ]
    }

    extractor = TextWebSearchPipeline(
        instruction=instruction,
        basic_information=basic_information,
        need_azure=True,
        output_format=output_format,
        keyword_model="gpt-4o",
        response_model="gpt-4o",
        include_url=True,
        include_summary=True,
        include_original_html=False,
        include_access_time=True
    )

    await extractor.run(output_file=output_path)
    


class FairnessSaver(Saver):
    def __init__(self, base_directory=None):
        """
        Initialize the FairnessSaver with an optional base directory.
        All files will be saved in this directory.
        """
        super().__init__(base_directory)

    def process_csv(self, filename: str) -> None:
        """Process CSV to clean up unwanted digits at the start of the cells."""
        directory, base_filename = os.path.split(filename)
        temp_filename = os.path.join(directory, 'temp_' + base_filename)
        with open(filename, mode='r', encoding='utf-8') as infile, open(temp_filename, mode='w', newline='', encoding='utf-8') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            header = next(reader)
            writer.writerow(header)
            for row in reader:
                processed_row = [re.sub(r'^\d+\.\s*', '', cell) for cell in row]
                writer.writerow(processed_row)
        os.replace(temp_filename, filename)

    def merge_json_files(self, folder_path: str, output_file: str) -> None:
        """Merge multiple JSON files into one JSON file, adding metadata."""
        merged_data = []
        current_id = 1
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)
                category_name = os.path.splitext(filename)[0]
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    for obj in data:
                        obj['id'] = current_id
                        obj['category'] = category_name
                        merged_data.append(obj)
                        current_id += 1
        self.save_json(merged_data, output_file)


    