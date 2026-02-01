import os
import json
import csv
import yaml
from typing import Union, List, Dict


class Saver:
    def __init__(self, base_folder_path: str):
        """
        Initialize Saver with a base folder path.
        
        Args:
            base_folder_path: The base directory where files will be saved.
        """
        self.base_folder_path = base_folder_path

    def _get_full_path(self, file_path: str) -> str:
        """
        Return the full absolute path for the given file path.
        
        Args:
            file_path: The relative or absolute path of the file.
        
        Returns:
            Absolute file path.
        """
        return os.path.abspath(file_path) if os.path.isabs(file_path) else os.path.join(self.base_folder_path, file_path)

    def list_files(self, directory: str) -> List[str]:
        """
        List all files in the specified directory.
        
        Args:
            directory: The directory to list files from.
        
        Returns:
            List of file names in the directory.
        """
        return os.listdir(self._get_full_path(directory))

    def exists(self, file_path: str) -> bool:
        """
        Check if the specified file exists.
        
        Args:
            file_path: The relative or absolute path of the file.
        
        Returns:
            True if the file exists, False otherwise.
        """
        return os.path.exists(self._get_full_path(file_path))

    def ensure_directory_exists(self, file_path: str) -> None:
        """
        Ensure that the directory for the specified file path exists.
        
        Args:
            file_path: The relative or absolute path of the file.
        """
        directory = os.path.dirname(self._get_full_path(file_path))
        if not os.path.exists(directory):
            os.makedirs(directory)

    def save_json(self, data: Union[Dict, List], file_path: str) -> None:
        """
        Save data to a JSON file.
        
        Args:
            data: Data to save (must be a dictionary or list).
            file_path: The relative or absolute path of the file.
        """
        self._save_file(data, file_path, json.dump, indent=4)

    def save_csv(self, data: List[List[str]], file_path: str, headers: List[str] = None) -> None:
        """
        Save data to a CSV file.
        
        Args:
            data: Data to save (list of rows).
            file_path: The relative or absolute path of the file.
            headers: Optional headers for CSV file.
        """
        self._save_file(data, file_path, self._write_csv, headers=headers)

    def save_yaml(self, data: Union[Dict, List], file_path: str) -> None:
        """
        Save data to a YAML file.
        
        Args:
            data: Data to save (must be a dictionary or list).
            file_path: The relative or absolute path of the file.
        """
        self._save_file(data, file_path, yaml.dump, allow_unicode=True)

    def _save_file(self, data, file_path: str, writer, **kwargs) -> None:
        """
        Helper function to save data to a file using the specified writer function.
        
        Args:
            data: Data to save.
            file_path: The relative or absolute path of the file.
            writer: Function to use for saving data (json.dump, csv.writer, etc.).
            **kwargs: Additional arguments to pass to the writer function.
        """
        full_path = self._get_full_path(file_path)
        self.ensure_directory_exists(full_path)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            writer(data, f, **kwargs)

    def _write_csv(self, data, f, headers: List[str] = None) -> None:
        """
        Helper function to write data to a CSV file.
        
        Args:
            data: Data to save (list of rows).
            f: File object.
            headers: Optional headers for CSV file.
        """
        writer = csv.writer(f)
        if headers:
            writer.writerow(headers)
        writer.writerows(data)

    def read_file(self, file_path: str) -> Union[Dict, List, List[List[str]]]:
        """
        Read a file based on its extension and return the content.
        
        Args:
            file_path: The relative or absolute path of the file.
        
        Returns:
            Parsed content of the file.
        
        Raises:
            ValueError: If the file extension is unsupported.
        """
        _, ext = os.path.splitext(file_path)
        full_path = self._get_full_path(file_path)

        with open(full_path, 'r', encoding='utf-8') as f:
            if ext == '.json':
                return json.load(f)
            elif ext == '.csv':
                return list(csv.reader(f))
            elif ext in ('.yaml', '.yml'):
                return yaml.safe_load(f)
            elif ext == '.jsonl':
                return [json.loads(line.strip()) for line in f]
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
    
    def copy_file(self, source_file_path: str, target_file_path: str) -> None:
        """
        Copy content from the source file to the target file.
        
        Args:
            source_file_path: The relative or absolute path of the source file.
            target_file_path: The relative or absolute path of the target file.
        """
        data = self.read_file(source_file_path)
        self.save_data(data, target_file_path)

    def save_data(self, data: Union[Dict, List, List[List[str]]], target_file_path: str) -> None:
        """
        Save data to the target file, determining format based on the file extension.
        
        Args:
            data: The data to save (dictionary, list, or list of rows).
            target_file_path: The relative or absolute path of the file.
        
        Raises:
            ValueError: If the file extension is unsupported.
        """
        _, ext = os.path.splitext(target_file_path)
        print(f"Saving data to {target_file_path}")
        if ext == '.json':
            self.save_json(data, target_file_path)
        elif ext == '.csv':
            self.save_csv(data, target_file_path)
        elif ext in ('.yaml', '.yml'):
            self.save_yaml(data, target_file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")