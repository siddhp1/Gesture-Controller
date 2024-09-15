import os
import json
from typing import Union


class Config:
    """
    Class to handle configuration file operations.
    """

    def __init__(self, config_path: str) -> None:
        """
        Initialize the config instance.
        """

        self.config_path = config_path
        self.config_data = self._load_config()

    def _load_config(self) -> dict:
        """
        Load the configuration file into a dictionary.
        """

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r") as config_file:
            return json.load(config_file)

    def get_value(self, key: str) -> Union[str, float, int, dict]:
        """
        Get a specific configuration value.
        """

        return self.config_data.get(key)

    def update_value(self, key: str, value: Union[str, float, int, dict]) -> None:
        """
        Update a configuration value in the dictionary and save it to the file.
        """

        self.config_data[key] = value
        self._save_config()

    def _save_config(self) -> None:
        """
        Save the configuration data back to the file.
        """

        with open(self.config_path, "w") as config_file:
            json.dump(self.config_data, config_file, indent=4)
