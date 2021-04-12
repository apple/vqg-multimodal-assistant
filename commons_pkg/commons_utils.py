"""
This script contains the utility functions
"""
import logging
import sys
import yaml
logger = logging.getLogger(__name__)


def load_config(config_file_path):
    with open(config_file_path, 'r') as config_file:
        try:
            return yaml.safe_load(config_file)
        except yaml.YAMLError as exc:
            logger.error("Exception thrown while parsing the config yaml file: %s" % exc)
