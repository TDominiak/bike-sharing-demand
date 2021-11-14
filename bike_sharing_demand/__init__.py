__version__ = 'v1.0.0'

from bike_sharing_demand.data_loader.data_loader import check_data_folder
from bike_sharing_demand.log import setup_logger

setup_logger()
check_data_folder()
