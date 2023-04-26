import numpy as np
import logging
import pathlib
from sample2 import somefun

# dir_path = pathlib.Path(__file__).parent.resolve()
# log_path = dir_path.joinpath("results.log")


logging.basicConfig(filename="results.log",
                    format="[%(asctime)s] [%(levelname)8s] --- %(message)s (%(filename)s:%(lineno)s)", datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.DEBUG,
                    filemode= 'w')

logger = logging.getLogger(__file__)
x = np.linspace(0,5)
logger.info("This message is from logger")
logging.info("Just an information")
logging.info("This is a result")
logging.info("This is a new line")

somefun(1)

# # print(x)

# importing module
# import logging
 
# # Create and configure logger
# logging.basicConfig(filename="newfile.log",
#                     format='%(asctime)s %(message)s',
#                     filemode='w')
 
# # Creating an object
# logger = logging.getLogger()
 
# # Setting the threshold of logger to DEBUG
# logger.setLevel(logging.DEBUG)
 
# # Test messages
# logger.debug("Harmless debug Message")
# logger.warning("Its a Warning")
# logger.error("Did you try to divide by zero")
# logger.critical("Internet is down")