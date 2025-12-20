import sys
import logging
from src.components.handling import logger

'''
Exception handing


Handles and logs errors
'''

def error_message_details(error, error_det:sys):
    _,_,exc_tb = error_det.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured! in {0}, line number {1} threw error : {2}".format(
    filename,exc_tb.tb_lineno,str(error) 
    )
    return error_message

class CustomException(Exception):
    def __init__(self,error_message, error_det:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message,error_det)
        logging.info(f'Error : {error_message}')
        
    
    def __str__(self):
        
        return self.error_message
    
