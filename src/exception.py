import sys  # retrieves exception details
from src.logger import logging


def error_message_details(error, error_details:sys):
    """
    generates a custom error message with details about the error
    :param error: errors encountered
    :param error_details: sys module for retrieving exception details
    :return: formatted error message
    """
    # information about the execution error, what line, file etc. will be stored in exc_tb
    _,_,exc_tb = error_details.exc_info()
    # the filename
    filename = exc_tb.tb_frame.f_code.co_filename
    error_message ="Error occurred in python script name: [{0}] line number: [{1}] error message: [{2}]".format(
        filename,exc_tb.tb_lineno,str(error)
    )

    return error_message



# create my own custom exception class

class CustomException(Exception):
    """
     sometimes regular error messages aren’t detailed enough,and I want to make them more helpful.
    raise this CustomException when I catch an error, It uses the function (error_message_details)
    to create a detailed message about what went wrong
    """
    def __init__(self, error_message, error_details:sys):
        # Call the parent class's __init__ method
        super().__init__(error_message)
        # Generate and store the detailed error message
        self.error_message = error_message_details(error_message, error_details=error_details)


    def __str__(self):
        return self.error_message

#Test
#if __name__ == '__main__':
#    try:
#        p=0
#        f=0/p
#        print(f)
#    except Exception as e:
#        logging.info('divide by zero')
#        raise CustomException(e, sys)




