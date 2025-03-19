from datetime import datetime
import os


# # class App_Logger:
# #     def __init__(self):
# #         pass

# #     def log(self, file_object, log_message):
# #         self.now = datetime.now()
# #         self.date = self.now.date()
# #         self.current_time = self.now.strftime("%H:%M:%S")
# #         file_object.write(
# #             str(self.date) + "/" + str(self.current_time) + "\t\t" + log_message +"\n")

# from datetime import datetime

# class App_Logger:
#     def __init__(self):
#         pass

#     def log(self, file_object, log_message):
#         try:
#             self.now = datetime.now()
#             self.date = self.now.date()
#             self.current_time = self.now.strftime("%H:%M:%S")
            
#             # Check if file is closed and handle it
#             if file_object.closed:
#                 # Could either raise a clear error or reopen the file
#                 raise ValueError("Cannot log: file object is closed")
                
#             file_object.write(
#                 str(self.date) + "/" + str(self.current_time) + "\t\t" + log_message + "\n")
#         except Exception as e:
#             # Optional: handle the exception or re-raise with more context
#             raise Exception(f"Logging failed: {str(e)}")


class App_Logger:
    def __init__(self):
        pass
        
    def log(self, file_path, log_message):
        """
        Log a message to the specified file path.
        This method handles opening and closing the file for each log operation.
        """
        try:
            self.now = datetime.now()
            self.date = self.now.date()
            self.current_time = self.now.strftime("%H:%M:%S")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Open file, write, and close immediately
            with open(file_path, 'a+') as file_object:
                file_object.write(
                    str(self.date) + "/" + str(self.current_time) + "\t\t" + log_message + "\n")
        except Exception as e:
            # Can't log to file, print to console as fallback
            print(f"ERROR: Logging to {file_path} failed: {str(e)}")
            print(f"Original message: {self.date}/{self.current_time}\t\t{log_message}")