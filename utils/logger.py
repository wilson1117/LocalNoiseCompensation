import os
import json

class Logger:
    def __init__(self, log_dir):
        if os.path.exists(log_dir):
            count = 1
            while os.path.exists(log_dir + f"_{count}"):
                count += 1
            
            log_dir += f"_{count}"


        self.log_dir = log_dir

        os.makedirs(self.log_dir, exist_ok=True)

    def create_folder(self, name):
        folder = os.path.join(self.log_dir, name)
        os.makedirs(folder, exist_ok=True)

        return folder

    def create_log(self, name):
        return self.SubLogger(self.log_dir, name)

    class SubLogger:
        def __init__(self, folder, name):
            if os.path.exists(os.path.join(folder, name)):
                count = 1
                split_name = name.split(".")
                name = split_name[0]
                ext = ".".join(split_name[1:])
                if ext:
                    ext = "." + ext

                while os.path.exists(os.path.join(folder, "{}_{}{}".format(name, count, ext))):
                    count += 1
                
                name += "{}_{}{}".format(name, count, ext)

            self.file = open(os.path.join(folder, name), "w")

        def write_json(self, data):
            json.dump(data, self.file, indent=4)

        def log(self, message):
            self.file.write(message + "\n")

        def close(self):
            self.file.close()