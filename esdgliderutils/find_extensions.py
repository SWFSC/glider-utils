import os
import pathlib

def find_extensions(dir_path): #,  excluded = ['', '.txt', '.lnk']):
    """
    Get all the file extensions in the given directory
    From https://stackoverflow.com/questions/45256250
    """
    extensions = set()
    for _, _, files in os.walk(dir_path):   
        for f in files:
            extensions.add(pathlib.Path(f).suffix)
            # ext = Path(f).suffix.lower()
            # if not ext in excluded:
            #     extensions.add(ext)
    return extensions
