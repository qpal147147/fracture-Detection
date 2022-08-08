import os
import re

# Define the extensions supported by the library
image_types = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
video_types = ('.mkv', '.mp4')


def atof(text):
    try:
        ret_val = float(text)
    except ValueError:
        ret_val = text
    return ret_val


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    """
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]


def list_images(base_path, contains=None):
    # return the set of files that are valid
    return list_files(base_path, valid_exts=image_types, contains=contains)


def list_videos(base_path, contains=None):
    # return the set of files that are valid
    return list_files(base_path, valid_exts=video_types, contains=contains)


def list_files(base_path, valid_exts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(base_path):
        filenames.sort(key=natural_keys)
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the file does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind('.'):].lower()

            # check to see if the file is an image and should be processed
            if valid_exts is None or ext.endswith(valid_exts):
                # construct the path to the image and yield it
                file_path = os.path.join(rootDir, filename)
                yield file_path
