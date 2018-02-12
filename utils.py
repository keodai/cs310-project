import os


def format_string(s):
    return '' if s is None else s.encode('utf-8')


def visible(src_path):
    return [os.path.join(src_path, file)
            for file in os.listdir(src_path)
            if not file.startswith('.') and os.path.isfile(os.path.join(src_path, file))]