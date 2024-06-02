import os
import re
import argparse

def replace_in_files(folder_path, source_string, target_string):
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            with open(filepath, 'r') as file:
                contents = file.read()

            new_contents = re.sub(source_string, target_string, contents, flags=re.IGNORECASE)

            with open(filepath, 'w') as file:
                file.write(new_contents)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace text within files in a folder.")
    parser.add_argument("folder_path", help="The path to the folder containing the files.")
    parser.add_argument("source_string", help="The string to be replaced.")
    parser.add_argument("target_string", help="The replacement string.")

    args = parser.parse_args()

    replace_in_files(args.folder_path, args.source_string, args.target_string)