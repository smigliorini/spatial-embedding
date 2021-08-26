import argparse
import os
import subprocess
import time
from os import listdir, mkdir, path, sep
from os.path import isfile, join


beast_command = 'beast summary iformat:envelope separator:, '
output_on_file_command = '.txt 2>&1' #prima metti namefile del log

def summarize(path: str, logs_path: str):
    # create logs directory if not exists
    if not os.path.isdir(logs_path):
        os.mkdir(logs_path)
    # explore file tree
    files = get_files_path(path)
    print('Found {0} files'.format(len(files)))
    start_time = time.time()
    count = 0
    # for each file, launch Beast
    for structure in files:
        print("Summarization of: {0}.txt".format(structure.split('/')[-2]))
        subprocess.check_output(beast_command + structure + '> ' + args.logs + '/' + structure.split('/')[-2] + output_on_file_command, shell=True)
        count += 1
        print("{0}/{1}".format(count, len(files)))
    print("Total time: {0} seconds".format(time.time() - start_time))

# composite explorer
def get_files_path(path: str):
    list_files_paths = []
    for structure in listdir(path):
        sub_path = join(path, structure)
        if isfile(sub_path):
            list_files_paths.append(sub_path)
        else:
            list_files_paths = list_files_paths + get_files_path(sub_path)
    return list_files_paths

# CSV generator
def generate_csv_summary(logs_path: str, output_file: str, separator: str):
    files = get_files_path(logs_path)
    output = open(output_file, "w")
    # write the heading with the proper separator
    heading = "file name,cardinality,size,X min,Y min,X max,Y max,\n".replace(',', separator)
    output.write(heading)
    # start parsing logs
    for file in files:
        f = open(file, 'r')
        text = f.read()
        # extract extent
        extent = extract_data_array('"extent"', text).split(',')
        # extract size
        size = extract_data('"size"', text)
        # extract cardinality
        cardinality = extract_data('"num_non_empty_features"', text)
        # file name without extension
        line = "{0}{1}".format(file.replace(logs_path + '/', '').split('.txt')[0], separator)
        line = line + "{0}{1}".format(cardinality, separator)
        line = line + "{0}{1}".format(size, separator)
        line = line + "{0}{1}".format(extent[0], separator)
        line = line + "{0}{1}".format(extent[1], separator)
        line = line + "{0}{1}".format(extent[2], separator)
        line = line + "{0}{1}".format(extent[3], separator) + "\n"
        output.write(line)
        f.close()
    output.close()


def extract_data_array(key: str, text: str):
    key_index = text.find(key)
    key_value = text[text.find('[', key_index) + 1 : text.find(']', key_index)]
    key_value = key_value.replace(' ', '')
    return key_value


def extract_data(key: str, text: str):
    key_index = text.find(key)
    key_value = text[text.find(':', key_index) + 1 : text.find(',', key_index)]
    key_value = key_value.replace(' ', '')
    return key_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automatic summary of datasets with Beast')

    parser.add_argument('--path', '-p', help='path of the dir with all datasets')
    parser.add_argument('--output', '-o', default='summary_generated.csv', help='output filename')
    parser.add_argument('--logs', '-l', default='./logs', help='logs dir path')
    parser.add_argument('--separator', '-s', default=';', help='separator used in CSV')

    args = parser.parse_args()

    summarize(args.path, args.logs)
    generate_csv_summary(args.logs, args.output, args.separator)
 