import argparse
import os
import subprocess
import time
from os import listdir, mkdir, path, sep
from os.path import isfile, join
from typing import List


beast_command = "beast index iformat:envelope separator:, gindex:grid pcriterion:'size(" # m)' "

def index(path: str, output_path: str, sizes: List): 
    # create output directory if not exists
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    log = open(output_path + '/log_indexing', 'w')
    # explore file tree
    files = get_files_path(path)
    log.write('Found {0} files'.format(len(files)))
    log.write('Selected size: [s: {0}, sm: {1}, m: {2}, ml: {3}, l: {4}]'.format(sizes[0], sizes[1], sizes[2], sizes[3], sizes[4]))
    print('Found {0} files'.format(len(files)))
    print('Selected size: [s: {0}, sm: {1}, m: {2}, ml: {3}, l: {4}]'.format(sizes[0], sizes[1], sizes[2], sizes[3], sizes[4]))
    start_time = time.time()
    count = 0
    # for each file, launch Beast
    for structure in files:
        size, dir_path = get_correct_size(structure, sizes)
        log.write("Indexing of: {0} with size: {1}".format(structure.split('/')[-2], size))
        print("Indexing of: {0} with size: {1}".format(structure.split('/')[-2], size))
        subprocess.check_output(beast_command + size + ")' " + structure + ' ' + output_path + '/' + dir_path +'_grid > /dev/null 2>&1', shell=True)
        count += 1
        log.write("{0}/{1}".format(count, len(files)))
        print("{0}/{1}".format(count, len(files)))
    log.write("Total time: {0} seconds".format(time.time() - start_time))
    print("Total time: {0} seconds".format(time.time() - start_time))
    log.close()


# check size
def get_correct_size(name: str, sizes: List):
    # check if size is declared in subdir or in the main dir
    splitted = name.split('/')
    for i in range(len(splitted)-1, 0, -1):
        result = check_size_abbreviation(splitted[i])
        if result < 0:
            # altro metodo
            result, abbreviation = check_size_extended(splitted[i])
            if result >= 0:
                # controllo se sono nel caso large o small/medium
                if abbreviation == 'l':
                    return str(sizes[result]), splitted[i+1].lower()+'_' + abbreviation + '_' + splitted[-1].split('.')[0].split('-')[1]
                else:
                    return str(sizes[result]), splitted[i+1].lower().split('.')[0]+'_'+abbreviation
        else: 
            return str(sizes[result]), splitted[i]
    return Exception("Cannot recognize a valid pattern (s, sm, m, ml, l)")

def check_size_extended(name: str):
    splitted = name.split('_')
    splitted = splitted[0]
    if splitted == 'small':
        return 0, 's'
    elif splitted == 'medium':
        return 2, 'm'
    elif splitted == 'large':
        return 4, 'l'
    # which size for real
    elif splitted == 'real':
        return 4, 'r'
    else:
        return -1, 'e'

def check_size_abbreviation(name: str):
    splitted = name.split('_')
    splitted = splitted[len(splitted) - 1]
    if splitted == 's':
        return 0
    elif splitted == 'sm':
        return 1
    elif splitted == 'm':
        return 2
    elif splitted == 'ml':
        return 3
    elif splitted == 'l':
        return 4
    else:
        return -1

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automatic indexing of datasets with Beast')

    parser.add_argument('--path', '-p', help='path of the dir with all datasets')
    parser.add_argument('--size-small', '-s', default='128k', help='size and unit of measure of the partitioning of small datasets')
    parser.add_argument('--size-small-medium', '-sm', default='5m', help='size and unit of measure of the partitioning of small medium datasets')
    parser.add_argument('--size-medium', '-m', default='10m', help='size and unit of measure of the partitioning of medium datasets')
    parser.add_argument('--size-medium-large', '-ml', default='64m', help='size and unit of measure of the partitioning of medium large datasets')
    parser.add_argument('--size-large', '-l', default='128m', help='size and unit of measure of the partitioning of large datasets')
    parser.add_argument('--output', '-o', default='./dataset_grid', help='output folder')

    args = parser.parse_args()

    index(args.path, args.output, [args.size_small, args.size_small_medium, args.size_medium, args.size_medium_large, args.size_large]) 

    # to do: aggiungere il mkdir per le cartelle dataset_grid e co