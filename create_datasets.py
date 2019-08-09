"""Rations data set for training and testing purpose."""
import os
from shutil import copyfile
import sys


# Takes the list of ratios in which to divide the data in list format. All ratios must sum to 1.
def create_sets(ratios, source, destination):
    """Creates different sets with number of samples rationed as per input ratios
    Parameters: ratio, source folder, destination folder
   """
    files_copied = set()
    if sum(ratios) != 1:
        print('Ratio should sum to 1')

    else:
        for path, _b, _f in os.walk(source):
            for d_set, ratio in enumerate(ratios):
                if path != source:
                    bucket_path, _, files = next(os.walk(path))
                    # print(bucket_path, dirs, files)
                    n_files = len(files)
                    # print(bucket_path, n_files)

                    bucket = [x for x in bucket_path.split('/') if x != ''][-1]

                    n_files_copied = 0

                    for file in files:
                        if file not in files_copied:
                            # print(bucket_path,file)
                            dest = os.path.join(destination, 'set_'+str(d_set), bucket)

                            if not os.path.exists(dest):
                                os.makedirs(dest)

                            file_src = os.path.join(bucket_path, file)
                            file_dst = os.path.join(dest, file)

                            print(file_src, file_dst)
                            copyfile(file_src, file_dst)
                            files_copied.add(file)

                            n_files_copied += 1
                            if n_files_copied == int(ratio*n_files):
                                # print('files in set_'+str(d_set), n_files_copied)
                                break


def main():
    """Driver code"""

    print(len(sys.argv))

    if len(sys.argv) > 1 and len(sys.argv) == 4:
        print(sys.argv[1],)
        ratios = [float(x) for x in sys.argv[1].strip('[]').split(',')]
        print(type(ratios))
        src = sys.argv[2]
        dst = sys.argv[3]

        create_sets(ratios, src, dst)
        print('Done')
    else:
        print('Takes 3 arguments: list of ratios, source folder, target folder')


if __name__ == '__main__':
    main()
