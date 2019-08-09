import os
import csv
from shutil import copyfile

buckets = ['Hardware', 'Other', 'Script', 'Software', 'Tools']


# Consider only the DRs with JT and TOBY logs in them
def file_copy(src, dest):
    for path, dirs, files in os.walk(src):
        if any('output.xml' in item for item in files):  # if TOBY
            for item in files:
                if (item.endswith('.log') and item.find('.') == (len(item) - 4)) and item != 'stdout.log' or item.endswith('output.xml'):
                    source = os.path.join(path, item)
                    destination = os.path.join(dest, item)
                    print('Copy Source', source)
                    print('Copy Destination', destination)
                    if not os.path.exists(dest):
                        os.makedirs(dest)
                    copyfile(source, destination)
        elif any('.pl.log' in item for item in files):  # if JT
            for item in files:
                if item.endswith('.pl.log') or item.endswith('.expect'):
                    source = os.path.join(path, item)
                    destination = os.path.join(dest, item)
                    print('Copy Source', source)
                    print('Copy Destination', destination)
                    if not os.path.exists(dest):
                        os.makedirs(dest)
                    copyfile(source, destination)


def read_csv():
    for bucket in buckets:
        # with open(os.getcwd()+'/buckets/'+bucket+'.csv', encoding='utf-8-sig') as csv_file:
        #     tmp_reader = csv.DictReader(csv_file)
        #     total = len(list(tmp_reader))

        with open(os.getcwd()+'/buckets/'+bucket+'.csv', encoding='utf-8-sig') as csv_file:
            reader = csv.DictReader(csv_file)

            # batch_size = total // 4
            # print('Total Files: ', total)
            # print('Batch Size: ', batch_size)

            # for fold in range(1, 5):
            #     files_moved = 0
            for row in reader:
                # if files_moved == batch_size:
                #     print('Total Files Moved:', files_moved)
                #     break
                log_path = row['logpath']
                tag = row['debug_tag']
                local_path = log_path.split('prod')[1]

                # SERVER PATH
                source = os.getcwd() + '/re_data/' + tag + local_path
                tmp = list(filter(lambda x: x != '', local_path.split('/')))
                exec_id = tmp[-3]
                destination = os.getcwd() + '/fre_data/' + bucket + '/' + exec_id + '/'

                # print(source,destination)
                file_copy(source, destination)
                # files_moved += 1

            print('Done')

read_csv()