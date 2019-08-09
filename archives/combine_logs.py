import shutil
import os
from data_ingest.parse_toby_xml import parse_xml


def combine_logs(source_path, destination_path=''):
    print('Combining..')

    for log_path, _, log_names in os.walk(source_path):
        destination_path = '/c/jenie/exec_env/combined_logs/'
        # for log_name in log_names:
        #     print(log_path, log_name)cd fil

        folders = log_path.split('/')

        if len(folders) >= 7:
            destination_path = os.path.join(destination_path, folders[-2])
            destination_path = os.path.join(destination_path, folders[-1]+'.txt')

            with open(destination_path, 'wb') as big_file:
                for log_name in log_names:
                    in_file = os.path.join(log_path, log_name)

                    # Removes some tags here while combining the .log and .xml
                    if '.xml' in log_name:
                        # print(log_path, log_name, len(log_names))
                        expect_log = log_name.replace('_output.xml', '.log')
                        if expect_log in log_names:
                            content = parse_xml(in_file, os.path.join(log_path, expect_log))
                            log_names.remove(log_name)
                            log_names.remove(expect_log)
                        else:
                            content = parse_xml(in_file)

                        big_file.write(content.encode('utf-8'))
                    #   Find out a way to merge this xml to the big_file file.
                    else:
                        if log_name.replace('.log', '_output.xml') not in log_names:
                            with open(in_file, 'rb') as log:
                                shutil.copyfileobj(log, big_file)

            print(destination_path)

        elif len(folders) >= 6:
            if not os.path.exists(destination_path+folders[-1]):
                destination_path = os.path.join(destination_path, folders[-1])
                # print(destination_path)
                os.mkdir(destination_path)
                print(destination_path)


source = '/Users/patila/PycharmProjects/Jenie/fre_data/Hardware/67769987'
combine_logs(source)
