import paramiko
import os
from stat import S_ISDIR
import csv


paramiko.util.log_to_file('/tmp/paramiko.log')

host = "ttqc-shell03"
port = 22
transport = paramiko.Transport((host, port))
password = "user_password"
username = "user_name"
transport.connect(username=username, password=password)
sftp = paramiko.SFTPClient.from_transport(transport)


def sftp_walk(remote_path):
    path = remote_path
    files = []
    folders = []
    for f in sftp.listdir_attr(remote_path):
        if S_ISDIR(f.st_mode):
            folders.append(f.filename)
        else:
            files.append(f.filename)
    if files:
        yield path, files
    for folder in folders:
        new_path = os.path.join(remote_path,folder)
        for x in sftp_walk(new_path):
            yield x


def download_file(local_path, remote_path):
    # local_path = '/Users/patila/Desktop/Data'+'/2019/May/SUB_3293283/67753094/attempt_2/script-exec/'
    # remote_path = '/volume/testtech/fusion-riad/results/fusion/prod/2019/May/SUB_3293283/67753094/attempt_2/script-exec/'

    try:
        for path, files in sftp_walk(remote_path):
            for item in files:
                sftp.get(os.path.join(os.path.join(path, item)), local_path+item)
        return 'Found'
    except IOError as e:
        print('File Not Found '+remote_path+' '+str(e))
        return 'Not Found'


def read_csv():

    with open('dr_dwnld_data.csv', encoding='utf-8-sig') as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            log_path = row['logpath']
            print(log_path)
            if 'prod' in log_path:
                tag = row['debug_tag']
                local_path = log_path.split('prod')[1]
                # local_path = '/Users/'+username+'/Desktop/Data/'+tag+local_path

                # SERVER PATH
                local_path = os.getcwd()+'/re_data/'+tag+local_path

                if not os.path.exists(local_path):
                    os.makedirs(local_path)
                    row['download_status'] = download_file(local_path, log_path)

                with open('downloaded_dr_data.csv', 'a') as dwn_file:
                    cols = ['submission_id', 'test_exec_id', 'logpath', 'dpath', 'prs', 'debug_comment', 'debug_tag',
                            'result_id',
                            'failure_type', 'tc_fail', 'reg_exitcode', 'debug_updated', 'download_status', '', '']
                    t_writer = csv.DictWriter(dwn_file, fieldnames=cols)
                    t_writer.writerow(row)

read_csv()
print('Done')