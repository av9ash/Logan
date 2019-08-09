"""Get data from quincy server using the csv file,
clean the data, merge all logs in one file and
save in corresponding buckets."""

import os
import re
from collections import OrderedDict


class Preproc:
    def __init__(self):
        self.exitcode_class_map = {'HW': 'Hardware',
                        'SW': 'Software', 'TOXIC-PR': 'Software', 'TOXIC-BUILD': 'Software',
                        'SCRIPT': 'Script',
                        'TOOLS': 'Tools',
                        'FALSE-FAILURE-ANALYZED': 'Other',
                        'FALSE-FAILURE-RESOLVED': 'Other', 'JBATCH-UPDATE': 'Other',
                        'AUTO-UPDATE': 'Other', 'PARAMS-ISSUE': 'Other', 'OTHER': 'Other', 'matchUp': 'Other'}

        self.RE_LIST = [
            r'Script:\s+[^\n\r]+[\n]+',
            r'Log directory:\s+\S+[\n]+',
            r'(Jan?|Feb?|Mar?|Apr?|May|Jun?|Jul?|Aug?|Sep?|Oct?|Nov?|Dec?)\s+\d{1,2}\s+',
            r'\d{2}:\d{2}:\d{2}\s+',
            r'[\[].*?[\]]',
            r'<\/?data>',
            r'<\/?value>',
            r'<\/?valueType>',
            r'<\/?name>',
            r'<\/?valueUnit>',
            r'<\/?object-value>',
            r'<\/?object-value-type>',
            r'<\/?cli>',
            r'<[0-9]{4,5}>',
            r'<\/?DATA>'
        ]

    # Remove all extraneous tags from all of the log files and absolute path of script
    def clean_unwanted_data(self, text):
        """Performs intial cleaning of logs as per RE_LIST.
        Parameters: file path
        Returns: edited text from log
       """
        # Flags any ERROR tag in the log.
        text = re.sub(r'[\[]ERROR[\]]\s+', 'ERROR ', text)

        for i in self.RE_LIST:
            text = re.sub(i, '', text)

        return text

    def process_log(self, source):
        """get the source file from same server in destination folder.
        Parameters: source path, destination path
        Returns: string: Found, Not Found
       """
        text = None
        try:

            logs = os.listdir(source)
            print('Total Logs: ', len(logs))
            if 'stdout.log' in logs:
                logs.remove('stdout.log')

            output_xmls = []
            other_logs = []

            for log in logs:
                if log.endswith('output.xml'):
                    output_xmls.append(log)

                elif log.endswith('.log') or log.endswith('.expect'):
                    other_logs.append(log)

            text = ''
            for log in output_xmls:
                with open(os.path.join(source, log), errors='ignore') as f_log:
                    text += parse_xml(f_log.read())

            for log in other_logs:
                with open(os.path.join(source, log), errors='ignore') as f_log:
                    text += re.sub(r'[\[].*[\] ]', '', f_log.read())

            text = self.clean_unwanted_data(text)

            dkt = OrderedDict()
            for line in text.split('\n'):
                if line in dkt:
                    dkt[line] += 1
                else:
                    dkt[line] = 1

            text = '\n'.join(dkt.keys())

        except Exception as ex:
            print(ex)

        finally:
            if text is not None:
                return text
            else:
                return 'Path Not Accessible'

    def get_logs(self, source, destination):
        """get the source file from same server in destination folder.
        Parameters: source path, destination path
        Returns: string: Found, Not Found
       """
        text = self.process_log(source)
        with open(destination + '.txt', 'w') as big_file:
            big_file.writelines(text)

    def get_class(self, debug_tag):
        """Get the bucket corresponding to debug path."""

        bucket = None

        if debug_tag in self.exitcode_class_map:
            bucket = self.exitcode_class_map[debug_tag]
        else:
            match = re.search(r'^.*?(?=-)', debug_tag)
            if match and match.group():
                bucket = self.exitcode_class_map.get(match.group())

        # print(debug_tag, bucket)
        return bucket


def parse_xml(text):
    """Remove xml tags and merger output_xml and .log files.
    Parameters: file path
    Returns: unique text from log
   """
    text = re.sub(r'<[^<]+>', '', text)
    # Rid of all tags from XML file
    text = re.sub(r'[\[].*[\]]', '', text)
    text = re.sub(r'&lt', '<', text)
    text = re.sub(r'&gt', '>', text)

    return text
