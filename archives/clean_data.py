import os
import re
import sys

re_list = [
    r'(Jan?|Feb?|Mar?|Apr?|May|Jun?|Jul?|Aug?|Sep?|Oct?|Nov?|Dec?)\s+\d{1,2}\s+',
    r'\d{2}:\d{2}:\d{2}\s+',
    r'[\[].*?[\]]',
    '<\/?data>',
    '<\/?value>',
    '<\/?valueType>',
    '<\/?name>',
    '<\/?valueUnit>',
    '<\/?object-value>',
    '<\/?object-value-type>',
    '<\/?cli>',
    '<[0-9]{4,5}>',
    '<\/?DATA>'
]


stop_texts = ['add license complete no errors', 'bool', 'collects the shmlogs if a testcase fails',
                 'connectivity already established skipping procedure', 'connect to devices',
                 'core check enable', 'core is not found on the device', 'executing command timeout',
                 'exceptions py', 'fails if', 'fv core check enable', 'last_verify_result', 'is master',
                 'ls ltd', 'ls usr', 'no such file or directory', 'parse the license file', 'run keyword',
                 'runs the given keyword', 'runs the specified keyword', 'runs the keyword', 'scanning for ',
                 'setting default', 'show system core dumps', 'successfully disconnected', 'successfully invoked',
                 'traceoptions', 'trace set', 'trace ', 'validating framework variable fv core check',
                 'volume regressions toby test suites', 'with args', 'var log', 'var crash cores', 'var tmp core',
                 'var tmp cores', 'var tmp corefiles core', 'var tmp pics cores', 'invocation of juniper ixia',
                 'invocation of ixia', 'invoking juniper ixia', 'invoking ixia', 'ixia connection status', 'arg',
                 'args', 'arguments', 'auth vlans false', 'bit error seconds bit error seconds', 'check any cores',
                 'check for core', 'check the rg', 'checks spirent statistics will fail if', 'ddos_protocol_violation',
                 'debug_collector false', 'debugcollection false', 'decode the jflow dump', 'decode_dump',
                 'decode_jflow_dump',
                 'description', 'duplicate keys will become an error in future releases', 'error check', ' error none ',
                 'errored blocks seconds errored blocks seconds', 'failover',
                 'fails the test with the given message and optionally alters its tags', 'fails unless',
                 'file usr local lib python', 'framework_variables', 'kw', 'kwargs', 'no core dumps found',
                 'no corefiles present on the host', 'no more', 'ordereddict', 'returns variable',
                 'rm rf', 'set chassis', 'set groups global', 'set routing instances', 'set security alarms',
                 'set security gprs', 'set services ssl', 'set system ports', 'set system syslog', 'show chassis fpc',
                 'show cmerror', 'show long', 'show pfe statistics', 'show security ipsec', 'show services accounting',
                 'show system errors', 'skip false', 'spirent invoke', 'strict_check false', 'this api',
                 'this is executed', 'this is the api', 'this keyword', 'this method', 'tmp dump', 'trying exec',
                 'user_variables',
                 'usr bin', 'usr local', 'usr sbin', 'uv log', 'vc system false', 'vmxstatus false'
                 ]

intresting_vocab = ['abort', 'core', 'could not', 'connect_fail', 'connect_lost', 'command_timeout', 'connection_closed',
                'die', 'disconnect', 'dump', 'device_port_unreachable', 'device_unreachable', 'does not exist',
                'error', 'exception',
                'fail', 'failed', 'failure', 'false',
                'impossible', 'invalid', 'link_fail', 'lost', 'mandatory', 'missing',
                'no longer available', 'not connected', 'not listening', 'not present', 'not set', 'not supported',
                'not as expected',
                'spirent_connect_error', 'spirent_license_error',
                'tobyexception',
                'unable', 'unavailable', 'unreachable','unexpectedly','no such',]


def trimLogs(filepath): #Remove all extraneous tags from all of the log files and absolute path of script
    scriptmarker = 'script-exec/' #Remove absolute path found in beginning of script log.
    text = open(filepath, errors='replace').read();
    text = re.sub('[\[]ERROR[\]]\s+', 'ERROR ', text)  # Flags any ERROR tag in the log.
    for i in re_list:
        text = re.sub(i, '', text);
    newLine = text.find(scriptmarker);
    text = text[newLine + len(scriptmarker):];
    return text


def clean_logs(data_path):
    for log_path, _, log_names in os.walk(data_path):
        print(log_path)
        for log_name in log_names:
            # print(log_path, log_name)
            input_log = os.path.join(log_path, log_name)
            out_path = log_path.replace('f2m_et', 'clean_data')

            if not os.path.exists(out_path):
                os.makedirs(out_path)

            output_log = os.path.join(out_path, log_name)

            print(output_log)
            lines_seen = set()  # holds lines already seen
            outfile = open(output_log, "w", errors='ignore')
            text = trimLogs(input_log).split('\n')

            for line in text:
                # Removes duplicate lines
                if line not in lines_seen and len(line) < 400:
                    if any(vocab in line.lower() for vocab in intresting_vocab) and \
                            not any(text in line.lower() for text in stop_texts):
                        outfile.write(line + '\n')
                        lines_seen.add(line)
            outfile.close()


clean_logs(os.getcwd()+'/f2m_et/set_1')
print('Done')


# def main():
#
#     if len(sys.argv) > 1 and len(sys.argv) == 3:
#         src_path = sys.argv[1]
#         target_dir = sys.argv[2]
#         clean_logs(os.getcwd()+src_path, target_dir)
#         print('Done')
#     else:
#         print('Takes 2 arguments: source_path & destination_folder_name.')
#
#
# if __name__ == '__main__':
#     main()