import re


def parse_xml(*filepath):
    text = re.sub(r'<[^<]+>', "", open(filepath[0], errors='ignore').read())
    text = re.sub(r'[\[].*[\]]', '', text) #Rid of all tags from XML file
    lines = text.split("\n")
    uniqueLines = set(lines)
    uniqueLines = list(uniqueLines) #Create list of all unique lines found in XML file
    count = 0
    for i in uniqueLines:
        if len(i) != 0:
            uniqueLines[count] = raw_string_xml(uniqueLines[count])
        count += 1
    uniqueLines = set(uniqueLines)

    if len(filepath) > 1:
        with open(filepath[1]) as fp:
            line = fp.readline()
            while line:
                davo = raw_string_log(line) #Get rid of all extraneous characteristics in string
                if davo not in uniqueLines:
                    text += line + '\n' #Append line to text to be added to .txt file if not in uniqueLines
                line = fp.readline()
    return text


def raw_string_xml(text): #Get rid of extraneous characters in lines
    text = text.lower().lstrip()
    text = text.rstrip('\n')
    text = text.rstrip('\r')
    return text


def raw_string_log(line): #Remove all tags from file
    davo = re.sub(r'[\[].*[\]]', '', line)
    davo = davo.rstrip('\n')
    davo = davo.rstrip('\r')
    davo = davo.replace('<', '&lt')
    davo = davo.replace('>', '&gt')
    davo = davo.lower().lstrip()
    return davo