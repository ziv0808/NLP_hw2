

def read_file_and_preprocess(
        file_path,
        include_y = True):

    word_list = ['ROOT']
    pos_list  = ['ROOT']
    if (True == include_y):
        head_list = ['ROOT']
    else:
        head_list = []

    with open(file_path) as f:
        content = f.readlines()

    for idx, line in enumerate(content):
        broken_line = line.split('\t')
        if len(broken_line) < 2:
            # empty line - end of sentence
            word_list.extend(['ROOT'])
            pos_list.extend(['ROOT'])
            if (True == include_y):
                head_list.extend(['ROOT'])
            continue
        else:
            word_list.append(broken_line[1])
            pos_list.append(broken_line[3])
            if (True == include_y):
                head_list.append(int(broken_line[6]))

    return  word_list, pos_list, head_list
