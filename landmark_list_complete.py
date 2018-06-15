import sys

f = sys.argv[1]

with open(f, 'r') as l_file:
    last_output = ""
    for line in l_file:
        last_output += "<image file='dataset/helen/{0}.jpg'></image>".format(
            line.strip())
        print(
            "<image file='dataset/helen/{0}.jpg'></image>".format(line.strip()))
        anno = open('dataset/helen/annotation/{}.txt'.format(line.strip()), 'r')
        count = -1
        for al in anno:
            if count == -1:
                count = 0
                continue
            ans = [int(float(x.strip())) for x in al.split(',')]
            print(
                "<part name='{0:02}' x='{1}' y='{2}'/>".format(count, ans[0], ans[1]))
            count += 1
        print('---')
    print(last_output)
