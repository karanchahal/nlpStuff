import os
import string
import sys

folders=os.listdir("Sum")
if not os.path.exists("Peers"):
    os.mkdir("Peers")
for folder in folders:
    articles= os.listdir("Sum/"+folder)
    if not os.path.exists("Peers/" + folder):
        os.mkdir("Peers/"+folder)

    for article in articles:
        count_lines = 0;
        f1=open('Sum/'+folder+'/'+ article, 'r')
        article = article.strip('.txt')
        f2=open('Peers/' +folder+'/'+ article + '.htm', 'w')
        f2.write("<html>\n")
        f2.write("<head>\n")
        f2.write("<title>out.html</title>\n")
        f2.write("</head>\n")
        f2.write("<body bgcolor=\"white\">\n")
        lim=1
        flag = 0;
        for line in f1.readlines():
            count_lines = count_lines+1;


            if(line is None):
                break
            if (line[0]!='\n'):
                s=line
                temp=str(lim)
                f2.write("<a name=\"")
                f2.write(temp)
                f2.write("\"")
                f2.write(">")
                f2.write("[")
                f2.write(temp)
                f2.write("]")
                f2.write("</a> <a href=\"#")
                f2.write(temp)
                f2.write("\"")
                f2.write(" id=")
                f2.write(temp)
                f2.write(">")
                f2.write(s)
                pos=f2.tell()-1
                f2.seek(pos,0)
                f2.write("</a><br>\n")
                lim=lim+1

        f2.write("</a>\n")
        f2.write("</a>\n")
        f2.write("</body>\n")
        f2.write("</html>\n")
        print('done')
