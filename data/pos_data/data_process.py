#!/usr/bin/python
#-*- coding:utf8 -*-

if __name__ == '__main__':
    target_file = open('test_data','w')
    with open('wiki-en-test.pos','r') as f1:
        with open('wiki-en-test.tok','r') as f2:
            for pos,tok in zip(f1,f2):
                pos_seq = pos.strip().split()
                tok_seq = tok.strip().split()

                line_str_list = []
                for i in xrange(len(pos_seq)):
                    if tok_seq[i] and pos_seq[i]:
                        line_str_list.append(tok_seq[i]+'_'+pos_seq[i])
                    else:
                        print tok_seq[i],pos_seq[i]
                target_file.write(' '.join(line_str_list)+'\n')
