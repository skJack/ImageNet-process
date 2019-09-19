import os 
import pdb
def get_labeltxt(root = '../imagenet/ImageNet2012',file_path = 'train.txt'):
    '''
    @param root :is your imagenet rootpath
    @param file_path: is txt text
    '''
    lines_in_txt = open(os.path.join(root,file_path),'r')
    current_label = 0
    begin_row = 1
    end_row = 0
    with open('./index.txt','w') as f:
        for line in lines_in_txt:
            
            line = line.rstrip() 
            split_str = line.split()
            tmp_label = split_str[1]
            if tmp_label == str(current_label):
                end_row += 1
            else:
                message = f'{int(current_label)} {int(begin_row)} {int(end_row)}'
                current_label += 1 
                begin_row = end_row+1
                end_row += 1
                f.write(message)
                f.write('\n')
        f.close()
get_labeltxt()
                                                                                                                    