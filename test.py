from data_loader.reader import mulerx_tsvx_reader


if __name__ == '__main__':
    dir_name = 'datasets/mulerx/subevent-es-20/dev/'
    file_name = 'aviation_accidents-week4-fiona-4156115_chunk_5.ann.tsvx'
    my_dict = mulerx_tsvx_reader(dir_name, file_name, model='mBERT')
    print(my_dict)

