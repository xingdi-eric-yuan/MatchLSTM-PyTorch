# encoding: utf-8
import zipfile
import numpy as np
import h5py


def export_data_h5(vocabulary, embedding_matrix, output='embedding.h5'):
    f = h5py.File(output, "w")
    compress_option = dict(compression="gzip", compression_opts=9, shuffle=True)
    words_flatten = '\n'.join(vocabulary)
    f.attrs['vocab_len'] = len(vocabulary)
    dt = h5py.special_dtype(vlen=str)
    _dset_vocab = f.create_dataset('words_flatten', (1, ), dtype=dt, **compress_option)
    _dset_vocab[...] = [words_flatten]
    _dset = f.create_dataset('embedding', embedding_matrix.shape, dtype=embedding_matrix.dtype, **compress_option)
    _dset[...] = embedding_matrix
    f.flush()
    f.close()


def glove_export(embedding_file):
    with zipfile.ZipFile(embedding_file) as zf:
        for name in zf.namelist():
            vocabulary = []
            embeddings = []
            with zf.open(name) as f:
                for line in f:
                    vals = line.split(' ')
                    vocabulary.append(vals[0])
                    embeddings.append([float(x) for x in vals[1:]])
            export_data_h5(vocabulary, np.array(embeddings, dtype=np.float32), output=name + ".h5")


if __name__ == '__main__':
    glove_export('glove.840B.300d.zip')
