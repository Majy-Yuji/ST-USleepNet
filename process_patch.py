import numpy as np
import scipy.io as scio
from os import path
from scipy import signal
from tqdm import tqdm

def read_psg(path_Extracted, sub_id, channels, resample = 3000):
    psg = scio.loadmat(path.join(path_Extracted, 'subject%d.mat' % (sub_id)))
    psg_use = []
    for c in channels:
        psg_use.append(
            np.expand_dims(signal.resample(psg[c], resample, axis=-1), 1))
    psg_use = np.concatenate(psg_use, axis=1)
    return psg_use

def read_label(path_RawData, sub_id, ignore = 30):
    label = []
    with open(path.join(path_RawData, '%d/%d_1.txt' % (sub_id, sub_id))) as f:
        s = f.readline()
        while True:
            a = s.replace('\n', '')
            label.append(int(a))
            s = f.readline()
            if s == '' or s == '\n':
                break
    return np.array(label[:-ignore])


if __name__ == '__main__':
    path_Extracted = './data/ISRUC_S3/ExtractedChannels/'
    path_RawData = './data/ISRUC_S3/RawData/'
    path_output = './data/ISRUC_S3/'
    channels = ['F3_A2', 'C3_A2', 'O1_A2', 'F4_A1', 'C4_A1', 'O2_A1',
                'ROC_A1', 'LOC_A2', 'X1', 'X2']

    datas = []
    labels = []
    for sub in tqdm(range(1, 11)):
        label = read_label(path_RawData, sub)
        psg = read_psg(path_Extracted, sub, channels)
        assert len(label) == len(psg)

        # in ISRUC, 0-Wake, 1-N1, 2-N2, 3-N3, 5-REM
        label[label == 5] = 4  # make 4 correspond to REM
        patch_num = 5
        segment = int(3000 / patch_num)
        for i in range(10):
            labels.append(label[i])
            psg_tmp = []
            for j in range(patch_num):
                for row in psg[i]:
                    psg_tmp.append(row[j*segment : (j+1)*segment])
            datas.append(psg_tmp)

    np.savez(path.join(path_output, 'ISRUC_S3.npz'),
             datas = datas,
             labels = labels
             )
    print('Saved to', path.join(path_output, 'ISRUC_S3.npz'))