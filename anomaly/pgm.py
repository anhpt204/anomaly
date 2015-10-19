'''
Created on Oct 19, 2015

@author: nhatuan
'''
from os.path import join, basename
import glob

from libpgm.nodedata import NodeData
from libpgm.graphskeleton import GraphSkeleton
from libpgm.lgbayesiannetwork import LGBayesianNetwork
from libpgm.pgmlearner import PGMLearner

from BNfinder.BDE import BDE
from BNfinder.data import dataset
 

data='data/20150531.txt'

def read_data_libpgm(file_path):
    
    lines = open(file_path, 'r').readlines()
    data=[]

    keys= lines[0].strip().split()[1:]
    print len(lines)
#     print keys
    for line in lines[1:]:
        vs = [float(v) for v in line.strip().split()[1:]]
        sample={}
        if vs[1]==vs[2]==vs[3]==vs[4]==0:
            continue
        
        for k,v in zip(keys, vs):
            sample[k]=v
        data.append(sample)
    print len(data)
    return data

def prepare_data_bnfinder():
    files = glob.glob(join('data', '*.txt'))
    for file in files:
        
        file_name=basename(file)
        
        lines = open(file).readlines()
        vs = lines[0].split()
        rows=[[v] for v in vs[1:]]
        
        preamble_lines=['#continuous ' + ' '.join(vs[1:]) + '\n',]
        
        for line in lines[1:100]:
            xs = line.split()
            for row,x in zip(rows, xs[1:]):
                row.append(x)
                
                
        # write to file
        f = open(join('data', file_name+'.in'), 'w')
        
        lines=[' '.join(row) for row in rows]
        lines=[line+'\n' for line in lines]
        
        f.writelines(preamble_lines+ lines)
        
def anomaly_BNfinder():
    score = eval("BDE")(data_factor=1.0, chi_alpha=0.9999, sloops=False)
#     print score
    files = glob.glob(join('data', '*.in'))
    for file in files[0:1]:
        print file
        dataset_name=basename(file)
        d = dataset(dataset_name).fromNewFile(open(file))
        score2, g, subpars = d.learn(score=score, data_factor=1.0)
        
        d.write_bif(g, dataset_name+".bif")
        d.write_cpd(g, open(dataset_name+"_cpd.txt", "w"))
        
        return score2, g, subpars

def anomaly_libpgm():
    files = glob.glob(join('data', '*.txt'))
    for file in files[0:1]:
        print file
        data=read_data_libpgm(file)
        learner = PGMLearner()
        
        result=learner.lg_estimatebn(data, indegree=3)
        
        print result.E
        
            
if __name__ == '__main__':
#     prepare_data_bnfinder()

#     anomaly_libpgm()
    
    anomaly_BNfinder()