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

from sklearn import mixture
import numpy as np
import math
from operator import itemgetter
 

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
        
        preamble_lines=['#continuous ' + ' '.join(vs[1:]) + '\n',
                        '#parents m_spent m_received\n',
                        '#parents i_sold i_bought\n',
                        '#regulators m_received i_bought \n']
        print len(lines)
        counts=0
        for line in lines[1:]:
            xs = line.split()
#             print xs
            try:
                vs = [float(v) for v in xs[1:]]
                if vs[1]==vs[2]==vs[3]==vs[4]==0:
                    continue
                counts +=1
                for row,x in zip(rows, xs[1:]):
                    row.append(x)
            except:
                pass
            
        print counts
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
        
def pdf(network_file, points):
    '''
    @param network_file: a bayesian network file that output from BNfinder 
    @param points: list of points, 
    each point in the form of a dict {'name':name, 'm_received':v1, 'm_spent':v2, 'i_bought':v3, 'i_sold':v4} 
    '''
    #load network
    bn = eval(open(network_file).read())
    
    nodes = [name for name in points[0].keys() if name != 'name']
    model={}    
    parents = {}
    #load model
    for node in nodes:
        # if node have no parents
        states = [()]
        model[node]=[]
        
        print bn[node]['pars']

        if len(bn[node]['pars']) > 0:
            states=[(0,), (1,)]
            
            
        for state in states:
#             print bn[node]['cpds'][state]
#             print bn[node]['floatParams']
                    
            params = eval(bn[node]['floatParams'])
#             print params[2]
                        
            gmm=mixture.GMM(n_components=2)
            
            gmm.weights_=np.array([[bn[node]['cpds'][state][0]], [bn[node]['cpds'][state][1]]])
            gmm.means_=np.array([[params[0]], [params[1]]])
            gmm.covars_= np.array([[params[2]],[params[2]]]) ** 2 
            
            model[node].append(gmm)
            
    # calculate pdf for each point
    cal_order=['m_received', 'i_bought', 'm_spent', 'i_sold']
    
    result=[]
    for point in points:
        node_state = {}
        log_likelihood=0
        
        for node in cal_order:
            gauss_idx=0
            if len(model[node]) > 1:
                parent_node = bn[node]['pars'][0]
                
                gauss_idx = node_state[parent_node]
                
            gauss = model[node][gauss_idx]
            
#             probs_gauss = gauss.predict_proba([point[node]])
            probs = gauss.score([point[node]])
            
            if probs[0] > probs[1]:
                node_state[node]=0
                log_likelihood += probs[0]
            else:
                node_state[node]=1
                log_likelihood += probs[1]
        
        result.append((point['name'], log_likelihood, point))
        
    result = sorted(result, key=itemgetter(1))
    
    num_of_outliers = int(0.01 * len(result))
    
    print num_of_outliers
    
    print result[:num_of_outliers]
    users = [str(usr[0])+'\n' for usr in result[:num_of_outliers]]
    open(basename(network_file)+'.out', 'w').writelines(users)
            
def get_data_points(data_file):
    lines = open(data_file).readlines()
    points = []
    for line in lines[1:]:
        
        xs = line.split()
        
        point = {}
        point['name']=xs[0]
        vs = [float(x) for x in xs[2:]]
        point['m_spent'], point['m_received'], point['i_sold'], point['i_bought'] = vs
        
        points.append(point)
        
    return points
        
if __name__ == '__main__':
#     prepare_data_bnfinder()
#     vars = ['i_sold', 'm_spent', 'i_bought', 'm_received']
    points = get_data_points('data/20150629.txt')
    pdf('data/20150629.cpd', points)

#     anomaly_libpgm()
    
#     anomaly_BNfinder()