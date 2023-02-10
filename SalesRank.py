#!/usr/bin/env python

# -*- coding: utf-8 -*-
import sys
import argparse
import numpy as np
from pyspark import SparkContext
from time import time

def swap(xxx_todo_changeme):
    """ Swap the elements of a pair tuple.
    """
    (x,y) = xxx_todo_changeme
    return (y,x)

def deleteFirstTwo (x):
    """ Deletes the first two elements"""
    del x[0:2]
    return x

def FeatureExtraction(data,N):
    """ Inpuut:
        Dataset RDD and N is the number of partitions.
        ASIN_extractor is an RDD of tuples with two element (i,m) where m is the products' ASIN and i is the index.
        Salesrank_extractor is an RDD of tuples with two element (i,m) where m is the products' sales rank and i is the index.
        Similar_extractor is an RDD of tuples with two element (i,m) where m is the groups of products connected to this ptoduct and i is the index.
        Group_extractor is an RDD of tuples with two element (i,m) where m indicates the category of this ptoduct and i is the index.
        Reviews_extractor is an RDD of tuples with two element (i,m) where m indicates the average review of this ptoduct and i is the index.
        features is an RDD of tuples with two element (a,b) where a represents the AASIN and b is a list consits of category, connection, review factor and sales rank of the product.
        ASIN_Group_Review is an RDD of tuples with two elements (a,b) where a is ASIN an b is an array of categories and review.
        ASIN_Connection is an RDD of tuples with two elements (a,b) where a is ASIN an b is connections of the node.
        ASIN_Salesrank is an RDD of tuples with two elements (a,b) where a is ASIN an b is the sales rank.
        N=number of partitions.
        output: features, ASIN_Group_Review,ASIN_Connection,ASIN_Salesrank

    """
    ASIN_extractor=data.map(lambda s: s.encode("utf-8")) \
        .filter(lambda x: 'ASIN:' in x)\
        .map(lambda x: ( x.split(' ').pop(), ) )\
        .zipWithIndex()\
        .map(swap)
    Salesrank_extractor=data.map(lambda s: s.encode("utf-8")) \
        .filter(lambda x: '  salesrank: ' in x or 'discontinued product' in x) \
        .map(lambda x: '  salesrank: 0' if 'discontinued product' in x else x) \
        .map(lambda x: ( float(x.split(' ').pop())/100000, ) )\
        .zipWithIndex()\
        .map(swap)
    Similar_extractor=data.map(lambda s: s.encode("utf-8")) \
        .filter(lambda x: '  similar: ' in x or 'discontinued product' in x)\
        .map(lambda x: 'similar: 0' if 'discontinued product' in x else x) \
        .map(lambda x: ( deleteFirstTwo(x.split('  ')), ) )\
        .zipWithIndex()\
        .map(swap)
    Group_extractor=data.map(lambda s: s.encode("utf-8")) \
        .filter(lambda x: '  group: ' in x or 'discontinued product' in x)\
        .map(lambda x: '  group: None' if 'discontinued product' in x else x) \
        .map(lambda x: (GroupToBinary(x.split(' ').pop()), ) )\
        .zipWithIndex()\
        .map(swap)
    Reviews_extractor=data.map(lambda s: s.encode("utf-8")) \
        .filter(lambda x: '  reviews: ' in x or 'discontinued product' in x)\
        .map(lambda x: 'reviews: avg rating: 0' if 'discontinued product' in x else x) \
        .map(lambda x: ( float(x.split(' ').pop()), ) )\
        .zipWithIndex()\
        .map(swap)
    features = sc.union([ASIN_extractor, Group_extractor, Similar_extractor, Reviews_extractor, Salesrank_extractor]) \
                 .reduceByKey(lambda x,y:x+y,numPartitions = N) \
                 .map (lambda key_value : (key_value[1][0],key_value[1][1:]))
    ASIN_Group_Review=features.map(lambda x_y2: ( x_y2[0], np.array( x_y2[1][0]+[x_y2[1][2]] ) ) )
    ASIN_Connection=features.map(lambda x_y3:(x_y3[0],x_y3[1][1]))
    ASIN_Salesrank=features.map(lambda x_y4:(x_y4[0],x_y4[1][3]))
    return features, ASIN_Group_Review,ASIN_Connection,ASIN_Salesrank

def GroupToBinary(group_counter):
    """ input: group counter is actually the features defined in the previos function. It is an RDD.
        output: is an RDD replacing the category type with a list."""
    if 'Book' in group_counter:
    	return [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,5.0]
    if 'CE' in group_counter:
    	return [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,5.0,0.0]
    if 'DVD' in group_counter:
    	return [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,5.0,0.0,0.0]
    if 'Games' in group_counter:
    	return [0.0,0.0,0.0,0.0,0.0,0.0,0.0,5.0,0.0,0.0,0.0]
    if 'Music' in group_counter:
    	return [0.0,0.0,0.0,0.0,0.0,0.0,5.0,0.0,0.0,0.0,0.0]
    if 'None' in group_counter:
    	return [0.0,0.0,0.0,0.0,0.0,5.0,0.0,0.0,0.0,0.0,0.0]
    if 'Product' in group_counter:
    	return [0.0,0.0,0.0,0.0,5.0,0.0,0.0,0.0,0.0,0.0,0.0]
    if 'Sports' in group_counter:
        return [0.0,0.0,0.0,5.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    if 'Software' in group_counter:
        return [0.0,0.0,5.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    if 'Toy' in group_counter:
        return [0.0,5.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    if 'Video' in group_counter:
        return [5.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]


def gradient_beta(ASIN_Salesrank, ASIN_beta, ASIN_Group_Review,ASIN_Connection, mu,lam,N):

    """ Inputs:

        ASIN_Salesrank is an RDD of tuples with two elements (a,b) where a is ASIN an b is the sales rank.
        ASIN_beta: is an RDD of tuples with two elements (a,b) where a is ASIN and b is the beta of that ASIN.
        ASIN_Group_Review is an RDD of tuples with two elements (a,b) where a is ASIN an b is an array of categories and review.
        ASIN_Connection is an RDD of tuples with two elements (a,b) where a is ASIN an b is connections of the node.
        mu: is a float number, regularization parameter.
        lam: is a float number.
        N=number of partitions.
        output: is the RDD of gradients. The equation is provided in the report.
        """
    first_term=ASIN_beta.join(ASIN_Group_Review,numPartitions=N).map(lambda x_y5: (x_y5[0],np.inner(x_y5[1][0],x_y5[1][1])) ).cache()
    second_term=ASIN_Salesrank.join(first_term,numPartitions=N).map(lambda x_y6:(x_y6[0],x_y6[1][0]-x_y6[1][1])).cache()
    First_gradient_term =second_term.join(ASIN_Group_Review,numPartitions=N).map(lambda x_y7:(x_y7[0],-2.*x_y7[1][0]*x_y7[1][1] )).cache()
    Second_gradient_term=ASIN_beta.map(lambda x_y8:(x_y8[0],2*mu*x_y8[1])).cache()
    flatedswapped = ASIN_Connection.flatMap(lambda x_y9: [(x_y9[0],i) for i in x_y9[1]]).map(swap).cache()
    Third_gradient_term = flatedswapped.join(ASIN_beta,numPartitions=N).map(lambda x_y10: (x_y10[1][0],x_y10[1][1]) ).join(ASIN_beta,numPartitions=N).mapValues(lambda x: x[1]-x[0]).reduceByKey(lambda x,y:x+y,numPartitions=N).map(lambda x_y11:(x_y11[0],4*lam*x_y11[1]) ).cache()
    return sc.union([First_gradient_term,Second_gradient_term,Third_gradient_term]).reduceByKey(lambda x,y:x+y,numPartitions=N)


def obj_function(ASIN_Salesrank, ASIN_beta, ASIN_Group_Review,ASIN_Connection, mu,lam,N):
    """ Inputs:
        ASIN_Salesrank is an RDD of tuples with two elements (a,b) where a is ASIN an b is the sales rank.
        ASIN_beta: is an RDD of tuples with two elements (a,b) where a is ASIN and b is the beta of that ASIN.
        ASIN_Group_Review is an RDD of tuples with two elements (a,b) where a is ASIN an b is an array of categories and review.
        ASIN_Connection is an RDD of tuples with two elements (a,b) where a is ASIN an b is connections of the node.
        mu: is a float number, regularization parameter.
        lam: is a float number.
        N=number of partitions.
        output: is the objective function value. The equation is provided in the report.
        """
    first_term=ASIN_beta.join(ASIN_Group_Review,numPartitions=N).map(lambda x_y12: (x_y12[0],np.inner(x_y12[1][0],x_y12[1][1])) ).cache()
    First_obj_term=ASIN_Salesrank.join(first_term,numPartitions=N).map(lambda x_y13: np.linalg.norm(x_y13[1][0]-x_y13[1][1])**2).reduce(lambda x,y:x+y)
    Second_obj_term=ASIN_beta.map( lambda x_y14: mu*(np.linalg.norm(x_y14[1])**2) ).reduce(lambda x,y:x+y)
    flatedswapped = ASIN_Connection.flatMap(lambda x_y15: [(x_y15[0],i) for i in x_y15[1]]).map(swap)
    Third_obj_term=flatedswapped.join(ASIN_beta,numPartitions=N)\
    	.map(lambda x_y16: (x_y16[1][0],x_y16[1][1]) )\
        .join(ASIN_beta,numPartitions=N)\
        .map(lambda x_y17: lam* (np.linalg.norm(x_y17[1][1]-x_y17[1][0])**2) )\
        .reduce(lambda x,y:x+y)
    return First_obj_term + Second_obj_term + Third_obj_term


def test_error(testRDD,trained_ASIN_beta,N):
    """ Input:
        testRDD is an RDD of the test set.
        trained_ASIN_beta is the RDD of tuples with two elements. first elements shows the ASIN for the train set and the second element its trained beta.
        N=number of partitions.
        Output: value of total test error.
        """
    features, ASIN_Group_Review,ASIN_Connection,ASIN_Salesrank=FeatureExtraction(testRDD,N)
    flatedswapped = ASIN_Connection.flatMap(lambda x_y18: [(x_y18[0],i) for i in x_y18[1]]).map(swap).cache()
    new_beta= flatedswapped.join(trained_ASIN_beta,numPartitions=N)\
        .map(lambda x_y19: (x_y19[1][0],x_y19[1][1]) )\
        .map(lambda x_y20:(x_y20[0],(x_y20[1],1)))\
        .reduceByKey(lambda x,y:(x[0]+y[0],x[1]+y[1]),numPartitions=N)\
        .map(lambda x_y21: (x_y21[0],1.*x_y21[1][0]/x_y21[1][1]) )
    first_term=new_beta.join(ASIN_Group_Review,numPartitions=N).map(lambda x_y22: (x_y22[0],np.inner(x_y22[1][0],x_y22[1][1])) ).cache()
    First_obj_term=ASIN_Salesrank.join(first_term,numPartitions=N)\
        .map(lambda x_y23: np.linalg.norm(x_y23[1][0]-x_y23[1][1])**2)\
        .reduce(lambda x,y:x+y)
    error = First_obj_term
    print('test error = {}'.format(error))
    return error




def train(trainRDD,mu,lam,max_iter,eps,gain,pow,N):
    """Inputs:
       trainRDD is an RDD of the train set.
       mu: is a float number, regularization parameter.
       lam: is a float number.
       N=number of partitions.
       max_iter: ma number of iterations for estimating beta.
       gain and pow are two float numbers for defining our deminish gradient decent step size.
       Output: ASIN_beta: is an RDD of tuples with two elements (a,b) where a is ASIN and b is the beta of that ASIN.
       """
    features,ASIN_Group_Review,ASIN_Connection,ASIN_Salesrank=FeatureExtraction(trainRDD,N)
    ASIN_Group_Review.cache()
    ASIN_Connection.cache()
    ASIN_Salesrank.cache()
    i=0
    max_normgrad= 1.e99
    start = time()
    ASIN_beta=ASIN_Connection.mapValues(lambda x:np.zeros(12)).cache()
    while(i<max_iter and max_normgrad>eps):
        i+=1
        gradient=gradient_beta(ASIN_Salesrank, ASIN_beta, ASIN_Group_Review,ASIN_Connection,mu,lam,N).cache()
        max_normgrad=gradient.map(lambda x_y: np.inner(x_y[1],x_y[1])).max()
        obj=obj_function(ASIN_Salesrank, ASIN_beta, ASIN_Group_Review,ASIN_Connection, mu,lam,N)
        print('{} : max_normgradient in {} iteration = {}. obj is {}'.format(time()-start,i,max_normgrad,obj))
        alpha = 1.*gain / (i**pow)
        ASIN_beta = ASIN_beta.join(gradient,numPartitions=N).map (lambda x_y1 : (x_y1[0],x_y1[1][0] - alpha*x_y1[1][1] )).cache()
    return ASIN_beta

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'SalesRank',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input',default=None, help='Address of fold folder')
    parser.add_argument('folds',default=None,type = int, help='Number of folds')
    parser.add_argument('--epsilon',default=1.e-99,type=float,help ="Desired objective accuracy")
    parser.add_argument('--gain',default=0.001,type=float,help ="gain of step size")
    parser.add_argument('--pow',default=0.2,type=float,help ="pow of step size")
    parser.add_argument('--lam',default=0.0,type=float,help ="Regularization parameter for user features")
    parser.add_argument('--mu',default=0.0,type=float,help ="Regularization parameter for item features")
    parser.add_argument('--maxiter',default=20,type=int, help='Maximum number of iterations')
    parser.add_argument('--N',default=40,type=int, help='Parallelization Level')
    parser.add_argument('--master',default = "local[40]",help = "Spark Master")
    verbosity_group = parser.add_mutually_exclusive_group(required=False)
    verbosity_group.add_argument('--verbose', dest='verbose', action='store_true')
    verbosity_group.add_argument('--silent', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)

    args = parser.parse_args()

    sc = SparkContext(args.master,'SalesRank')

    if not args.verbose :
    	sc.setLogLevel("ERROR")
    #This part perfors K-fold cross validation.

    folds = {}

    for k in range(args.folds):
        folds[k] = sc.textFile(args.input+"/fold"+str(k))



    cross_val_rmses = []

    for k in folds:
        train_folds = [folds[j] for j in folds if j is not k ]
        trainRDD = train_folds[0]
        for fold in  train_folds[1:]:
            trainRDD=trainRDD.union(fold)
        trainRDD.repartition(args.N).cache()
        testRDD = folds[k].repartition(args.N).cache()
        trained_ASIN_beta = train(trainRDD,args.mu,args.lam,args.maxiter,args.epsilon,args.gain,args.pow,args.N)
        cross_val_rmses.append(test_error(testRDD,trained_ASIN_beta,args.N))




    print("%d-fold cross validation error is: %f " % (args.folds, np.mean(cross_val_rmses)))
