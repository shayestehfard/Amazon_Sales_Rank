﻿Amazon Salesrank Estimation Based on Network Lasso

Khashayar Kamran Kimia ShayestehFard

December 12, 2017

Incentive: Predicting Amazon SalesRank

.

Sales rank of an item on Amazon is a 1 to 8 digit number that indicates the popularity of an item. It captures the previous sale success of an item and how it is gonna sell in the future. Amazon calculates this number but does not disclose the algorithm. Knowing an item's sales rank would help Amazon sellers of all products answer this question: Should I buy, or should I pass? Or more accurately: Is there a demand for this product?

We have obtained a huge data set consist of Amazon Products from SNAP [1] project with about 160,000 dierent products (Books, music CDs, DVDs and VHS video tapes and etc). Each product is labeled with its unique ASIN as you can see on the Amazon website. The data set forms a graph where each node is a product and is connected to products which got co-purchased with that product. This co-purchased network is sparse and each product is usually connected to 0 to 5 products. There are categorical data and customers reviews available in the data set along with the sales rank of the products.

Problem Formulation : Network Lasso

.

The method used in this project is called Network Lasso. It focuses on optimization and clustering on

the large graphs. The formulation usually looks like this:

Consider the graph G = (V;E), where V is the vertex set and E the set of edges. Each node i represents a data point and the edges represent the connection of particular data point to each other.

X X

minimize fi( i) + gjk ( j ; k) (1)

i2V (j;k)2E

Where fi is the cost function at node i, and gjk is the cost function associated with edge (j;k). An example of simple linear regression model with squared loss function and the convex problem denition would look like this:

X X

minimize jjSalesranki   i xijj22 + jj ijj2 + wjk jj j   kjj22

T 2

i2V (j;k)2E

Note that here, unlike the regular linear regression model, each data point has its own i and the model tries to jointly choose is for the whole training set in order to minimize the objective. So one question is

that how to infer the value of j of a new node j for example in the test set? As mentioned in [2], after solving for , we can interpolate the solution to estimate the value of on a new node j, for example during

j

cross-validation on a test set. Given j, all we need is its feature and its connections within the network. With this information, we treat j like a dummy node, with fj ( j ) = 0. We solve for j just like in problem (1) except without the objective function fj :

X

minimize wjk jj j   kjj22 (2)



k2N (j)

where N(j) is the set of neighbors of node j.

can be viewed as a single parameter which is tuned to yield dierent global results. denes a trade- o for the nodes between minimizing its own objective and agreeing with its neighbors. Let's look at two extreme values of . At = 0 ,  the solution at node i, is simply a minimizer of fi. This can be computed

i

locally at each node, since when = 0 the edges of the network have no eect. At the other extreme, as

! 1, problem 1 turns into

X           minimize fi(~)

i2V

[^1]where one should nd a global identical ~ for all the nodes. One can nd the optimal value of using techniques like cross-validation.

Our approach

We are going to use a linear regression model in the network lasso setting to predict the sales rank of a product. We are going to use Apache Spark to solve this problem in a parallelize and distributed fashion using gradient descent with diminishing step size.

![](Aspose.Words.9f6d4f75-1345-4873-b6ad-c8a254a943f6.001.png)

Feature Extraction:

We extract the following RDDs from our data set:

Salesrank RDD: An RDD consisting of pairs of (ASIN, Salesrank) where we divide the original saler- ank value to 100000 to be in range of [1-100] and be in the same order of other features.

Features RDD: An RDD consisting of pairs of (ASIN, x) where feature vector x is a 1  12 vector. The rst 11 elements are dummy elements indicating the category group of the product. Based on the category of the product, one of these 11 elements is set to 5.0 and others are set to 0.0. The 12th feature is the average rating of the product.

Connection RDD: The connection information of the graph.

Objective Function and Steps of Algorithm:

We also assume all the weights on the links to be equal to 1. Our objective function looks like:

X X

minimizeF = jjSalesranki   iTxijj[^2]2 + jj ijj2 + jj j   kjj22

2

Using the gradient descent we update each i individually in each iteration as follows:

(k+1) = (k)   (k)r(k)F

i i i

The question is when do we stop the iteration process? Since in this setting we have a large RDD of s at each data point, we can not look at all the individual norm of gradients. So we use maxfjjr(k)F jj2g for

i 2

our convergence metric. We continue the iteration until maxfjjr(k)F jj2g <  or we reach to a maximum

i 2

number of iterations.

Gradients are calculated as:

X

r(k)F =  2(Salesranki   Txi)xi + 2i + 2( i   k) 2

i i

k2N (i)

The last term is multiplied by 2 since we calculate it 2 times for each i: once when looking at its own neighbors and once when it appears on its neighbors links. We use a diminishing step size: (k) = gain

kpow

Testing

After nding i on the training set, for each test data point j that has at least one neighbor in the training set, we infer its beta according to problem 2:

P

^ = k2N (j) k j jN(j)j

We use this to predict the sales rank of a product. After that we can calculate the Total Test Error (TTE):

Total Test Error = X jjSalesrank   ^ Tx jj2

i i i 2

i2Vtest

Distributed Implementation:

Looking at the problem formulation and the steps of the training algorithm, we can see that the problem can be solved in a distributed manner. Each can be trained individually and independently of the whole network and the only information it needs is the value of on its neighbors.

Parallelization:

We can see that each step of the algorithm is performed on one data point and informations from its neighbors. Each node has limited neighbors and the network connection is sparse. Dimension of each feature is 12. So as the size of problem grows the step performed on each data point does not grow. So we can break the data into a large number of machines and partitions and be done in parallel. All the steps can be done using map, reduce and join.

Validation

We use 5-fold cross validation on our data set of size 160000 to test our approach. We pick a constant regularization factor = 5 and change the value of to nd the minimum total test error(TTE). You can see the other variables (gain, pow, N, memory, maxiteration and etc) and the convergence of the algorithm below:

![](Aspose.Words.9f6d4f75-1345-4873-b6ad-c8a254a943f6.002.png)

![](Aspose.Words.9f6d4f75-1345-4873-b6ad-c8a254a943f6.003.png)

You can see the Total Test Error results for = 5 below:

254800![](Aspose.Words.9f6d4f75-1345-4873-b6ad-c8a254a943f6.004.png)

254700

254600

0 20 40 60

lamda

The minimum Total Test Result is achieved at = 10. In small values of and such as the achieved optimal point of = 10;= 5 you can use larger gain to achieve faster convergence.

![](Aspose.Words.9f6d4f75-1345-4873-b6ad-c8a254a943f6.005.png)

![](Aspose.Words.9f6d4f75-1345-4873-b6ad-c8a254a943f6.006.png)

Parallelism Performance

As explained in parallelization analysis part we expect that on large data sets, breaking the data set to a large number of partitions on dierent machines be time-wise benecial. We run our algorithm on a single core with one partition and compare its run time to when we run it on 40 core with 40 partitions. We repeat this by increasing the size of the data set to see the eect of parallelism and breaking the data into partitions. We used compute node Compute-0-087 and below you can see the result:

![](Aspose.Words.9f6d4f75-1345-4873-b6ad-c8a254a943f6.007.png)

Bibliography

1. Jure Leskovec and Andrej Krevl. SNAP Datasets: Stanford large network dataset collection, June 2014.
1. David Hallac, Jure Leskovec, and Stephen Boyd. Network lasso: Clustering and optimization in large graphs. In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD '15, pages 387{396, New York, NY, USA, 2015. ACM.
7

[^1]: i2V (j;k)2E
[^2]: 