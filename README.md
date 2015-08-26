# CARSKit

### Background

**Recommender systems (RS)** have been developed for decades, and many classical recommendation algorithms, such as KNN-based collaborative filtering, matrix factorization, slope one recommender, have been applied in both research and academic areas. Meanwhile, many more new algorithms were proposed and built upon those classic algorithms. Researchers in the RS domain have already built some open-source recommendation engines, such as MyMediaLite (in C#), LensKit (in Java), Mahount (in Java), and so on.

Just in the past decade, some novel RS emerged, such as **context-aware recommender systems (CARS)**, group recommender systems, social recommender systems, and so on. However, the open-source recommendation engines mentioned above usually just include classical recommendation algorithms in their library without any expansion on those new type of recommender systems. One of the reasons why is that those new type of RS typically have different formats on the data and additional information is required to be incorporated into the recomendation algorithms. Therefore, it is not that easy to have those algorithms embeded in the existing toolkit.

### Introduction

In the domain of context-aware recommender systems, many contextual recommendation algorithms have been proposed and developed in the past decade. Many of them have been widely considered as baseline algorithms in this domain, but there are still no libraries to support those algorithms for evaluation purposes. **CARSKit** is a Java-based context-aware recommendation engine, and the first one specifically for the context-aware recommendation problems. It includes almost all popular context-aware recommendation algorithms (e.g., context pre-filtering, post-filtering and contextual modeling algorithms) in the toolkit. In addition, it also provides some preprocessing tools for users to convert their data to the common format in CARS.

### Architecture and Design

![CARSKit Structure](http://students.depaul.edu/~yzheng8/images/CARSKit.png)

### Algorithms

In terms of the context-aware recommendation algorithms, CARSKit simply divides it into two categories: **Transformation Algorithms** and **Adaptation Algorithms**. The transformation algorithms try to convert the mulidimensional recommendation problem into traditional 2-dimensional problem, so that the traditional recommendation algorithms can still be used. In contrast to those algorithms based on transformation algorithms, the adaptation algorithms focus on their effect on building algorithms by adapting to the multidimensional rating space. Most of the algorithms inside belong to the contextual modeling algorithms, such as Context-aware Matrix Facatorization, Tensor Factorization, Contextual Sparse Linear Method, Factorization Machines, and so forth.

### Data Sets

A list of context-aware data sets can be found here: http://tiny.cc/contextdata 

Please consider filling out our survey on movie ratings: http://tinyurl.com/surveycars

### Open Source

The project will be hosted on Github: https://github.com/irecsys/CARSKit. But the current development and testing are performed offline, the source codes were not released on Github yet.

### Contacts

Relevant requests or questions, please send emails to [recsys.carskit AT gmail.com]. For bug reports or related questions to the CARSKit itself (e.g., implementations or optimizations), please submit the ticket to this github.

### Release Notes

2015/08/25, Pre-Release

Most algorithms (traditional recsys algorithms and context-aware recsys algorithms) have been implemented. Only a few of more changed will be made, and probably some new CARS algorithms (published in this year) will be implemented. API documents will be created, as well as a short demo. The CARSKit is almost done, and it will be released in September, 2015, estimatedly.







