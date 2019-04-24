# Incremental Linear Discriminant Analysis and its variants
Incremental Linear Discriminant Analysis is among the most common and popular research areas of Machine Learning for handling dynamic data such as face data. Supervised Learning techniques are most commonly used for classification for the advantages it offers. But when the data is dynamic, meaning the data is changing or is being added or updated from time to time, Machine Learning models have an additional overhead of entirely storing previous data as well as overhead of computational complexity for re-compiling the model. Incremental Learning on the other hand, offers a great advantage as the previous data need not be stored as well as the model can be updated instead of recompilation. Thus, Incremental LDA offers several advantages in fields with dynamically changing data such as Face Recognition, etc.

For the testing, we have used a K-Fold validation technique with several K values (2, 3, 4, 5) over popular face datasets publicly available such as AR, CACD, Yale B, FERET, ORL. These images are loaded as numpy array and are trained over the models. A variation of ILDA called Weighted ILDA has been proposed and implemented. Weighted ILDA uses a Weighted Pariwise Fischer Criterion for updating weights over classes that are not well separated. This method offers a slight advantage over ILDA as the classes with less separation are more separated offering better classification accuracy.


References
----------
Shaoning Pang, S. Ozawa and N. Kasabov, "Incremental linear discriminant analysis for classification of data streams," in IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), vol. 35, no. 5, pp. 905-914, Oct. 2005.

Yixiong Liang, Chengrong Li, Weiguo Gong, Yingjun Pan, Uncorrelated linear discriminant analysis based on weighted pairwise Fisher criterion, Pattern Recognition, Volume 40, Issue 12, 2007, Pages 3606-3615.
