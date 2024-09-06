- The Optimal Decision and Model trees, univariate and multivariate, can be computed running the TrainingFraework.py file.
    the program can read files in aarf format stored in the ClassificationProblems (RegressionProblems, respectively);
    the user can specify the size of validation and test set as a fraction of the whole dataset, and also input the minimum and maximum number of splits allowed;
    the user can specify whether the data should be stratified or not, set the time limit for each MILP problem, and the number of runs with different random seeds (this determines how the dataset is split);
    the user can specify whether the "Weston and Watkins" (WW) formulation for multi-class problems should be used or not;
    the user can decide to perform splits only on meta features and compute SVMs only on numeric features (only available for model trees). In this case the dictonary "meta_features" must be updated with the lists of meta and numeric features for the specific dataset
    based on the specified parameters the tree that yields the best result on the validation set (among trees with different number of splits and regularization coefficients) will be evaluated on the test set. The results are reported in the ClassificationResults ( RegressionResults, respectively) folder. 
- The "WekaMethods" file allows to run LMT (M5P, respectively) algorithm for classification (regression, respectively) problems.
- The "SklearnMethods" file allows to run RandomForest, CART, and SVM for classification and regression problems.
    
