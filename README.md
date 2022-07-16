# MLB Moneyline Predictor
Simple machine learning streamlit to predict winning probabilities of baseball games


<span style="color:blue">
    
**Inputs**  
- Last 4+ seasons of basic game stats

**Source:** 
- https://github.com/jldbc/pybaseball

**Observed outcome**
- W or L

**Models**

- 'knn'           : KNeighborsClassifier(),
- 'decision tree' : DecisionTreeClassifier(),
- 'random forest' : RandomForestClassifier(),
- 'SVM'           : SVC(),
- 'logistic regression': LogisticRegression(),
- 'Naive Bayes' : GaussianNB(),
- 'Gradient Boost' : GradientBoostingClassifier()

**Things left to do**
- Apply grid search to find optimal parameters

**Output**  
- Your team's winning probability for a specific game
    
</span>
