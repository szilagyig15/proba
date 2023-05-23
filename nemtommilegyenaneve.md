

GOAL: predict risk, optimal decay factor

Steps
- data, daily return
- feature creation (lagged squared return)
- ewma weight implementation
- ML model
- hyperparameter tuning with cross validation (optimal decay factor)
- r^2(t)=Szumma(i=1 to N)(wi(lambda)*r^2(t-i))