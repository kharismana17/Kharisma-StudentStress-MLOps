
from sklearn.model_selection import GridSearchCV

param_grid={

"n_estimators":[100,200],

"max_depth":[5,10]
}

grid=GridSearchCV(
model,
param_grid,
cv=5
)

grid.fit(
X_train,
y_train
)

print(
grid.best_params_
)
