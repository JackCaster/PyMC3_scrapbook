#%%
import numpy as np
import pandas as pd
import patsy

#%% Create dummy data
size = 15
data = pd.DataFrame({
    'x1':np.random.uniform(size=size),
    'x2':np.random.uniform(size=size),
    'x3':np.random.choice(['A', 'B'], size=size)})

print(data)

#%%
# --- Design matrix X for fixed effect
X = patsy.dmatrix('1 + x1', data=data)
X = np.asarray(X)
print(f'Design matrix X:\n{X}')

#%%
# --- Design matrix Z for random effects
# First, we need to define the grouping variable matrix. That is, the matrix 
# that links each observation to the right level of the grouping variable
grp = patsy.dmatrix('0 + x3', data=data)
grp = np.asarray(grp)
# print(f'Grouping matrix:\n{grp}')

# Second, we need the matrix of the random effects for the predictors. 
# Let's assume that we want both intercept and slope to vary across levels
# of the grouping variable: (1 + x1 | x3)
rnd = patsy.dmatrix('1 + x1', data=data) # in this case it is the same as X
# print(f'Random predictors matrix:\n{rnd}')

# Finally, stacking the columns of the element-wise multiplication between 
# the grouping matrix with each column of the random predictors matrix, 
# yields the design matrix Z 
Z = np.column_stack(([np.multiply(grp, col[:, None]) for col in rnd.T]))
print(f'Design matrix Z:\n{Z}')

#%%#
# An alternative to build the design matrix Z is to condition the fixed effects
# on the grouping variable (removing the intercept term) 
np.array_equal(Z, np.asarray(patsy.dmatrix('0 + x3 + x1:x3', data=data)))
