from sklearn.neighbors import KNeighborsClassifier

### Collecting data
X = [[0],[1]]
Y = [1]

### Training
neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(X=X,Y=Y) # type: ignore

### Inference
print("Neigh : ",neigh.predict([[1.1]])) # type: ignore
