import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([[1980], [1985], [1990], [1995], [2000], [2005], [2010], [2015]])
y = np.array([23, 23, 24, 25, 26, 27, 29, 30])

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print(f'Koefisien: {model.coef_[0]}')
print(f'Intercept: {model.intercept_}')

plt.scatter(X, y, color='blue', label='Data Asli')
plt.plot(X, y_pred, color='red', label='Regresi Linear')
plt.xlabel('Tahun')
plt.ylabel('Umur Rata-rata')
plt.title('Regresi Linear Sederhana')
plt.legend()
plt.show()
