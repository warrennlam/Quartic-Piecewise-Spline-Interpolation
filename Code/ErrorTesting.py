import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

from google.colab import auth
auth.authenticate_user()

import gspread
from google.auth import default
creds, _ = default()

gc = gspread.authorize(creds)

worksheet = gc.open('Model101-Moment Force').sheet1

# get_all_values gives a list of rows.
rows = worksheet.get_all_values()
print(rows)

# Convert to a DataFrame and render.
import pandas as pd
pd.DataFrame.from_records(rows)

#Convert to a DataFrame 
df = pd.DataFrame(rows)
df.tail(-1)


# importing the matplotlib library
import matplotlib.pyplot as plt

StrainProbe = df.iloc[1:,3]
StrainProbe
StrainProbeInt = StrainProbe.to_numpy()
result = StrainProbeInt.flatten()
## [A] Equivalent Elastic Strain (Min) [pm/pm]
from scipy.interpolate import interp1d
import numpy as np
from scipy.interpolate import lagrange
from scipy.interpolate import PchipInterpolator
x = np.linspace(1, 10, num=10, endpoint=True)
y = result.astype(np.float)
f = interp1d(x, y)
f2 = interp1d(x, y, kind='cubic')
f3 = lagrange(x, y)
f4 = PchipInterpolator(x, y)
xnew = np.linspace(1, 10, num=20, endpoint=True)
import matplotlib.pyplot as plt
plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--', xnew, f3(xnew), '.', xnew, f4(xnew), '*')
plt.legend(['data', 'linear', 'cubic', 'lagrange', 'PchipInterpolator'], loc='best')
plt.show()
print (x)



## [B] Equivalent Elastic Strain (Max) [pm/pm]
ElasticStrain = df.iloc[:, [4]]
ElasticStrainS = ElasticStrain.tail(-1)
enp = ElasticStrainS.to_numpy()
result = enp.flatten()
StrEn = result.astype(np.float)
from scipy.interpolate import interp1d
import numpy as np
from scipy.interpolate import lagrange
from scipy.interpolate import PchipInterpolator
x = np.linspace(1, 10, num=10, endpoint=True)
y = StrEn
f = interp1d(x, y)
f2 = interp1d(x, y, kind='cubic')
f3 = lagrange(x, y)
f4 = PchipInterpolator(x, y)
xnew = np.linspace(1, 10, num=20, endpoint=True)
import matplotlib.pyplot as plt
plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--', xnew, f3(xnew), '.', xnew, f4(xnew), "*")
plt.legend(['data', 'linear', 'cubic', 'lagrange', 'PchipInterpolator'], loc='best')
plt.show()

## [B] Equivalent Elastic Strain (Max) [pm/pm]
enp = ElasticStrainS.to_numpy()
result = enp.flatten()
StrEn = result.astype(np.float)
from scipy.interpolate import interp1d
import numpy as np
from scipy.interpolate import lagrange
x = np.linspace(1, 10, num=10, endpoint=True)
y = StrEn
f = interp1d(x, y)
f2 = interp1d(x, y, kind='cubic')
f3 = lagrange(x, y)
xnew = np.linspace(1, 10, num=20, endpoint=True)
import matplotlib.pyplot as plt
plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--', xnew, f3(xnew), '.')
plt.legend(['data', 'linear', 'cubic', 'lagrange'], loc='best')
plt.show()

### [C] Deformation Probe (X) [nm]
DeformProb = df.iloc[1:,5]
numpDeform = DeformProb.to_numpy()
arrDeform = numpDeform.flatten()
StrDeform = arrDeform.astype(float)
x = np.linspace(1, 10, num=10, endpoint=True)
y = StrDeform
f = interp1d(x, y)
f2 = interp1d(x, y, kind='cubic')
f3 = lagrange(x, y)
xnew = np.linspace(1, 10, num=20, endpoint=True)
import matplotlib.pyplot as plt
plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--', xnew, f3(xnew), '.')
plt.legend(['data', 'linear', 'cubic', 'lagrange'], loc='best')
plt.show()

### [D] Deformation Probe 2 (Y) [nm]

DeformProbY = df.iloc[1:,6]
numpDeformY = DeformProbY.to_numpy()
arrDeformY = numpDeformY.flatten()
StrDeformY = arrDeformY.astype(float)
x = np.linspace(1, 10, num=10, endpoint=True)
y = StrDeformY
f = interp1d(x, y)
f2 = interp1d(x, y, kind='cubic')
f3 = lagrange(x, y)
xnew = np.linspace(1, 10, num=20, endpoint=True)
import matplotlib.pyplot as plt
plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--', xnew, f3(xnew), '.')
plt.legend(['data', 'linear', 'cubic', 'lagrange'], loc='best')
plt.show()

###[F] Stress Probe (NormX) [Pa]	
StressProbe = df.iloc[1:,7]
numpStressProbe = StressProbe.to_numpy()
arrStressProbe = numpStressProbe.flatten()
StrStressProbe = arrStressProbe.astype(float)
x = np.linspace(1, 10, num=10, endpoint=True)
y = StrStressProbe
f = interp1d(x, y)
f2 = interp1d(x, y, kind='cubic')
f3 = lagrange(x, y)
xnew = np.linspace(1, 10, num=20, endpoint=True)
import matplotlib.pyplot as plt
plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--', xnew, f3(xnew), '.')
plt.legend(['data', 'linear', 'cubic', 'lagrange'], loc='best')
plt.show()

###[E] Deformation Probe 3 (Z) [nm]

DefProbe = df.iloc[1:,8]
numpDefProbe = DefProbe.to_numpy()
arrDefProbe = numpDefProbe.flatten()
StrDefProbe = arrDefProbe.astype(float)
x = np.linspace(1, 10, num=10, endpoint=True)
y = StrDefProbe
f = interp1d(x, y)
f2 = interp1d(x, y, kind='cubic')
f3 = lagrange(x, y)
xnew = np.linspace(1, 10, num=20, endpoint=True)
import matplotlib.pyplot as plt
plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--', xnew, f3(xnew), '.')
plt.legend(['data', 'linear', 'cubic', 'lagrange'], loc='best')
plt.show()
