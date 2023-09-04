from google.colab import auth
auth.authenticate_user()
from scipy.misc import derivative
from scipy.interpolate import *
from sympy import Symbol, Function, diff
import numpy as np

import matplotlib.pyplot as plt

import gspread
from google.auth import default
creds, _ = default()

gc = gspread.authorize(creds)

worksheet = gc.open('Copy of newA-Model102').sheet1

# get_all_values gives a list of rows.
rows = worksheet.get_all_values()
#print(rows)

# Convert to a DataFrame and render.
import pandas as pd
pd.DataFrame.from_records(rows)

#Convert to a DataFrame 
df = pd.DataFrame(rows)
df.tail(-1)
DeformationTotal = df.iloc[1:,6]
DeformationTotal
DeformationTotalInt = DeformationTotal.to_numpy()
DeformationResult = DeformationTotalInt.flatten()
df.iloc[1:,6]

from sympy import Symbol, Function, diff
def liststrtofloat(y):
  templist = []
  for i in y:
    templist.append(float(i))
  return templist

def umaker(y,tdiff):
  templist = []
  pointnum = 1
  for m in range(len(y)):
    if (m==0) or (m==(len(y)-1)):
      templist.append(y[m])
    else:
      templist.append(y[m])
      templist.append(y[m])
  while len(templist)<((len(y)-1)*5):
    templist.append(0)
  return templist
def inputarray(r):
  x = Symbol("x")
  f1 = x**4
  f2 = x**3
  f3 = x**2
  f4 = x**1
  f5 = x**0
  y = []
  b = 3
  for q in range(len(r)):
    y.append(q+1)
#large group separator
  counter = 1
  temparray = []
  row = []
  count = 0
#column separator
  groupnum = 0
#group 2 mini group separator
  derivnum = 1
#group 2 starting number
  derivdnum = 1
  for i in range(len(y)-1):
#1st row
    for r in range(groupnum*5):
      row.append(0)
    row.append(y[i]**4)
    row.append(y[i]**3)
    row.append(y[i]**2)
    row.append(y[i])
    row.append(1)
    for f in range(((len(y)-2)-groupnum)*5):
      row.append(0)
    temparray.append([])
    count += 1
    for queue in range(len(row)):
      temparray[count-1].append(row[queue])
    row.clear()
    counter += 1
#2nd row same column
    for k in range((groupnum*5)):
      row.append(0)
    row.append((y[i+1])**4)
    row.append((y[i+1])**3)
    row.append((y[i+1])**2)
    row.append(y[i+1])
    row.append(1)
    for j in range(((len(y)-2)-groupnum)*5):
      row.append(0)
    temparray.append([])
    count += 1
    for i in range(len(row)):
      temparray[count-1].append(row[i])
    row.clear()
    groupnum += 1
    counter += 1
  groupnum = 0  
  #print ("Done with first group")
#############################################################################################
  for z in range(2*(len(y)-2)):
    if derivdnum == y[-2]:
      groupnum = 0
      derivdnum = 1
      derivnum = 2
    for fqw in range(groupnum*5):
      row.append(0)
    b = derivnum+1
    if (b % 2)==0:
      b += 1
    row.append(float(diff(f1, x, derivnum).evalf(subs={x: y[derivdnum]})))
    row.append(float(diff(f2, x, derivnum).evalf(subs={x: y[derivdnum]})))
    row.append(float(diff(f3, x, derivnum).evalf(subs={x: y[derivdnum]})))
    row.append(float(diff(f4, x, derivnum).evalf(subs={x: y[derivdnum]})))
    row.append(float(diff(f5, x, derivnum).evalf(subs={x: y[derivdnum]})))
    row.append(-float((diff(f1, x, derivnum).evalf(subs={x: y[derivdnum]}))))
    row.append(-float((diff(f2, x, derivnum).evalf(subs={x: y[derivdnum]}))))
    row.append(-float((diff(f3, x, derivnum).evalf(subs={x: y[derivdnum]}))))
    row.append(-float((diff(f4, x, derivnum).evalf(subs={x: y[derivdnum]}))))
    row.append(-float((diff(f5, x, derivnum).evalf(subs={x: y[derivdnum]}))))
    for m in range(((len(y)-3)-groupnum)*5):
      row.append(0)
    temparray.append([])
    count += 1
    for i in range(len(row)):
      temparray[count-1].append(row[i])
    row.clear()
    derivdnum += 1
    counter += 1
    groupnum += 1
  #print ("Done with second group")
#############################################################################################
  b = derivnum+1
  if (b % 2)==0:
    b += 1
  row.append(float(diff(f1, x, derivnum).evalf(subs={x: y[0]})))
  row.append(float(diff(f2, x, derivnum).evalf(subs={x: y[0]})))
  row.append(float(diff(f3, x, derivnum).evalf(subs={x: y[0]})))
  row.append(float(diff(f4, x, derivnum).evalf(subs={x: y[0]})))
  row.append(float(diff(f5, x, derivnum).evalf(subs={x: y[0]})))
  for l in range((len(y)-2)*5):
    row.append(0)
  temparray.append([])
  count = count + 1
  for i in range(len(row)):
    temparray[count-1].append(row[i])
  row.clear()
  counter += 1
  for p in range((len(y)-2)*5):
    row.append(0)
  row.append(float(diff(f1, x, derivnum).evalf(subs={x: y[len(y)-1]})))
  row.append(float(diff(f2, x, derivnum).evalf(subs={x: y[len(y)-1]})))
  row.append(float(diff(f3, x, derivnum).evalf(subs={x: y[len(y)-1]})))
  row.append(float(diff(f4, x, derivnum).evalf(subs={x: y[len(y)-1]})))
  row.append(float(diff(f5, x, derivnum).evalf(subs={x: y[len(y)-1]})))
  # row.append(derivative(a1c, y[len(y)-1], 1.0,derivnum,args=(),order=b))
  # row.append(derivative(b1c, y[len(y)-1], 1.0,derivnum,args=(),order=b))
  # row.append(derivative(c1c, y[len(y)-1], 1.0,derivnum,args=(),order=b))
  # row.append(derivative(d1c, y[len(y)-1], 1.0,derivnum,args=(),order=b))
  # row.append(derivative(e1c, y[len(y)-1], 1.0,derivnum,args=(),order=b))
  temparray.append([])
  count = count + 1
  for i in range(len(row)):
    temparray[count-1].append(row[i])
  row.clear()
  counter += 1
  groupnum = 0
  #print ("Done with third group")
#############################################################################################
  groupnum = 0
  derivdnum = 0
  derivnum = 3
  for s in range((len(y)-1)):
    b = derivnum+1
    if (b % 2)==0:
      b += 1
    if derivdnum == 0:
      for d in range(groupnum*5):
        row.append(0)
      row.append(float(diff(f1, x, derivnum).evalf(subs={x: y[0]})))
      row.append(float(diff(f2, x, derivnum).evalf(subs={x: y[0]})))
      row.append(float(diff(f3, x, derivnum).evalf(subs={x: y[0]})))
      row.append(float(diff(f4, x, derivnum).evalf(subs={x: y[0]})))
      row.append(float(diff(f5, x, derivnum).evalf(subs={x: y[0]})))
      for j in range(((len(y)-2)-groupnum)*5):
        row.append(0)
      groupnum += 1
    elif(groupnum == (len(y)-1)):
      break
    else:
      for d in range((groupnum)*5):
        row.append(0)
      row.append(float(diff(f1, x, derivnum).evalf(subs={x: y[derivdnum]})))
      row.append(float(diff(f2, x, derivnum).evalf(subs={x: y[derivdnum]})))
      row.append(float(diff(f3, x, derivnum).evalf(subs={x: y[derivdnum]})))
      row.append(float(diff(f4, x, derivnum).evalf(subs={x: y[derivdnum]})))
      row.append(float(diff(f5, x, derivnum).evalf(subs={x: y[derivdnum]})))
      groupnum += 1
      for j in range((((len(y)-1)-groupnum)*5)):
        row.append(0)  
    temparray.append([])
    count += 1
    for i in range(len(row)):
      temparray[count-1].append(row[i])
    row.clear()
    derivdnum += 1
    counter += 1
  #print ("Done with last group")    
  return temparray
def findcoefs(defres):
  # umaker(liststrtofloat(defres), fin,1.0)
  b = np.array(umaker(liststrtofloat(defres),1.0))
  b = b[:, np.newaxis]
  A = np.array(inputarray(defres))
  return (np.dot(np.linalg.inv(A), b))

def f6(x, values):
  y = np.empty_like(x)
  mask1 = (1 <= x) & (x <= 2)
  mask2 = (2 < x) & (x <= 3)
  mask3 = (3 < x) & (x <= 4)
  mask4 = (4 < x) & (x <= 5)
  mask5 = (5 < x) & (x <= 6)
  mask6 = (6 < x) & (x <= 7)
  mask7 = (7 < x) & (x <= 8)
  mask8 = (8 < x) & (x <= 9)
  mask9 = (9 < x) & (x <= 10)
  y[mask1] = (values[0])*(x[mask1])**4 + (values[1])*(x[mask1])**3 + (values[2])*(x[mask1])**2 + (values[3])*(x[mask1]) + values[4]
  y[mask2] = (values[5])*(x[mask2])**4 + (values[6])*(x[mask2])**3 + (values[7])*(x[mask2])**2 + (values[8])*(x[mask2]) + values[9]
  y[mask3] = (values[10])*(x[mask3])**4 + (values[11])*(x[mask3])**3 + (values[12])*(x[mask3])**2 + (values[13])*(x[mask3]) + values[14]
  y[mask4] = (values[15])*(x[mask4])**4 + (values[16])*(x[mask4])**3 + (values[17])*(x[mask4])**2 + (values[18])*(x[mask4]) + values[19]
  y[mask5] = (values[20])*(x[mask5])**4 + (values[21])*(x[mask5])**3 + (values[22])*(x[mask5])**2 + (values[23])*(x[mask5]) + values[24]
  y[mask6] = (values[25])*(x[mask6])**4 + (values[26])*(x[mask6])**3 + (values[27])*(x[mask6])**2 + (values[28])*(x[mask6]) + values[29]
  y[mask7] = (values[30])*(x[mask7])**4 + (values[31])*(x[mask7])**3 + (values[32])*(x[mask7])**2 + (values[33])*(x[mask7]) + values[34]
  y[mask8] = (values[35])*(x[mask8])**4 + (values[36])*(x[mask8])**3 + (values[37])*(x[mask8])**2 + (values[38])*(x[mask8]) + values[39]
  y[mask9] = (values[40])*(x[mask9])**4 + (values[41])*(x[mask9])**3 + (values[42])*(x[mask9])**2 + (values[43])*(x[mask9]) + values[44]
  return y

StrDefProbe=((((df.iloc[1:,6]).to_numpy()).flatten()).astype(float))

StrDefProbe2=((((df2.iloc[1:,6]).to_numpy()).flatten()))
StrDef3 = []
print (StrDefProbe2)
for i in range(len(StrDefProbe2)):
  if StrDefProbe2[i] != '':
    StrDef3.append(StrDefProbe2[i])
for i in range(len(inputarray(StrDefProbe))):
  print (inputarray(StrDefProbe)[i])


coefs = findcoefs(StrDefProbe)
x = np.linspace(1, 10, num=10, endpoint=True)
y = StrDefProbe
f = interp1d(x, y)
xnew = np.linspace(1, 10, num=50, endpoint=True)
plt.plot(x,y,'o', xnew, f6(xnew, coefs), ".-")
plt.legend(['data','quart'])
plt.show()

