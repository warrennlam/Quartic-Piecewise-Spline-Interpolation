from google.colab import auth
auth.authenticate_user()
from scipy.misc import derivative


import gspread
from google.auth import default
creds, _ = default()

gc = gspread.authorize(creds)

worksheet = gc.open('A-Model201').sheet1

# get_all_values gives a list of rows.
rows = worksheet.get_all_values()
print(rows)

# Convert to a DataFrame and render.
import pandas as pd
pd.DataFrame.from_records(rows)

#Convert to a DataFrame
df = pd.DataFrame(rows)
df.tail(-1)
