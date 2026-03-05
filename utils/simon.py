import pandas as pd
import plotly.express as px
from nptdms import TdmsFile
simon = (pd.read_csv('Flow_data.csv'))
print(simon.head())
maximum = simon['WaterConsumption'].max()
fig = px.line(simon['WaterConsumption'])
fig.show()



tic = time.perf_counter()

now = datetime.now()

Chosen_Date = "2025-05-27"

# current_date= now.date()

current_date = datetime.strptime(Chosen_Date, "%Y-%m-%d")

date_range = pd.date_range(start=current_date, periods=1, freq='D')

Production_file = 'C:/Users/Labothap/Desktop/UNLEASHLabview/measurements/250527000001.tdms'

df_production = TdmsFile.read(Production_file)

data_group = df_production["Data"]

data_dict = {}

for channel in data_group.channels():
    name = channel.name.strip()  # Clean up whitespace or newline in names

    data_dict[name] = channel[:]

df_production = pd.DataFrame(data_dict)

df_production["DateTime"] = df_production["DateTime"].dt.tz_localize("UTC").dt.tz_convert("Europe/Berlin")
