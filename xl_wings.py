

import pandas as pd
import xlwings as xw


df = pd.read_csv('data/euromillions.csv')
df.sample(5)

wb = xw.Book() # wb = xw.Book(filename) would open an existing file
print(xw.apps)
print(wb.sheets)
ws = wb.sheets["Blatt1"]
ws.name = "EuroMillions"

# copy data to excel..
ws.range("A1").value = df
# clear content & copy data w\o index..
ws.clear_contents()
ws.range("A1").options(index=False).value = df

# api property: in essence, xlwings is just a smart wrapper around pywin32 on Windows and appscript on Mac.

# add new data..
wb.sheets.add('Frequencies')
frequencies = wb.sheets['Frequencies']


wb.sheets.add('Graphs')
graphs = wb.sheets['Graphs']
nr_freq = xw.Chart()
nr_freq.name = 'Number Frequencies'
nr_freq.set_source_data(frequencies.range('Frequencies!E1:E13'))
nr_freq.api[1].FullSeriesCollection(1).XValues = '=Frequencies!D1:D13'
nr_freq.chart_type = 'column_clustered'
nr_freq.height = 250
nr_freq.width = 750
nr_freq.api[1].SetElement(2)  # Place chart title at the top
nr_freq.api[1].ChartTitle.Text = 'Number Frequencies'
nr_freq.api[1].HasLegend = 0
nr_freq.api[1].Axes(1).TickLabelSpacing = 1
frequencies.shapes.api('Number Frequencies').Line.Visible = 0


jackpot = xw.Chart()
jackpot.top = 500
jackpot.name = 'Jackpot'
last_row = ws.range(1,1).end('down').row
jackpot.set_source_data(ws.range('Euromillions!J2:J{}'.format(last_row)))
jackpot.api[1].FullSeriesCollection(1).XValues\
= 'Euromillions!L2:L{}'.format(last_row)
jackpot.chart_type = 'line'
jackpot.height = 250
jackpot.width = 750
jackpot.api[1].SetElement(2)
jackpot.api[1].ChartTitle.Text = 'Jackpot'
jackpot.api[1].HasLegend = 0
graphs.shapes.api('Jackpot').Line.Visible = 0

wb.save('EuroMillions.xlsx')
xw.apps[0].quit()







