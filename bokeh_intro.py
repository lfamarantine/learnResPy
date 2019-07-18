# Interactive data visualization with Bokeh
# -----------------------------------------
# Bokeh is an interactive data visualization library for Python (and other languages!) that targets modern web
# browsers for presentation. It can create versatile, data-driven graphics, and connect the full power of the entire
# Python data-science stack to rich, interactive visualizations.

# Features:
# - interactive visualization, controls and tools
# - versatile and high-levels graphics
# - streaming, dynamic, large data
# - for the browser, with or without a server

import numpy as np
import pandas as pd

# 1. Basic plotting with Bokeh
# ----------------------------
# glyphs: visual properties of shapes are called glyphs (circles, squares, triangles, lines,
# rectangles with properties attached to data)
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource

help(figure)
# example use..
plot = figure(plot_width=400, tools='pan,box_zoom')
plot.circle([1,2,3,4,5], [8,6,5,2,3])
output_file('circle.html')
show(plot)

# additional glyphs..
x = [1,2,3,4,5]
y = [8,6,5,2,3]
plot = figure()
plot.line(x, y, line_width=3)
plot.circle(x, y, fill_color='white', size=10)
output_file('line.html')
show(plot)

# other available glyphs:
# annulus(), annular_wedge(), wedge()
# rect(), quad(), vbar(), hbar()
# image(), image_rgba(), image_url()
# patch(), patches()
# line(), multi_line()
# circle(), oval(), ellipse()
# arc(), quadratic(), bezier()


# data formats:
# numpy arrays..
x = np.linspace(0, 10, 1000)
y = np.sin(x) + np.random.random(1000) + 0.2
plot = figure()
plot.line(x, y)
output_file('ex_bokeh.html')
show(plot)

# column data source..
# common fundamental data structure for bokeh
# maps string names to sequences of data
source = ColumnDataSource(data={
    'x': [1,2,3,4,5],
    'y': [8,6,5,2,3]
})
source.data

# create CDS directly with pandas..
from bokeh.sampledata.iris import flowers as df
df = pd.read_csv('data/bokeh_sprint.csv',sep=',')
source = ColumnDataSource(df)
p = figure(x_axis_label='Year', y_axis_label='Time')
p.circle('Year','Time',source=source,size=8,color='color')
output_file('ex_bokeh.html')
show(p)

# customizing glyphs: box-select, hover-tools, color-mapping..
# ---
# box_select tool to a figure and change the selected and non-selected circle glyph properties
# .. box-select:
p = figure(x_axis_label='Year',y_axis_label='Time',tools='box_select')
p.circle('Year', 'Time', source=source, selection_color='red', nonselection_alpha=0.1)
output_file('ex_bokeh.html')
show(p)

# .. hover-tools:
from bokeh.models import HoverTool
p.circle(x, y, size=10,
         fill_color='grey', alpha=0.1, line_color=None,
         hover_fill_color='firebrick', hover_alpha=0.5,
         hover_line_color='white')
hover = HoverTool(tooltips=None,mode='vline')
p.add_tools(hover)
output_file('ex_bokeh.html')
show(p)

# color-mapping:
from bokeh.models import CategoricalColorMapper
df = pd.read_csv('data/bokeh_auto-mpg.csv',sep=',')
source = ColumnDataSource(df)
# make a CategoricalColorMapper object..
color_mapper = CategoricalColorMapper(factors=['Europe', 'Asia', 'US'],
                                      palette=['red', 'green', 'blue'])
# add a circle glyph to the figure p..
p.circle('weight', 'mpg', source=source,
            color=dict(field='origin', transform=color_mapper),
            legend='origin')
output_file('ex_bokeh.html')
show(p)


# 2. Layouts, Interactions & Annotations
# --------------------------------------
df = pd.read_csv('data/bokeh_literacy_birth_rate.csv',sep=',')
source2 = ColumnDataSource(df)

# creating rows of plots..
# ---
from bokeh.layouts import row
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')
p1.circle('fertility', 'female literacy', source=source2)
p2 = figure(x_axis_label='population', y_axis_label='female_literacy (% population)')
p2.circle('population', 'female literacy', source=source2)
layout = row(p1,p2)
output_file('ex_bokeh2.html')
show(layout)


# advanced layouts..
# ---

# linking plots together..
# ---
# - linking axes (eg. 2 or more plots react the same when zooming in)
# - linking selections for all plots to react

# annotations & guides..
# ---
# - help relate scale info to user (axes, grids)
# - legends
# - drill down into details not visible in the plot (hover tooltips)


# 3. Building interactive apps with Bokeh
# ---------------------------------------















