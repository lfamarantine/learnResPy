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

# 1. Plotting with glyphs
# -----------------------
# glyphs: visual properties of shapes are called glyphs (circles, squares, triangles, lines,
# rectangles with properties attached to data)
from bokeh.io import output_file, show
from bokeh.plotting import figure

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













