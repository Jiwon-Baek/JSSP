"""
Package Configurations
python                    3.11.3
simpy                     4.0.1
"""

from config import *
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import numpy as np
# If you also want to get the image bytes as a variable, you can use BytesIO
from io import BytesIO

simmode = ''

# create a column with the color for each department
def color(row):
    c_dict = {'Part0':'#E64646', 'Part1':'#E69646', 'Part2':'#34D05C', 'Part3':'#34D0C3', 'Part4':'#3475D0'}
    return c_dict[row['Job'][0:5]]


def Gantt(result, num, printmode = True, writemode = False):

    df = result.iloc[0:num]

    # project start date
    proj_start = df.Start.min()

    # number of days from project start to task start
    df['start_num'] = (df.Start - proj_start)
    # number of days from project start to end of tasks
    df['end_num'] = (df.Finish - proj_start)
    # days between start and end of each task
    df['days_start_to_end'] = df.end_num - df.start_num

    df['color'] = df.apply(color, axis=1)

    fig, ax = plt.subplots(1, figsize=(16, 10))
    ax.barh(df.Machine, df.days_start_to_end, left=df.start_num, color=df.color, edgecolor='black')
    ##### LEGENDS #####
    c_dict = {'Part0': '#E64646', 'Part1': '#E69646', 'Part2': '#34D05C', 'Part3': '#34D0C3', 'Part4': '#3475D0'}
    legend_elements = [Patch(facecolor=c_dict[i], label=i) for i in c_dict]
    plt.legend(handles=legend_elements)

    ##### TICKS #####

    plt.show()

    # Save the figure as an image file
    fig.savefig(save_path + '/' + filename + '.png', format='png')

    # Create a BytesIO object
    image_bytes_io = BytesIO()

    # Save the figure to the BytesIO object
    fig.savefig(image_bytes_io, format='png')

    # Get the image bytes
    image_bytes = image_bytes_io.getvalue()

    return image_bytes

def Gantt2(result, num, printmode = True, writemode = False):
    result_plot = result.iloc[0:num]
    fig = px.timeline(result_plot, x_start="Start", x_end="Finish", y="Machine", color="Job")
    fig.update_yaxes(autorange="reversed")  # otherwise tasks are listed from the bottom up
    fig.layout.xaxis.type = 'linear'

    for d in fig.data:
        filt = result_plot['Job'] == d.name
        d.x = result_plot[filt]['Delta'].tolist()

    # JobShopObjects 파일에서는 이 한줄만으로 모든게 잘 됐지만 이제는 안됨... 왤까?
    fig.data[0].x = result_plot.Delta.tolist()
    if printmode:
        fig.show()

    fig.update_traces(width=0.7)

    # Convert the Plotly graph to a static image (PNG format in this example)
    # image_bytes = pio.write_image인 경우에는 파일이름 지정 가능
    if writemode:
        image_bytes = pio.to_image(fig, format="png")
        image_file = pio.write_image(fig, save_path+'/'+filename+'.png', format='png')


    return image_bytes
