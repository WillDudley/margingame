import cufflinks as cf
from plotly.offline import init_notebook_mode, plot, iplot

# https://plot.ly/pandas/heatmaps/
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True, theme='pearl')


def heatmap(dataframe, ipython=False):
    if ipython:
        init_notebook_mode(connected=True)
    if ipython:
        return dataframe.iplot(kind='heatmap')
    else:
        return dataframe.plot(kind='heatmap')
