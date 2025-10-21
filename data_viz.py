import plotly.express as px

def create_boxplot(series, title):
    fig = px.box(
        y=series,
        points="all",
        title=title,)

    fig.update_layout(
        yaxis_title=title,)
    fig.show()
