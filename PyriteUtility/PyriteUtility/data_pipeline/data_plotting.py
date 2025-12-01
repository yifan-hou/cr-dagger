import numpy as np
import sys
import os


from plotly.offline import init_notebook_mode, iplot
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px

pio.templates.default = "plotly_dark"
pio.renderers.default = "browser"
# pio.renderers.default = "vscode"


from PyriteUtility.data_pipeline.indexing import (
    get_sample_ids,
    get_samples,
    get_dense_query_points_in_horizon,
)

# fmt: off

def plot_horizon(obs_time, action_time, obs_pos, obs_wrench, action, marker, fig = None):
    if fig is None:
        fig = make_subplots(
            rows=6, cols=3,
            shared_xaxes='all', subplot_titles=('Px', 'Fx', 'Ax',
                                               'Py', 'Fy', 'Ay',
                                               'Pz', 'Fz', 'Az',
                                               'R',  'Tx', 'Ar',
                                               'P',  'Ty', 'Ap',
                                               'Y',  'Tz', 'Ay'))

    fig.add_trace(go.Scatter(x=obs_time, y=obs_pos[:,0], name='obs_pos0', mode='lines+markers', marker=marker),row=1, col=1)
    fig.add_trace(go.Scatter(x=obs_time, y=obs_pos[:,1], name='obs_pos1', mode='lines+markers', marker=marker),row=2, col=1)
    fig.add_trace(go.Scatter(x=obs_time, y=obs_pos[:,2], name='obs_pos2', mode='lines+markers', marker=marker),row=3, col=1)
    
    fig.add_trace(go.Scatter(x=obs_time, y=obs_wrench[:,0], name='obs_f0', mode='lines+markers', marker=marker),row=1, col=2)
    fig.add_trace(go.Scatter(x=obs_time, y=obs_wrench[:,1], name='obs_f1', mode='lines+markers', marker=marker),row=2, col=2)
    fig.add_trace(go.Scatter(x=obs_time, y=obs_wrench[:,2], name='obs_f2', mode='lines+markers', marker=marker),row=3, col=2)
    fig.add_trace(go.Scatter(x=obs_time, y=obs_wrench[:,3], name='obs_f3', mode='lines+markers', marker=marker),row=4, col=2)
    fig.add_trace(go.Scatter(x=obs_time, y=obs_wrench[:,4], name='obs_f4', mode='lines+markers', marker=marker),row=5, col=2)
    fig.add_trace(go.Scatter(x=obs_time, y=obs_wrench[:,5], name='obs_f5', mode='lines+markers', marker=marker),row=6, col=2)
    
    fig.add_trace(go.Scatter(x=action_time, y=action[:,0], name='action_pos0', mode='lines+markers', marker=marker),row=1, col=3)
    fig.add_trace(go.Scatter(x=action_time, y=action[:,1], name='action_pos1', mode='lines+markers', marker=marker),row=2, col=3)
    fig.add_trace(go.Scatter(x=action_time, y=action[:,2], name='action_pos2', mode='lines+markers', marker=marker),row=3, col=3)

    fig.update_layout(height=1800, width=1800, title_text="Horizon Plot")
    fig.update_layout(hovermode="x unified")

    return fig


def plot_ts_action(sparse_action_time, sparse_action, dense_action_time = None, dense_action = None, fig = None, title = 'TS Action Plot'):
    # print('action shapes: ', sparse_action.shape, dense_action.shape)
    # print('time shapes: ', sparse_action_time.shape, dense_action_time.shape)
    if fig is None:
        fig = make_subplots(
            rows=9, cols=1,
            shared_xaxes='all',
            subplot_titles=('Ax', 'Ay', 'Az', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6')
        )
    marker=dict(
        size=3,
        line=dict(
            width=1
        ),
        opacity=0.5,
    )
    fig.add_trace(go.Scatter(x=sparse_action_time, y=sparse_action[:,0], name='action_pos0', mode='lines+markers', marker=marker),row=1, col=1)
    fig.add_trace(go.Scatter(x=sparse_action_time, y=sparse_action[:,1], name='action_pos1', mode='lines+markers', marker=marker),row=2, col=1)
    fig.add_trace(go.Scatter(x=sparse_action_time, y=sparse_action[:,2], name='action_pos2', mode='lines+markers', marker=marker),row=3, col=1)
    fig.add_trace(go.Scatter(x=sparse_action_time, y=sparse_action[:,3], name='action_pos3', mode='lines+markers', marker=marker),row=4, col=1)
    fig.add_trace(go.Scatter(x=sparse_action_time, y=sparse_action[:,4], name='action_pos4', mode='lines+markers', marker=marker),row=5, col=1)
    fig.add_trace(go.Scatter(x=sparse_action_time, y=sparse_action[:,5], name='action_pos5', mode='lines+markers', marker=marker),row=6, col=1)
    fig.add_trace(go.Scatter(x=sparse_action_time, y=sparse_action[:,6], name='action_pos6', mode='lines+markers', marker=marker),row=7, col=1)
    fig.add_trace(go.Scatter(x=sparse_action_time, y=sparse_action[:,7], name='action_pos7', mode='lines+markers', marker=marker),row=8, col=1)
    fig.add_trace(go.Scatter(x=sparse_action_time, y=sparse_action[:,8], name='action_pos8', mode='lines+markers', marker=marker),row=9, col=1)

    if dense_action_time is None or dense_action is None:
        fig.update_layout(height=1200, width=1200, title_text=title)
        fig.update_layout(hovermode="x unified")
        fig.show()
        return fig

    marker=dict(
        size=15,
        line=dict(
            width=2
        ),
        opacity=0.5,
    )
    fig.add_trace(go.Scatter(x=dense_action_time, y=dense_action[:,0], name='action_pos0', mode='lines+markers', marker=marker),row=1, col=1)
    fig.add_trace(go.Scatter(x=dense_action_time, y=dense_action[:,1], name='action_pos1', mode='lines+markers', marker=marker),row=2, col=1)
    fig.add_trace(go.Scatter(x=dense_action_time, y=dense_action[:,2], name='action_pos2', mode='lines+markers', marker=marker),row=3, col=1)
    fig.add_trace(go.Scatter(x=dense_action_time, y=dense_action[:,3], name='action_pos3', mode='lines+markers', marker=marker),row=4, col=1)
    fig.add_trace(go.Scatter(x=dense_action_time, y=dense_action[:,4], name='action_pos4', mode='lines+markers', marker=marker),row=5, col=1)
    fig.add_trace(go.Scatter(x=dense_action_time, y=dense_action[:,5], name='action_pos5', mode='lines+markers', marker=marker),row=6, col=1)
    fig.add_trace(go.Scatter(x=dense_action_time, y=dense_action[:,6], name='action_pos6', mode='lines+markers', marker=marker),row=7, col=1)
    fig.add_trace(go.Scatter(x=dense_action_time, y=dense_action[:,7], name='action_pos7', mode='lines+markers', marker=marker),row=8, col=1)
    fig.add_trace(go.Scatter(x=dense_action_time, y=dense_action[:,8], name='action_pos8', mode='lines+markers', marker=marker),row=9, col=1)

    fig.update_layout(height=1200, width=1200, title_text=title)
    fig.update_layout(hovermode="x unified")

    fig.show()

    return fig


def plot_js_action(sparse_action_time, sparse_action, dense_action_time, dense_action, fig = None, title = 'JS Action Plot'):
    print('action shapes: ', sparse_action.shape, dense_action.shape)
    print('time shapes: ', sparse_action_time.shape, dense_action_time.shape)
    if fig is None:
        fig = make_subplots(
            rows=6, cols=1,
            shared_xaxes='all',
            subplot_titles=('j1', 'j2', 'j3', 'j4', 'j5', 'j6')
        )
    marker=dict(
        size=3,
        line=dict(
            width=1
        ),
        opacity=0.5,
    )
    fig.add_trace(go.Scatter(x=sparse_action_time, y=sparse_action[:,0], name='j0', mode='lines+markers', marker=marker),row=1, col=1)
    fig.add_trace(go.Scatter(x=sparse_action_time, y=sparse_action[:,1], name='j1', mode='lines+markers', marker=marker),row=2, col=1)
    fig.add_trace(go.Scatter(x=sparse_action_time, y=sparse_action[:,2], name='j2', mode='lines+markers', marker=marker),row=3, col=1)
    fig.add_trace(go.Scatter(x=sparse_action_time, y=sparse_action[:,3], name='j3', mode='lines+markers', marker=marker),row=4, col=1)
    fig.add_trace(go.Scatter(x=sparse_action_time, y=sparse_action[:,4], name='j4', mode='lines+markers', marker=marker),row=5, col=1)
    fig.add_trace(go.Scatter(x=sparse_action_time, y=sparse_action[:,5], name='j5', mode='lines+markers', marker=marker),row=6, col=1)

    marker=dict(
        size=15,
        line=dict(
            width=2
        ),
        opacity=0.5,
    )
    fig.add_trace(go.Scatter(x=dense_action_time, y=dense_action[:,0], name='j0', mode='lines+markers', marker=marker),row=1, col=1)
    fig.add_trace(go.Scatter(x=dense_action_time, y=dense_action[:,1], name='j1', mode='lines+markers', marker=marker),row=2, col=1)
    fig.add_trace(go.Scatter(x=dense_action_time, y=dense_action[:,2], name='j2', mode='lines+markers', marker=marker),row=3, col=1)
    fig.add_trace(go.Scatter(x=dense_action_time, y=dense_action[:,3], name='j3', mode='lines+markers', marker=marker),row=4, col=1)
    fig.add_trace(go.Scatter(x=dense_action_time, y=dense_action[:,4], name='j4', mode='lines+markers', marker=marker),row=5, col=1)
    fig.add_trace(go.Scatter(x=dense_action_time, y=dense_action[:,5], name='j5', mode='lines+markers', marker=marker),row=6, col=1)

    fig.update_layout(height=1200, width=1200, title_text=title)
    fig.update_layout(hovermode="x unified")

    fig.show()

    return fig


def plot_sample(total_obs, total_action, total_time,
                sparse_obs,
                dense_obs,
                sparse_action,
                dense_action, # (H, T, 9)
                sparse_obs_time, sparse_action_time,
                dense_obs_queries, dense_obs_horizon, dense_obs_down_sample_steps,
                dense_action_queries, dense_action_horizon, dense_action_down_sample_steps):
    sparse_obs_pos = sparse_obs['robot0_eef_pos']
    sparse_obs_wrench = sparse_obs['robot0_eef_wrench']
    dense_obs_pos = dense_obs['robot0_eef_pos']
    dense_obs_wrench = dense_obs['robot0_eef_wrench']

    dense_obs_sample_ids = get_sample_ids(dense_obs_queries, dense_obs_horizon, dense_obs_down_sample_steps, backwards=True, closed=False)
    dense_obs_sample_ids = np.maximum(dense_obs_sample_ids, 0)
    
    # (H, T)
    dense_action_sample_ids = get_sample_ids(dense_action_queries, dense_action_horizon, dense_action_down_sample_steps, backwards=False, closed=True)

    # plot total
    marker=dict(
        size=1,
        line=dict(
            width=1
        ),
        opacity=0.5,
    )
    fig = plot_horizon(total_time, total_time, total_obs['robot0_eef_pos'], total_obs['robot0_eef_wrench'], total_action, marker=marker)

    # plot sparse
    marker=dict(
        size=5,
        line=dict(
            color='LightSkyBlue',
            width=2
        ),
        opacity=0.5,
    )
    plot_horizon(sparse_obs_time, sparse_action_time, sparse_obs_pos, sparse_obs_wrench, sparse_action, fig=fig, marker=marker)

    # plot dense
    H, T, D = dense_action.shape
    marker=dict(
        size=15,
        line=dict(
            color='MediumPurple',
            width=2
        ),
        opacity=0.5,
    )
    fig_len_0 = len(fig.data)
    for h in range(H):
        fig = plot_horizon(dense_obs_sample_ids[h], dense_action_sample_ids[h], dense_obs_pos[h], dense_obs_wrench[h], dense_action[h], fig=fig, marker=marker)
        fig.show()
        input()
        fig.data = fig.data[:fig_len_0]

    # fig.write_html('output.html')

# fmt: on
