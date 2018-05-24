import numpy as np

vis = None
win = None

X = []
Y = []
Y2 = []


def visdom_plot(
        total_num_steps,
        mean_reward,
        n_violation
):
    # Lazily import visdom so that people don't need to install visdom
    # if they're not actually using it
    from visdom import Visdom

    global vis
    global win
    global avg_reward

    if vis is None:
        vis = Visdom(use_incoming_socket=False)
        assert vis.check_connection()
        # Close all existing plots
        vis.close()

    X.append(total_num_steps)
    Y.append(mean_reward)
    Y2.append(n_violation)

    # The plot with the handle 'win' is updated each time this is called
    win = vis.line(
        X=np.array(X),
        Y=np.array(Y),
        name='reward mean',
        opts=dict(
            xlabel='Total time steps',
            ytickmin=0,
            width=900,
            height=500,
        ),
        win=win
    )

    vis.line(
        X=np.array(X),
        Y=np.array(Y2),
        win=win,
        name='violations',
        update='append'
    )
