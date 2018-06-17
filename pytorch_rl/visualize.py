import numpy as np

vis = None
win = None
win2 = None

X = []
Y = []
Y2 = []
Y3 = []
Y4 = []

X_followed = []

last_violations = 0
followed_avg = 0


def visdom_plot(
        total_num_steps,
        mean_reward,
        n_violation,
        goal_reached,
        average_steps
):
    # Lazily import visdom so that people don't need to install visdom
    # if they're not actually using it
    from visdom import Visdom

    global vis
    global win
    global win2
    global last_violations
    global followed_avg

    if vis is None:
        vis = Visdom(use_incoming_socket=False, server="http://129.192.68.229")
        assert vis.check_connection()
        # Close all existing plots
        vis.close()

    X.append(total_num_steps)
    Y.append(mean_reward)
    Y2.append(n_violation)
    Y3.append(goal_reached)
    Y4.append(average_steps)

    # The plot with the handle 'win' is updated each time this is called
    win = vis.line(
        X=np.array(X),
        Y=np.array(Y),
        name='reward mean',
        opts=dict(
            title='Random Scenario with GOAP',
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
    ),
    vis.line(
        X=np.array(X),
        Y=np.array(Y3),
        name='goal',
        update='append',
        win=win
    ),
    vis.line(
        X=np.array(X),
        Y=np.array(Y4),
        name='average_n_steps',
        update='append',
        win=win
    )

    # win2 = vis.bar(
    #     X=np.array([followed_avg, finished]),
    #     opts=dict(
    #         rownames=['Followed %', 'Finished'],
    #         xtickstep=1
    #     ),
    #     win=win2
    # )
