import numpy as np

vis = None
win = None

X = []
Y = []
Y2 = []

last_episodes = 0
last_violations = 0
last_unsafe = 0


def visdom_plot(
        total_num_steps,
        mean_reward,
        n_violation,
        episodes
):
    # Lazily import visdom so that people don't need to install visdom
    # if they're not actually using it
    from visdom import Visdom

    global vis
    global win
    global last_episodes
    global last_violations
    global last_unsafe

    if vis is None:
        vis = Visdom(use_incoming_socket=False)
        assert vis.check_connection()
        # Close all existing plots
        vis.close()

    violation_change = n_violation - last_violations
    last_violations = n_violation

    episode_change = episodes - last_episodes
    last_episodes = episodes

    unsafe_actions_per_episode = 0
    if n_violation > 0:
        unsafe_actions_per_episode = round(episodes / n_violation, 2)
        # print('episodes ' + str(episodes))
        # print('unsafe ' + str(unsafe_actions_per_episode))

    X.append(total_num_steps)
    Y.append(mean_reward)
    Y2.append(violation_change)

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
