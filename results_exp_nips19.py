import os
import numpy as np
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from glob import glob
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from bench.monitor import load_results

matplotlib.use('TkAgg')  # Can change to 'Agg' for non-interactive mode
plt.rcParams['svg.fonttype'] = 'none'

X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 1
COLORS = ['red', 'blue', 'orange', 'green', 'purple', 'magenta', 'lavender', 'cyan', 'yellow', 'black', 'pink',
        'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'turquoise',
        'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def window_func(x, y, window, func, alg, env):
    yw = rolling_window(y, window)
    yw_func = func(yw, axis=-1)
    x = x[window - 1:]
    return x, yw_func


def xtimesteps_by_alg(alg):
    xtimesteps = ''
    if alg == 'ppo' or alg == 'a2c' or alg=='acktr':
        xtimesteps = 'total_timesteps'
    elif alg == 'ddpg':
        xtimesteps = 'total/steps'
    elif alg == 'trpo':
        xtimesteps = 'TimestepsSoFar'

    return xtimesteps


def xepisodes_by_alg(alg):
    xepisodes = ''
    if alg == 'ppo' or alg == 'acktr':
        xepisodes = 'total_timesteps'
    elif alg == 'ddpg':
        xepisodes = 'total/episodes'
    elif alg == 'trpo':
        xepisodes = 'EpisodesSoFar'

    return xepisodes


def label_by_alg(pol_alg, vf_alg):
    start = ''
    if pol_alg == 'ppo':
        start = 'PPO'
    elif pol_alg == 'ddpg':
        start = 'DDPG'
    elif pol_alg == 'trpo':
        start = 'TRPO'
    elif pol_alg == 'a2c':
        start = 'A2C'
    elif pol_alg == 'acktr':
        start = 'ACKTR'

    if vf_alg == 'kalman':
        end = 'KOVA'
    elif vf_alg == 'a2c':
        end = 'RMSProp'
    elif vf_alg == 'acktr':
        return start
    else:
        end = 'Adam'

    return start + ' with ' + end


def comb_by_alg_env(alg, env):
    combs=[]
    if alg=='ppo':
        if env == 'Swimmer-v2':
            combs = ['99', '13']  # max-ratio
        elif env == 'Hopper-v2':
            combs = ['99', '15']  # max-ratio
        elif env == 'HalfCheetah-v2':
            combs = ['99', '12']  # max-ratio
        elif env == 'Ant-v2':
            combs = ['99', '15']  # max-ratio
        elif env == 'Walker2d-v2':
            combs = ['99', '13']  # max-ratio
    elif alg=='trpo':
        if env == 'Swimmer-v2':
            combs = ['99', '12']  # max-ratio
        elif env == 'Hopper-v2':
            combs = ['99', '7']  # max-ratio
        elif env == 'HalfCheetah-v2':
            combs = ['99', '13']  # max-ratio
        elif env == 'Walker2d-v2':
            combs = ['99', '16']  # max-ratio
        elif env == 'Ant-v2':
            combs = ['99', '13']  # max-ratio
    elif alg=='acktr':
        combs = ['96']

    return combs


def ts2xy(ts, xaxis):
    if xaxis == X_TIMESTEPS:
        x = np.cumsum(ts.l.values)
        y = ts.r.values
    elif xaxis == X_EPISODES:
        x = np.arange(len(ts))
        y = ts.r.values
    elif xaxis == X_WALLTIME:
        x = ts.t.values / 3600.
        y = ts.r.values
    else:
        raise NotImplementedError
    return x, y


def df2xy(df, xaxis, yaxis, alg):
    if xaxis == X_TIMESTEPS:
        xtimesteps = xtimesteps_by_alg(alg)
        x_index = df.columns.get_loc(xtimesteps)
    elif xaxis == X_EPISODES:
        xepisodes = xepisodes_by_alg(alg)
        x_index = df.columns.get_loc(xepisodes)
    else:
        raise NotImplementedError
    x = np.squeeze(df.as_matrix(columns=[df.columns[x_index]]))
    y_index = df.columns.get_loc(yaxis)
    y = np.squeeze(df.as_matrix(columns=[df.columns[y_index]]))
    return x, y


def plot_curves(xy_list, xaxis, yaxis, exp_args_vals=None, results_file_name=None):
        maxx = max(xy[0][-1] for xy in xy_list)
        minx = 0
        policy_alg = exp_args_vals[1]["pol_alg"][4:]
        combs = comb_by_alg_env(alg=policy_alg, env=exp_args_vals[0]["env"])

        for i, comb in enumerate(combs):
            env=exp_args_vals[0]["env"]
            info = [inf for j, inf in enumerate(exp_args_vals) if inf["comb"] == comb]
            pol_alg = info[0]["pol_alg"][4:]
            xy_window=[window_func(x, y, EPISODES_WINDOW, np.mean, pol_alg, env) for idx, (x, y)
                       in enumerate(xy_list) if exp_args_vals[idx]["comb"]==comb]
            data = np.array(xy_window)

            data_mean = np.mean(data, axis=0)
            data_std = np.std(data, axis=0)
            color = COLORS[i]
            pol_alg = info[0]["pol_alg"][4:]
            vf_alg = info[0]["vf_alg"][3:]
            label = label_by_alg(pol_alg, vf_alg)

            # if vf_alg=='kalman':
            #     label=label + ' - ' + info[0]["eta"]

            plt.plot(data_mean[0], data_mean[1], color=color, linewidth=2, label=label)
            plt.fill_between(data_mean[0], data_mean[1] + data_std[1], data_mean[1] - data_std[1], facecolor=color, alpha=0.2)
            plt.yticks(size=10)
            if xaxis == X_TIMESTEPS:
                plt.xlim(minx, maxx)
                plt.xticks((minx, (minx+maxx)/2, maxx), (str(minx), "{:.0e}".format(round((minx+maxx)/2, -5)),
                                                         "{:.0e}".format(round(maxx, -5)) ),size=10)



        if exp_args_vals[0]["env"]=='HalfCheetah-v2' and yaxis=="policy_entropy":

            for i, comb in enumerate(combs):
                env = exp_args_vals[0]["env"]
                xy_window = [window_func(x, y, EPISODES_WINDOW, np.mean, policy_alg, env) for idx, (x, y) in
                             enumerate(xy_list) if exp_args_vals[idx]["comb"] == comb]

                data = np.array(xy_window)

                data_mean = np.mean(data, axis=0)
                color = COLORS[i]
                plt.axes([.83, .47, .1, .1])
                plt.plot(data_mean[0], data_mean[1], color=color, linewidth=2)
                plt.xlim(7e5, 1e6)
                plt.ylim(-.1, 0.2)
                plt.title('Zoom In', fontsize=10)
                plt.xticks((7e5, 1e6), ("7e+05", "1e+06"), size=8)
                plt.yticks((-1, -.6, -.2, .2), ("-1.0", "-0.6", "-0.2", "0.2"), size=8)
                plt.grid(which='both')


class LoadMonitorResultsError(Exception):
    pass

EXT = "preogress.csv"


def get_monitor_files(dir):
    return glob(osp.join(dir, "*" + EXT))


def load_results_progress(dir, alg):
    import pandas
    progress_files = (
        glob(osp.join(dir, "*progress.csv")))
    if not progress_files:
        raise LoadMonitorResultsError("no monitor files of the form *%s found in %s" % (EXT, dir))
    dfs = []
    for fname in progress_files:
        with open(fname, 'rt') as fh:
            df = pandas.read_csv(fh, header=0, index_col=None)
        dfs.append(df)
    df = pandas.concat(dfs)
    xtimesteps = xtimesteps_by_alg(alg)
    df.sort_values(xtimesteps, inplace=True)

    df.reset_index(inplace=True)
    return df


def plot_loss(dirs, alg, num_timesteps, xaxis, task_name, exp_args_vals=None):
    results_file_name = 'progress' # 'monitor'

    losses=[]
    if alg=='ppo' or alg=='acktr':
        losses = ['eprewmean', "policy_entropy", "policy_loss"] # "value_loss", "approxkl"
    elif alg=='trpo':
        losses = ['eprewmean', "entropy", "surrgain"] # "meankl"
    elif alg=='ddpg':
        losses = ['eprewmean', "train/loss_actor"] # "train/loss_critic"
    len_losses = len(losses)

    fig=plt.figure()
    with PdfPages(dirs[0][0] + '/train_reward.pdf') as pdf:
        yaxis = 'eprewmean'
        for i, env in enumerate(dirs):
            dflist = []
            tslist = []
            algs = []
            print("alg", alg)
            for k, dir in enumerate(env):
                if results_file_name == 'monitor':
                    ts = load_results(dir)
                    ts = ts[ts.l.cumsum() <= num_timesteps]
                    tslist.append(ts)
                else:
                    alg_temp = exp_args_vals[i][k]["pol_alg"][4:]
                    algs.append(alg_temp)
                    df = load_results_progress(dir, alg_temp)
                    dflist.append(df)
            temp = 251 + i
            ax=plt.subplot(temp)

            if results_file_name == 'monitor':
                xy_list = [ts2xy(ts, xaxis) for ts in tslist]
            else:
                xy_list = [df2xy(df, xaxis, yaxis, algs[k]) for k, df in enumerate(dflist)]

            plot_curves(xy_list, xaxis, 'Mean episode reward', exp_args_vals[i], results_file_name)
            if i==0:
                plt.ylabel('Mean episode reward', fontsize=12)

            plt.tight_layout(pad=0.01, w_pad=0.01, h_pad=0.2)
            plt.subplots_adjust(wspace=0.5)
            plt.xlabel("Time steps", fontsize=11)
            plt.title(exp_args_vals[i][0]["env"], fontsize=13)
            ax.grid(which='both')
            handles, labels = ax.get_legend_handles_labels()

        fig.legend(handles, labels, fontsize=10, ncol=5, bbox_to_anchor=(0.55, 0.22), loc="center",
                   fancybox=True, shadow=True, borderaxespad=0)
        pdf.savefig(bbox_inches='tight')

    if results_file_name is not 'monitor' and losses is not None:
        fig = plt.figure()
        with PdfPages(dirs[0][0] + '/losses.pdf') as pdf:
            for i, env in enumerate(dirs):
                dflist = []
                algs = []
                for k, dir in enumerate(env):
                    alg_temp = exp_args_vals[i][k]["pol_alg"][4:]
                    algs.append(alg_temp)
                    df = load_results_progress(dir, alg_temp)
                    dflist.append(df)
                for j in range(len(losses)):
                    count = (i+1) + j*len(dirs)
                    print("i=", i, "j=", j, "count=", count)
                    yaxis=losses[j]
                    xy_list = [df2xy(df, xaxis, yaxis, algs[k]) for k, df in enumerate(dflist)]
                    ax = plt.subplot(len_losses, len(dirs), count)
                    plot_curves(xy_list, xaxis, yaxis, exp_args_vals[i], plot_curves)
                    if j==0:
                        plt.title(exp_args_vals[i][0]["env"], fontsize=14)
                    if j==len_losses-1:
                        plt.xlabel('Time steps', fontsize=11)
                    if yaxis == 'eprewmean' and i == 0:
                        plt.ylabel("Mean Reward", fontsize=14)
                    elif yaxis == "policy_loss" and i == 0:
                        plt.ylabel("Policy loss", fontsize=14)
                    elif yaxis == "policy_entropy" and i == 0:
                        plt.ylabel("Policy entropy", fontsize=14)
                    elif yaxis == "optimgain" and i == 0:
                        plt.ylabel("Policy + Entropy loss", fontsize=14)
                    elif (yaxis == "meankl" or yaxis == "approxkl" )and i == 0:
                        plt.ylabel("Mean KL", fontsize=14)
                    elif yaxis == "entloss" and i == 0:
                        plt.ylabel("Entropy loss", fontsize=14)
                    elif yaxis == "surrgain" and i == 0:
                        plt.ylabel("Policy loss", fontsize=14)
                    elif yaxis == "entropy" and i == 0:
                        plt.ylabel("Entropy", fontsize=14)
                    elif yaxis == "value_loss" and i == 0:
                        plt.ylabel("Policy loss", fontsize=14)

                    ax.grid(which='both')
                    ymin, ymax = ax.get_ylim()

                    l = matplotlib.ticker.AutoLocator()
                    l.create_dummy_axis()
                    l.tick_values(ymin, ymax)
                    handles, labels = ax.get_legend_handles_labels()

            fig.legend(handles, labels, fontsize=10, ncol=4, bbox_to_anchor=(0.55, 0.01), loc="lower center",
                       fancybox=True, shadow=True, borderaxespad=0)
            plt.subplots_adjust(top=0.88, bottom=0.12, left=0.10, right=0.95, hspace=0.25, wspace=0.55)
            pdf.savefig(bbox_inches='tight')

def main():
    import argparse
    import os

    alg = 'ppo'
    envs = ['Swimmer-v2', 'Hopper-v2', 'HalfCheetah-v2', 'Ant-v2', 'Walker2d-v2']
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--results_dir', default="/results/"+alg)
    parser.add_argument('--exp_dirs', help='List of log directories', nargs = '*', default=envs)
    parser.add_argument('--num_timesteps', type=int, default=int(1e6))
    parser.add_argument('--xaxis', help = 'Varible on X-axis', default = X_TIMESTEPS) # X_TIMESTEPS X_EPISODES X_WALLTIME
    args = parser.parse_args()
    args.dirs = [os.path.join(args.results_dir, dir) for dir in args.exp_dirs]
    subdirs = [os.listdir(dir) for dir in args.dirs]
    if alg == 'ppo':
        arguments_names = ["exp", "date", "env", "pol_alg", "vf_alg", "learning_rate", "onv_coeff", "eta", "onv_type",
                           "body_mass", "batch_size", "seed", "comb", "last_layer", "separateVars",
                           "eval"]
    else:
        arguments_names = ["exp", "date", "env", "pol_alg", "vf_alg", "learning_rate", "onv_coeff", "eta", "onv_type",
                       "body_mass", "seed", "comb", "last_layer", "separateVars",
                       "eval"]
    temp2 = []
    exp_args_vals = []
    for idx, env in enumerate(subdirs):
        temp = [os.path.join(args.dirs[idx], dir) for dir in env]
        parse_dir = [string.split('_') for string in env]
        temp3 = [dict(zip(arguments_names, p)) for p in parse_dir]
        for d in temp3:
            d['plot_mean'] = True
        temp2.append(temp)
        exp_args_vals.append(temp3)
    args.dirs = temp2
    plot_loss(args.dirs, alg, args.num_timesteps, args.xaxis, 'EpRewMean', exp_args_vals=exp_args_vals)
    plt.show()

if __name__ == '__main__':
    main()