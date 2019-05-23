import itertools

def grid_search():
    envs = ['Ant-v2'] # 'HalfCheetah-v2' , 'Hopper-v2', 'Swimmer-v2', 'Walker2d-v2', 'Ant-v2'
    kalman_lr = [0.01] #[1., 0.1, 0.01]
    kalman_eta = [0.1, 0.01, 0.001]
    kalman_onv_coeff = [1.] # onv=observation noise variance
    kalman_onv_type = ['max-ratio'] #['batch-size', 'ratio', 'max-ratio', 'empirical-variance', 'empirical-variance-mean]

    combinations = list(itertools.product(envs, kalman_lr, kalman_eta, kalman_onv_coeff, kalman_onv_type))

    for inx, comb in enumerate(combinations):
        if inx >= 0:
            from kalman_algorithms.experiments.run import main_run
            print('comb: ', inx, comb)
            env=comb[0]
            kalman_lr = comb[1]
            kalman_eta = comb[2]
            kalman_onv_coeff = comb[3]
            kalman_onv_type = comb[4]
            comb_num = inx
            main_run(env, kalman_lr, kalman_eta, kalman_onv_coeff, kalman_onv_type, comb_num)

if __name__ == '__main__':
    grid_search()
