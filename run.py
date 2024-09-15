from simple_env import GalfitEnv
from galfit_alpha import GalfitAlpha, DeepQLearning
import platform
import matplotlib.pyplot as plt

os_name = platform.system()
if os_name == 'Darwin':
    CGS_file = ('IC5240', 'NGC1326')
elif os_name == 'Linux':
    CGS_file = ('IC5240', 'NGC1326')
pre_path = './CGS/'


if __name__ == '__main__':
    Q_net = GalfitAlpha(2, 2, 2042, 5)
    DQL = DeepQLearning(Q_net, 0.01, 0.9, 0.9, 1000, 32)
    step = 0
    for i in range(1000):
        galaxy = CGS_file[i % len(CGS_file)]
        env = GalfitEnv(pre_path+galaxy+'/'+galaxy+'_R_reg.fits')
        s = env.current_state
        while True:
            a = DQL.choose_action(s)
            s_, r, done = env.step(a)
            DQL.store_transition(s, a, r, s_)
            if step > 0 and step % 5 == 1:
                DQL.learn()
            if done:
                break
            s = s_
            step += 1
    DQL.eval_net.save('./model.pkl')
    DQL.plot_loss()
