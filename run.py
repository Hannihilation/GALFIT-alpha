from simple_env import GalfitEnv
from galfit_alpha import GalfitAlpha
from DQL import DeepQLearning
import platform
from numpy import random
import torch
# from torchsummary import summary

os_name = platform.system()
if os_name == 'Darwin':
    train_file = ('IC5240', 'NGC1326')
    test_file = ('IC5240', 'NGC1326')
elif os_name == 'Linux':
    CGS_file = ('IC5240', 'NGC1326', 'NGC1357', 'NGC1411', 'NGC1533', 'NGC1600',
                'NGC2784', 'NGC4786', 'NGC6118', 'NGC7083', 'NGC7329', 'NGC945')
    train_size = 10
    train_file = random.choice(CGS_file, train_size, replace=False)
    test_file = (x for x in CGS_file if x not in train_file)
pre_path = './CGS/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

if __name__ == '__main__':
    Q_net = GalfitAlpha(GalfitEnv.state_num, GalfitEnv.channel_num,
                        GalfitEnv.image_size, GalfitEnv.action_num, device)
    # summary(Q_net.float(), [(GalfitEnv.state_num,), (GalfitEnv.channel_num,
    #         GalfitEnv.image_size, GalfitEnv.image_size)], device='cpu')
    DQL = DeepQLearning(Q_net, 0.01, 0.9, 0.9, 1000, 32)
    step = 0
    for i in range(1000):
        galaxy = train_file[i % train_size]
        env = GalfitEnv(pre_path+galaxy+'/'+galaxy+'_R_reg.fits')
        s = env.current_state
        while True:
            a = DQL.choose_action(s)
            s_, r, done = env.step(a)
            DQL.store_transition(s, a, r, s_)
            if step > 0 and step % 10 == 0:
                DQL.learn()
            if done:
                break
            s = s_
            step += 1
    DQL.eval_net.save('./model.pkl')
    DQL.plot_loss()
