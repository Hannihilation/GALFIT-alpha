from simple_env import GalfitEnv
from galfit_alpha import GalfitAlpha
from DQL import DeepQLearning
import platform
from numpy import random
import torch
import argparse
from torchinfo import summary

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str,
                    default='CGS', help='select data set')
parser.add_argument('-m', '--multithread', type=int,
                    default=1, help='use multithread with n threads')
parser.add_argument('-t', '--train-size', type=int,
                    default=10, help='set train size')
parser.add_argument('-b', '--batch-size', type=int,
                    default=32, help='set batch size')
parser.add_argument('-e', '--epoch', type=int, default=100, help='set epoch')
parser.add_argument('-ls', '--learning-step', type=int,
                    default=10, help='set steps between learning')
parser.add_argument('-o', '--output-model', type=str,
                    default='./model.pkl', help='set output model path')
parser.add_argument('-s', '--summary', action='store_true',
                    help='show model summary')

os_name = platform.system()
if os_name == 'Darwin':
    train_size = 2
    train_file = ('IC5240', 'NGC1326')
    test_file = ('IC5240', 'NGC1326')
elif os_name == 'Linux':
    CGS_file = ('IC5240', 'NGC945', 'NGC1326', 'NGC1357', 'NGC1411', 'NGC1533', 'NGC1600',
                'NGC2784', 'NGC6118', 'NGC7083', 'NGC7329')
    train_size = 2
    # train_file = random.choice(CGS_file, train_size, replace=False)
    train_file = ('NGC1326', 'IC5240')
    # test_file = (x for x in CGS_file if x not in train_file)
pre_path = './CGS/'


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    Q_net = GalfitAlpha(GalfitEnv.state_num, GalfitEnv.channel_num,
                        GalfitEnv.image_size, GalfitEnv.action_num, device)
    DQL = DeepQLearning(Q_net, 0.01, 0.9, 0.9, 1000, args.batch_size)
    step = 0
    for i in range(args.epoch):
        galaxy = train_file[i % train_size]
        env = GalfitEnv(pre_path+galaxy+'/'+galaxy+'_R_reg.fits')
        s = env.current_state
        while True:
            a = DQL.choose_action(s)
            s_, r, done = env.step(a)
            print(f'step: {step}, action: {a}, current state: {s_[0]}\n')
            DQL.store_transition(s, a, r, s_)
            if step > 0 and step % args.learning_step == 0:
                DQL.learn()
            step += 1
            if done:
                break
            s = s_
    DQL.eval_net.save(args.output_model)
    DQL.plot_loss()


def sum(batch_size):
    Q_net = GalfitAlpha(GalfitEnv.state_num, GalfitEnv.channel_num,
                        GalfitEnv.image_size, GalfitEnv.action_num, 'cpu')
    summary(Q_net.float(), [(batch_size, GalfitEnv.state_num),
                            (batch_size, GalfitEnv.channel_num,
                            GalfitEnv.image_size, GalfitEnv.image_size)],
            device='cpu')


if __name__ == '__main__':
    args = parser.parse_args()
    if args.summary:
        sum(args.batch_size)
    else:
        run()
