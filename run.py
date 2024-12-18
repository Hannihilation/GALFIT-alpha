from task import Config
from simple_env import GalfitEnv
from galfit_alpha import GalfitAlpha
from DQL import DeepQLearning
from numpy import random
import torch
import argparse
from torchinfo import summary
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, default='S82',
                    help='select data set', choices=['S82', 'CGS', 'mock'])
parser.add_argument('-m', '--multithread', type=int, default=1,
                    help='use multithread with n threads', metavar='n')
parser.add_argument('-t', '--train-size', type=int,
                    default=10, help='set train size', metavar='size')
parser.add_argument('-b', '--batch-size', type=int,
                    default=32, help='set batch size', metavar='size')
parser.add_argument('-e', '--epoch', type=int, default=1000,
                    help='set epoch', metavar='epoch')
parser.add_argument('-ls', '--learning-step', type=int, default=10,
                    help='set steps between learning', metavar='step')
parser.add_argument('-o', '--output-model', type=str,
                    default='./model.pkl', help='set output model path', metavar='path')
parser.add_argument('-s', '--summary', action='store_true',
                    help='show model summary')
parser.add_argument('-p', '--plot-figure', action='store_true',
                    help='plot figure after training')


def sum(batch_size):
    Q_net = GalfitAlpha(GalfitEnv.state_num, GalfitEnv.channel_num,
                        GalfitEnv.image_size, GalfitEnv.action_num, 'cpu')
    summary(Q_net.float(), [(batch_size, GalfitEnv.state_num),
                            (batch_size, GalfitEnv.channel_num,
                            GalfitEnv.image_size, GalfitEnv.image_size)],
            device='cpu')


def plot_final_image(log_file, train_size):
    with open(log_file, 'r') as file:
        for input_file in file.readlines()[:train_size]:
            input_file = input_file.strip('\n')
            config = Config(input_file=input_file, psf_file='S82/psf_r_cut65x65.fits',
                            pixel_scale=0.396 / 3600, psf_scale=1, zeropoint=24)
            config.plot(pro_1D=True)


def gen_data(data, size):
    if data == "S82":
        path = Path('./S82')
        total = [f.as_posix() for f in path.glob('**/*ID*.fits') if f.stem[-1].isdigit()]
        masks = [f.as_posix() for f in path.glob('**/*_mm.fits')]
        psf = [(path / 'psf_r_cut65x65.fits').as_posix()] * len(total)
        psf_scale = 1
        zeropoint = 24
        pixel_scale = 0.396
    elif data == "CGS":
        path = Path('./CGS')
        total = [f.as_posix() for f in path.glob('**/*_R_reg.fits')]
        masks = [f.as_posix() for f in path.glob('**/*_R_reg_mm.fits')]
        psf = [f.as_posix() for f in path.glob('**/*_R_reg_ep.fits')]
        psf_scale = None
        zeropoint = None
        pixel_scale = None
    elif data == "mock":
        path = Path('./mock_galaxy')
        total = [f.as_posix() for f in path.glob('**/input.fits')]
        masks = ['none'] * len(total)
        psf = ['./S82/psf_r_cut65x65.fits'] * len(total)
        psf_scale = 1
        zeropoint = 24
        pixel_scale = 0.396
    else:
        raise ValueError("data set must be 'S82', 'CGS' or 'mock'")

    train_index = random.choice(len(total), size, replace=False)
    test_index = [i for i in range(len(total)) if i not in train_index]
    train_data = [(total[i], masks[i], psf[i]) for i in train_index]
    test_data = [(total[i], masks[i], psf[i]) for i in test_index]
    return train_data, test_data, psf_scale, zeropoint, pixel_scale


def run(train_data, test_data, psf_scale, zeropoint, pixel_scale):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    train_size = len(train_data)
    Q_net = GalfitAlpha(GalfitEnv.state_num, GalfitEnv.channel_num,
                        GalfitEnv.image_size, GalfitEnv.action_num, device)
    DQL = DeepQLearning(Q_net, 0.01, 0.9, 0.9, 1000, args.batch_size)
    step = 0
    for i in range(args.epoch):
        galaxy = train_data[i % train_size]
        config = Config(input_file=galaxy[0],
                        mask_file=galaxy[1],
                        psf_file=galaxy[2],
                        psf_scale=psf_scale, zeropoint=zeropoint,
                        pixel_scale=pixel_scale)
        env = GalfitEnv(config)
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
    # DQL.eval_net.save(args.output_model)
    torch.save(DQL.eval_net.state_dict(), args.output_model)
    DQL.plot_loss()

if __name__ == '__main__':
    args = parser.parse_args()
    if args.summary:
        sum(args.batch_size)
    elif args.plot_figure:
        plot_final_image('./run.log', args.train_size)
    else:
        data = gen_data(args.data, args.train_size)
        print(*(d[0] for d in data[0]), sep='\n')
        run(*data)
