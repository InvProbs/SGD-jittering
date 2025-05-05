import matplotlib, csv
import matplotlib.pyplot as plt
import torch, sys

sys.path.append("../")

from networks import seis_nets as nets
from utils.dataloader import *
from utils.misc import *
import configargparse, os
from tqdm import tqdm
from pandas import *
from utils.training_setup import *
from operators.operator import *
from operators import training_mode as tr
from utils.gen_seismic_data import gen_conv_mtx

matplotlib.use("Qt5Agg")
parser = configargparse.ArgParser()
parser.add_argument('-c', '--my-config', is_config_file=True, help='config file path')
parser.add_argument("--file_name", type=str, default="6-deconv/", help="saving directory")
parser.add_argument("--data_path", type=str, default="", help="path to load data")
parser.add_argument("--save_path", type=str, default="../prox_saved_models/", help="model saving directory")
parser.add_argument("--load_path", type=str, default="", help="model loading path")

# TRAINING PARAMETERS
parser.add_argument('--pretrain', type=bool, default=True, help='if True: load model')
parser.add_argument('--train', type=bool, default=False, help='train or eval mode')
parser.add_argument('--n_epochs', type=int, default=130)
parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
parser.add_argument('--batch_size_val', type=int, default=4, help='val/eval Batch size')
parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lr_gamma', type=float, default=0.5)
parser.add_argument('--sched_step', type=int, default=300)
parser.add_argument('--train_mode', type=str, default="MSE", help="AT, MSE, input-jitter, sgd-jitter")
parser.add_argument('--test_mode', type=str, default="block", help="AT, MSE, Xg_rand, block")
parser.add_argument('--test_full_digits', type=bool, default=False, help="eval with full digits or 0")
parser.add_argument('--epsilon', type=float, default=1, help="Adv.attck: ||e||<=epsilon")
parser.add_argument('--n_PGD_steps', type=int, default=20, help='Adv.attck: max #PGD steps')
parser.add_argument('--PGD_stepsize', type=float, default=0.1, help='Adv.attck: step-size')
parser.add_argument('--sigma_w', type=float, default=0.05, help="variance of input jittering")
parser.add_argument('--sigma_wk', type=float, default=0.1, help="variance of SGD jittering")

# DATA
parser.add_argument('--dataset', type=str, default="seis", help='dataset: MRI or seis')
parser.add_argument('--nc', type=int, default=1, help='number of channels in an image')
parser.add_argument('--noise_level', type=float, default=0.01)

# Networks
parser.add_argument('--eta', type=float, default=-1, help='initial eta, step size for GD in LU')
parser.add_argument('--hid_dim', type=int, default=128, help='hidden dim of MLP')
parser.add_argument('--n_layers', type=int, default=8, help='n_layer DnCNN')
parser.add_argument('--maxiters', type=int, default=5, help='Max # iterations for LU')
parser.add_argument('--seed', type=int, default=11, help='seed')

args = parser.parse_args()
args.shared_eta = True
print(args)
cuda = True if torch.cuda.is_available() else False
print('cuda is available: ' + str(cuda))
set_seed(args.seed)

""" LOAD DATA """
args.save_path = "../prox_saved_models/"
args.file_name = args.file_name + args.train_mode + '/'
if args.test_mode == 'AT': args.batch_size_val = 1
tr_loader, tr_length, val_loader, val_length, ts_loader, ts_length = load_data(args)
set_save_path(args)

""" NETWORK SETUP """
W = gen_conv_mtx().to(args.device)
forward_op = op.convolution(torch.clone(W), trace_length=352)
measurement_process = op.OperatorPlusNoise(forward_op, noise_sigma=args.noise_level).to(args.device)
dncnn = nets.single_layer(args).to(args.device)
net = nets.seis_proxgd(forward_op, dncnn, args).to(args.device) # SPGD jitter noise added in net

""" TRAINING PARAMETERS SETUP """
criteria = nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), lr=args.lr)

""" LOAD MODEL """
if args.pretrain:
    load_model(args, net)

if args.train:
    # set trajectory saving directory
    trajectory_path = args.save_path + '/trajectory.csv'
    fh = open(trajectory_path, 'a')
    csv_writer = csv.writer(fh)
    csv_writer.writerow(['train loss', 'val loss'])
    fh.close()

    """ BEGIN TRAINING """
    for epoch in range(args.n_epochs):
        loss_meters = AverageMeter()
        val_meter = AverageMeter()
        with tqdm(total=(tr_length - tr_length % args.batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, args.n_epochs))
            for X, y in tr_loader:
                bs = X.shape[0]
                X, y = X.unsqueeze(1).to(args.device).to(torch.float32), y.unsqueeze(1).to(args.device).to(
                    torch.float32)
                maxVal, y, X = normalize_seis(y, X, bs)

                # under adversarial training
                if args.train_mode == 'AT':
                    delta, loss_list = tr.PGD(net, X, torch.clone(y), y, args.epsilon, args.PGD_stepsize,
                                                 args.n_PGD_steps, dim=3, eps_mode='l2')
                    y += delta
                # with input injection
                elif args.train_mode == 'input-jitter':
                    y += torch.randn_like(y) * args.sigma_w
                X0 = torch.clone(y)

                # reconstruction
                Xk = net(X0, y)
                loss = criteria(Xk.squeeze(), X.squeeze())
                loss_meters.update(loss.item(), bs)
                opt.zero_grad()
                loss.backward()
                opt.step()

                torch.cuda.empty_cache()
                dict = {f'tr_mse': f'{loss_meters.avg:.6f}'}
                dict.update({'val_mse': f'{val_meter.avg:.6f}'})
                _tqdm.set_postfix(dict)
                _tqdm.update(bs)

            # save model and figures
            if (epoch + 1) % 10 == 0:
                state = {
                    'epoch': epoch,
                    'state_dict': net.state_dict()}
                torch.save(state, os.path.join(args.save_path, f'epoch_{epoch}.state'))
                plot_2D(X, Xk, y, args.save_path, epoch, title=args.train_mode)

            # validation
            with torch.no_grad():
                for X, y in val_loader:
                    bs = X.shape[0]
                    X, y = X.unsqueeze(1).to(args.device).to(torch.float32), y.unsqueeze(1).to(args.device).to(
                        torch.float32)
                    maxVal, y, X = normalize_seis(y, X, bs)
                    X0 = torch.clone(y)
                    Xk = net(X0, y)

                    val_loss = criteria(Xk, X)
                    val_meter.update(val_loss.item(), bs)

                    dict = {f'tr_mse': f'{loss_meters.avg:.6f}'}
                    dict.update({'val_mse': f'{val_meter.avg:.6f}'})
                    _tqdm.set_postfix(dict)
                    _tqdm.update(bs)

            # write losses
            fh = open(trajectory_path, 'a', newline='')
            csv_writer = csv.writer(fh)
            csv_writer.writerow([loss_meters.avg, val_meter.avg])
            fh.close()

        """ READ and PLOT TRAJECTORY """
        traj = read_csv(trajectory_path)
        tr_list = traj["train loss"].tolist()
        val_list = traj["val loss"].tolist()
        plt.figure()
        plt.semilogy(np.arange(len(tr_list)), tr_list)
        plt.semilogy(np.arange(len(val_list)), val_list)
        plt.legend(['train', 'val'])
        plt.xlabel('epochs')
        plt.ylabel('MSE')
        plt.savefig(os.path.join(args.save_path, 'trajectory.png'))
        plt.tight_layout()
        plt.close()

else:
    # set evaluation metrics
    criteria = nn.MSELoss()
    criteria_title = ['mse', 'avgInit', 'avgPSNR', 'deltaPSNR', 'avgSSIM']
    len_meter = len(criteria_title)
    loss_meters = [AverageMeter() for _ in range(len_meter)]
    ts_mse_meters = [AverageMeter() for _ in range(args.maxiters + 1)]

    # initialize evaluation save path
    trajectory_path = args.save_path + '/trajectory_test.csv'
    fh = open(trajectory_path, 'a')
    csv_writer = csv.writer(fh)
    csv_writer.writerow(criteria_title)
    fh.close()

    # start evaluation
    with tqdm(total=(ts_length - ts_length % args.batch_size_val)) as _tqdm:
        _tqdm.set_description('epoch: {}/{}'.format(1, 1))
        for X, y in val_loader:
            bs = X.shape[0]
            X, y = X.unsqueeze(1).to(args.device).to(torch.float32), y.unsqueeze(1).to(args.device).to(torch.float32)
            maxVal, y, X = normalize_seis(y, X, bs)

            # if Xg = X + random noise
            if args.test_mode == 'Xg_rand':
                g = torch.randn_like(X) * args.sigma_w
                X += g
                y = measurement_process(X)

            # if Xg = X + useful detail
            elif args.test_mode == 'block':
                X = add_detail_to_x(X, args)
                y = measurement_process(X)

            # under adversarial attack
            if args.test_mode == 'AT':
                delta, loss_list = tr.PGD(net, X, y, y, args.epsilon, args.PGD_stepsize, args.n_PGD_steps, dim=3,
                                          eps_mode='l2')
                y += delta

            # reconstruction
            with torch.no_grad():
                X0 = torch.clone(y)
                Xk = net(X0, y)

                # evaluation
                ts_loss = criteria(Xk, X)
                init_psnr, recon_psnr, delta_psnr, ssim = compute_metrics1chan(Xk, X, X0)
                criteria_list = [ts_loss, init_psnr, recon_psnr, delta_psnr, ssim]
                for k in range(len_meter):
                    loss_meters[k].update(criteria_list[k].item(), bs)

                torch.cuda.empty_cache()
                dict = {f'{criteria_title[k]}': f'{loss_meters[k].avg:.6f}' for k in range(len_meter)}
                _tqdm.set_postfix(dict)
                _tqdm.update(bs)

        # write evaluation metrics
        fh = open(trajectory_path, 'a', newline='')
        csv_writer = csv.writer(fh)
        csv_writer.writerow([loss_meters[k].avg for k in range(len_meter)])
        fh.close()

        title = "train mode: " + str(args.train_mode) + ', eval mode: ' + str(args.test_mode)
        plot_2D(X, Xk, y, args.save_path, -1, title="tr: " + args.train_mode + " ts: " + args.test_mode)
