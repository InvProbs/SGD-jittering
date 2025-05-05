import matplotlib, csv
import matplotlib.pyplot as plt
import torch, sys
sys.path.append("../")

from networks import mri_nets as nets
from utils.dataloader import *
from utils.misc import *
import configargparse, os
from tqdm import tqdm
from pandas import *
from utils.training_setup import *
from operators.operator import *
from operators import training_mode as tr


matplotlib.use("Qt5Agg")
parser = configargparse.ArgParser()
parser.add_argument('-c', '--my-config', is_config_file=True, help='config file path')
parser.add_argument("--file_name", type=str, default="5-mri/", help="saving directory")
parser.add_argument("--data_path", type=str, default='D:/data/fastMRI/')
parser.add_argument("--save_path", type=str, default="../prox_saved_models/", help="network saving directory")
parser.add_argument("--load_path", type=str, default="", help="model loading path")

# TRAINING and EVAL PARAMETERS
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
parser.add_argument('--test_mode', type=str, default="tumor", help="MSE, AT, Xg_rand, block, tumor")
parser.add_argument('--epsilon', type=float, default=1, help="Adv.attck: ||e||<=epsilon")
parser.add_argument('--n_PGD_steps', type=int, default=20, help='Adv.attck: max #PGD steps')
parser.add_argument('--PGD_stepsize', type=float, default=0.1, help='Adv.attck: step-size')
parser.add_argument('--sigma_w', type=float, default=0.05, help="variance of input jittering")
parser.add_argument('--sigma_wk', type=float, default=0.01, help="variance of SGD jittering")

# DATA
parser.add_argument('--dataset', type=str, default="MRI", help='dataset: MRI or seis')
parser.add_argument('--nc', type=int, default=2, help='number of channels in an image')

# Networks
parser.add_argument('--eta', type=float, default=-1, help='initial eta, step size for GD in LU')
parser.add_argument('--hid_dim', type=int, default=128, help='hidden dim of MLP')
parser.add_argument('--n_layers', type=int, default=8, help='n_layer DnCNN')
parser.add_argument('--maxiters', type=int, default=5, help='Max # iterations for LU')
parser.add_argument('--seed', type=int, default=11, help='set random seed')

args = parser.parse_args()
args.shared_eta = True
print(args)
cuda = True if torch.cuda.is_available() else False
print('cuda is available: ' + str(cuda))
set_seed(args.seed)

""" LOAD DATA """
args.file_name = args.file_name + args.train_mode + '/'
tr_loader, tr_length, val_loader, val_length, ts_loader, ts_length = load_data(args)
set_save_path(args)

""" NETWORK SETUP """
forward_op = mrimodel.cartesianSingleCoilMRI().to(args.device)
measurement_process = op.OperatorPlusNoise(forward_op, noise_sigma=0.0).to(args.device)
dncnn = nets.DnCNN(args.nc, num_of_layers=args.n_layers).to(args.device)
net = nets.MRI_prox(forward_op, dncnn, args).to(args.device) # SPGD jitter noise added in net

""" TRAINING PARAMETERS SETUP """
criteria = nn.MSELoss()
opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.sched_step, gamma=args.lr_gamma)

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
                X, y = X.to(args.device), y.to(args.device)

                # under adversarial training
                if args.train_mode == 'AT':
                    delta, loss_list = tr.PGD_v2(net, X, forward_op, y, args.epsilon, args.PGD_stepsize, args.n_PGD_steps, dim=3, eps_mode='l2')
                    y += delta

                # with input injection
                elif args.train_mode == 'input-jitter':
                    y += torch.randn_like(y) * args.sigma_w
                X0 = forward_op.adjoint(y)

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
            if (epoch + 1) % 1 == 0:
                state = {
                    'epoch': epoch,
                    'state_dict': net.state_dict()}
                torch.save(state, os.path.join(args.save_path, f'epoch_{epoch}.state'))
                plot_MRI(Xk, X0, X, criteria, args.save_path, epoch)

            # validation
            with torch.no_grad():
                for X, y in val_loader:
                    bs = X.shape[0]
                    X, y = X.to(args.device), y.to(args.device)
                    X0 = forward_op.adjoint(y)
                    Xk = net(X0, y)

                    val_loss = criteria(Xk, X)
                    val_meter.update(val_loss.item(), bs)

                    dict = {f'tr_mse': f'{loss_meters.avg:.6f}'}
                    dict.update({'val_mse': f'{val_meter.avg:.6f}'})
                    _tqdm.set_postfix(dict)
                    _tqdm.update(bs)

            # write losses
            fh = open(trajectory_path, 'a', newline='')  # a for append
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
    if args.test_mode != 'tumor':
        # redefine measurement process, y = Ax + z
        measurement_process = OperatorPlusNoise(forward_op, 0.05)  # same as in training data

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
        with tqdm(total=(ts_length - ts_length % args.batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(1, 1))
            for X, y in ts_loader:
                bs = X.shape[0]
                X = X.to(args.device)
                # if Xg = X + random noise
                if args.test_mode == 'Xg_rand':
                    g = torch.randn_like(X) * args.sigma_w
                    X += g
                # if Xg = X + useful detail
                elif args.test_mode == 'block':
                    X = add_detail_to_x(X, args)
                # generate measurement yg
                y = measurement_process(X)

                # under adversarial attack
                if args.test_mode == 'AT':
                    delta, loss_list = tr.PGD_v2(net, X, forward_op, y, args.epsilon, args.PGD_stepsize, args.n_PGD_steps, dim=3, eps_mode='l2')
                    y += delta

                # reconstruction
                with torch.no_grad():
                    X0 = forward_op.adjoint(y)
                    Xk = net(X0, y)

                    # evaluation
                    ts_loss = criteria(Xk, X)
                    init_psnr, recon_psnr, delta_psnr, ssim = compute_metrics2chan(Xk, X, X0)
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
        plot_MRI(Xk, X0, X, criteria, args.save_path, -1)
        print(f'{loss_meters[2].avg:.6f} / {loss_meters[4].avg:.6f}')

    else:
        # redefine measurement process, y = Ax + z
        measurement_process = OperatorPlusNoise(forward_op, 0.05)  # same as in training data

        # load Tumor data
        dataloader, ts_length = load_mri_knee_tumor(args)

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
            for X in dataloader:
                bs = X.shape[0]
                X = torch.cat((X, torch.zeros_like(X)), dim=1).to(args.device)
                y = measurement_process(X)
                # reconstruction
                with torch.no_grad():
                    X0 = forward_op.adjoint(y)
                    Xk = net(X0, y)

                    # evaluation
                    ts_loss = criteria(Xk, X)
                    init_psnr, recon_psnr, delta_psnr, ssim = compute_metrics2chan(Xk, X, X0)
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

        title = "train mode: " + str(args.train_mode) + ', eval mode: knee with tumor'
        plot_MRI(Xk, X0, X, criteria, args.save_path, -2)
        print(f'{loss_meters[2].avg:.6f} / {loss_meters[4].avg:.6f}')