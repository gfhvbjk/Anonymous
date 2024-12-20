'''
Main script with trainers
'''
import numpy as np
from reinforcement_learning.bandits import BNN_Bandit, Greedy_Bandit
from regression.reg_task import BNN_Regression, MLP_Regression, MCDropout_Regression
from classification.class_task import BNN_Classification, MLP_Classification, MCDropout_Classification
from tqdm import tqdm
from config import *
from utils.data_utils import *
from utils.plot_utils import *
import time
def reg_trainer():
    ''' Regression Task Trainer '''
    config = RegConfig
    X, Y = create_data_reg(train_size=config.train_size, gap=config.regression_clusters)
    train_ds = PrepareData(X, Y)
    train_ds = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    params = {
        'lr': config.lr,
        'hidden_units': config.hidden_units,
        'mode': config.mode,
        'batch_size': config.batch_size,
        'num_batches': len(train_ds),
        'x_shape': X.shape[1],
        'y_shape': Y.shape[1],
        'train_samples': config.train_samples,
        'test_samples': config.test_samples,
        'noise_tolerance': config.noise_tolerance,
        'mixture_prior': config.mixture_prior,
        'mu_init': config.mu_init,
        'rho_init': config.rho_init,
        'prior_init': config.prior_init,
        'save_dir': config.save_dir,
    }

    models = {
        'bnn_reg': BNN_Regression('bnn_regression', {**params, 'local_reparam': False}),
        'bnn_reg_lr' : BNN_Regression('bnn_regression_lr', {**params, 'local_reparam': True}),
        'mlp_reg': MLP_Regression('mlp_regression', {**params, 'local_reparam': False}),
        'mcdropout_reg': MCDropout_Regression('mcdropout_regression', {**params, 'local_reparam': False}),
    }

    epochs = config.epochs
    print(f"Initialising training on {DEVICE}...")

    # training loop
    for epoch in tqdm(range(epochs)):
        for m_name, model in models.items():
            model.train_step(train_ds)
            model.log_progress(epoch)
            model.scheduler.step()
            # save best model
            if model.epoch_loss < model.best_loss:
                model.best_loss = model.epoch_loss
                torch.save(model.net.state_dict(), model.save_model_path)

    # evaluate
    print("Evaluating and generating plots...")
    x_test = torch.linspace(-2., 2, config.num_test_points).reshape(-1, 1)
    for m_name, model in models.items():
        model.net.load_state_dict(torch.load(model.save_model_path, map_location=torch.device(DEVICE)))
        y_test = model.evaluate(x_test)
        if m_name == 'mlp_reg':
            create_regression_plot(x_test.cpu().numpy(), y_test.reshape(1, -1), train_ds, m_name)
        else:
            create_regression_plot(x_test.cpu().numpy(), y_test, train_ds, m_name)


def rl_trainer():
    ''' RL Bandit Task Trainer'''
    config = RLConfig
    X, Y = read_data_rl(config.data_dir)

    params = {
        'buffer_size': config.buffer_size,
        'batch_size': config.batch_size,
        'num_batches': config.num_batches,
        'lr': config.lr,
        'hidden_units': config.hidden_units,
        'mode': config.mode,
        'mixture_prior': config.mixture_prior,
        'mu_init': config.mu_init,
        'rho_init': config.rho_init,
        'prior_init': config.prior_init
    }

    bandits = {
        'bnn_bandit': BNN_Bandit('bnn_bandit', {**params, 'n_samples':2, 'epsilon':0}, X, Y),
        'greedy_bandit': Greedy_Bandit('greedy_bandit', {**params, 'n_samples':1, 'epsilon':0}, X, Y),
        '0.01_greedy_bandit': Greedy_Bandit('0.01_greedy_bandit', {**params, 'n_samples':1, 'epsilon':0.01}, X, Y),
        '0.05_greedy_bandit': Greedy_Bandit('0.05_greedy_bandit', {**params, 'n_samples':1, 'epsilon':0.05}, X, Y)
    }

    training_steps = config.training_steps
    print(f"Initialising training on {DEVICE}...")
    training_data_len = len(X)
    for step in tqdm(range(training_steps)):
        mushroom = np.random.randint(training_data_len)
        for _, bandit in bandits.items():
            bandit.update(mushroom)
            bandit.scheduler.step()
            if (step+1)%100 == 0:
                bandit.log_progress(step)


def class_trainer():
    ''' MNIST classification Task Trainer'''
    config = ClassConfig

    train_ds = create_data_class(train=True, batch_size=config.batch_size, shuffle=True)
    test_ds = create_data_class(train=False, batch_size=config.batch_size, shuffle=False)

    params = {
        'lr': config.lr,
        'hidden_units': config.hidden_units,
        'mode': config.mode,
        'batch_size': config.batch_size,
        'epochs': config.epochs,
        'x_shape': config.x_shape,
        'classes': config.classes,
        'num_batches': len(train_ds),
        'train_samples': config.train_samples,
        'test_samples': config.test_samples,
        'mu_init': config.mu_init,
        'rho_init': config.rho_init,
        'prior_init': config.prior_init,
        'mixture_prior':config.mixture_prior,
        'save_dir': config.save_dir,
    }

    models = {
        # 'bnn_class': BNN_Classification('bnn_classification', {**params, 'local_reparam': False, 'dropout': False}),
        'bnn_class_lr': BNN_Classification('bnn_classification_lr', {**params, 'local_reparam': True, 'dropout': False})
        # 'mlp_class': MLP_Classification('mlp_classification', {**params, 'dropout': False}),
        # 'dropout_class': MLP_Classification('dropout_classification', {**params, 'dropout': True}),
        # 'mcdropout_class': MCDropout_Classification('mcdropout_classification', {**params, 'dropout': True}),
        }
    
    epochs = config.epochs
    print(f"Initialising training on {DEVICE}...")
    # 初始化早停相关变量
    no_improve_count = 0
    acc_above_82 = False
    patience = 5  # 连续没有提升的 epoch 数
    time_start = time.time()
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        # for _, model in models.items():
        models['bnn_class_lr'].train_step(train_ds)
        models['bnn_class_lr'].evaluate(test_ds)
        # models['bnn_class_lr'].log_progress(epoch)
        models['bnn_class_lr'].scheduler.step()

        current_acc = models['bnn_class_lr'].acc
        best_acc = models['bnn_class_lr'].best_acc
        # print(current_acc)
        print(f'Current Accuracy achieved: {current_acc:.2f}%')
        if models['bnn_class_lr'].acc > models['bnn_class_lr'].best_acc:
            models['bnn_class_lr'].best_acc = models['bnn_class_lr'].acc
            torch.save(models['bnn_class_lr'].net.state_dict(), models['bnn_class_lr'].save_model_path)
            print(f'Accuracy improved to {current_acc:.2f}%. Saving model.')

        #     # 重置没有提升的计数
        #     no_improve_count = 0
        #
        #     # 如果准确率超过 82%，标记 acc_above_82
        #     if current_acc > 0.82:
        #         acc_above_82 = True
        # else:
        #     # 如果没有提升，且已经超过 82%，增加计数
        #     if acc_above_82:
        #         no_improve_count += 1
        #         print(f'No improvement for {no_improve_count} consecutive epochs.')
        #
        #         # 如果连续 5 次没有提升，且准确率已经超过 82%，则终止训练
        #         if no_improve_count >= patience:
        #             print(
        #                 f'Early stopping triggered after {no_improve_count} epochs without improvement and accuracy > 82%.')
        #             break

            # 可选：打印当前最佳准确率
        print(f'Best Accuracy: {models["bnn_class_lr"].best_acc:.2f}%')

        time_end = time.time()
        print(f'Training completed in {(time_end - time_start) / 60:.2f} minutes.')
        print(f'Best Accuracy achieved: {models["bnn_class_lr"].best_acc:.2f}%')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', metavar='m', default='classification', type=str)
    args = parser.parse_args()

    if args.model == 'regression':
        reg_trainer()
    elif args.model == 'classification':
        class_trainer()
    else:
        rl_trainer()