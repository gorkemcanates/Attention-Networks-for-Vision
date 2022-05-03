import os
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class TensorboardWriter(SummaryWriter):
    def __init__(self,
                 PATH,
                 num_data=48,
                 clear=None):
        super(TensorboardWriter, self).__init__(PATH)
        self.num_data = num_data
        if clear is not None:
            self.clear_Tensorboard(clear)

    def write_results(self,
                      keys: list,
                      results_train,
                      results_test,
                      epoch):
        for metric, index in zip(keys, range(len(results_test))):
            self.add_scalars(metric, {'Training': results_train[index],
                                      'Validation': results_test[index]},
                             epoch + 1)

    def write_images(self,
                     keys: list,
                     data: list,
                     step,
                     condition=False):

        ims, labels, targets = self.get_random_predictions(data=data,
                                                           num_data=self.num_data)
        if condition:
            self.visualize(ims, labels, targets, step)
        # self.add_figure(keys[0],
        #                 fig,
        #                 global_step=step)

    def visualize(self, data, points, targets, step):
        plt.ioff()
        fig = plt.figure(figsize=(16, 12))
        for i in range(self.num_data):
            ax = fig.add_subplot(8, 6, i + 1, xticks=[], yticks=[])
            plt.imshow(data[i].permute(1, 2, 0))
            plt.scatter(points[i][0], points[i][1], s=5, c='green', marker='o')
            plt.scatter(targets[i][0], targets[i][1], s=5, c='red', marker='x')

        if not os.path.exists(self.fig_path):
            os.mkdir(self.fig_path)
        fig.savefig(self.fig_path + str(step) + '.png')
        plt.close(fig)

    @staticmethod
    def visualize_single(data,
                         points,
                         fpath,
                         step):


        plt.ioff()
        fig = plt.figure()
        plt.imshow(data.permute(1, 2, 0))
        plt.scatter(points[0], points[1], s=7, c='green', marker='o')

        if not os.path.exists(fpath):
            os.mkdir(fpath)
        fig.savefig(fpath + str(step) + '.png')
        plt.close(fig)

    def write_hyperparams(self):
        pass

    def write_histogram(self):
        pass

    @staticmethod
    def clear_Tensorboard(file):
        dir = 'runs/' + file
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))

    @staticmethod
    def get_random_predictions(data: list,
                               num_data=32):
        seed = torch.randint(low=0,
                             high=len(data[0]),
                             size=(num_data,))
        random_data = data[0].detach().cpu()[seed]
        random_label = data[1].detach().cpu()[seed]
        random_target = data[2].detach().cpu()[seed]

        return random_data, random_label, random_target
