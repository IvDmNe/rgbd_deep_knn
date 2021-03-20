import torch


class knn_cuda:
    def __init__(self, datafile=None, save_file='knn_data.pth'):
        self.x_data = None
        self.y_data = None
        self.save_file = save_file

        if datafile:
            data = torch.load(datafile)
            self.x_data = data['x']
            self.y_data = data['y']

    def add_points(self, x, y):
        if self.x_data == None:
            self.x_data = x
            self.y_data = y
        else:
            self.x_data = torch.cat([self.x_data, x])
            self.y_data = torch.cat([self.y_data, y])

        torch.save({'x': self.x_data,
                    'y': self.y_data})

    def classify(self, x):

        dist = torch.norm(self.x_data - x, dim=1, p=None)
        knn = dist.topk(1, largest=False)

        nearest_idx = knn.indices[0]

        cl = self.y_data[nearest_idx]
        return cl