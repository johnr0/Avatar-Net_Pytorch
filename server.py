from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS

from test import network_test_web

import torch

from network import AvatarNet
from utils import imload, imsave, maskload

app = Flask(__name__)
api = Api(app)

# # set device
# device = torch.device('cuda' if args.gpu_no >= 0 else 'cpu')

# # load check point
# check_point = torch.load(args.check_point)

# # load network
# network = AvatarNet(args.layers)
# network.load_state_dict(check_point['state_dict'])
# network = network.to(device)


class Forward(Resource):
    def get(self):
        return "forward"
        
    def post(self):
        # do something
        return "forward"



api.add_resource(Forward, '/forward')


if __name__ == '__main__':
    CORS(app)
    app.run(port=5004)
