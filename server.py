from flask import Flask, request
from flask_restful import Resource, Api, reqparse
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
        parser = reqparse.RequestParser()
        parser.add_argument('content')
        parser.add_argument('style_num')
        content = parser.parse_args()['content']
        style_num = parser.parse_args()['style_num']
        styles = []
        style_weights = []
        for i in range(style_num):
            parser.add_argument('style_'+str(i))
            parser.add_argument('style_weight_'+str(i))
            style = parser.parse_args()['style_'+str(i)]
            style_weight = parser.parse_args()['style_weight_'+str(i)]
            styles.append(style)
            style_weights.append(style_weights)

        print(style_num, len(styles), len(style_weights))
        return "forward"



api.add_resource(Forward, '/forward')


if __name__ == '__main__':
    CORS(app)
    app.run(port=5004)
