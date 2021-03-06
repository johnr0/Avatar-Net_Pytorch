from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS

from test import network_test_web

import torch

from network import AvatarNet
from utils import imload, imsave, maskload, result_to_web

app = Flask(__name__)
api = Api(app)

# # set device
device = torch.device('cuda')# if args.gpu_no >= 0 else 'cpu')

# # load check point
check_point = torch.load('./models/check_point.pth')

# # load network
network = AvatarNet([1, 6, 11, 20])
network.load_state_dict(check_point['state_dict'])
network = network.to(device)


class Forward(Resource):
    def get(self):
        return "forward"

    def post(self):
        # do something
        parser = reqparse.RequestParser()
        parser.add_argument('content')
        parser.add_argument('masking')
        parser.add_argument('style_num')
        parser.add_argument('style_content_weight')
        content = parser.parse_args()['content']
        masking = parser.parse_args()['masking']
        style_num = parser.parse_args()['style_num']
        style_content_weight = float(parser.parse_args()['style_content_weight'])
        print('scw', style_content_weight)
        styles = []
        style_weights = []
        style_bboxes = []
        style_scales = []
        for i in range(int(style_num)):
            parser.add_argument('style_'+str(i))
            parser.add_argument('style_weight_'+str(i))
            parser.add_argument('style_bbox_'+str(i))
            parser.add_argument('style_scale_'+str(i))

            style = parser.parse_args()['style_'+str(i)]
            style_weight = parser.parse_args()['style_weight_'+str(i)]
            style_bbox = parser.parse_args()['style_bbox_'+str(i)]
            style_scale = parser.parse_args()['style_scale_'+str(i)]

            styles.append(style)
            style_weights.append(float(style_weight))
            if style_bbox!='undefined':
                style_bbox = style_bbox.split(',')
                style_bbox = [float(i) for i in style_bbox]
            style_bboxes.append(style_bbox)
            style_scales.append(float(style_scale))
            print(style_bbox, type(style_bbox))
        print("til here")
        result = network_test_web(device, network, 600, style_content_weight, 5, 1, style_weights, content, styles, None, False, style_bboxes, style_scales)

        img_result = result_to_web(result, masking)

        # print(style_num, len(styles), len(style_weights), type(content))
        return {'result': img_result}



api.add_resource(Forward, '/forward')


if __name__ == '__main__':
    CORS(app)
    app.run(port=5004)
