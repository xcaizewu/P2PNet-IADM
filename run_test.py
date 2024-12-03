import json
import mmcv
import glob
from engine import *
import os, onnxruntime
import warnings
warnings.filterwarnings('ignore')


class XpuAlgorithm:
    def __init__(self):
        self.h = None
        self.w = None
        self.threshold = 0.5
        self.input_data_dict = {}

        self.weight = 'xxx.onnx'
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        options = onnxruntime.SessionOptions()
        options.log_severity_level = 4
        self.session = onnxruntime.InferenceSession(self.weight, sess_options=options, providers=self.providers)

    def pre_process(self, RGBT):
        rgb = RGBT[0] / 255
        t = RGBT[1] / 255
        self.input_data_dict = {}
        input_shape = (1, 3, 480, 640)
        self.h, self.w, _ = rgb.shape
        rgb = mmcv.imresize(rgb, input_shape[2:][::-1])
        mean = np.array([0.407, 0.389, 0.396], dtype=np.float32)
        std = np.array([0.241, 0.246, 0.242], dtype=np.float32)
        rgb = mmcv.imnormalize(rgb, mean, std, to_rgb=False)

        rgb_nhwc = np.expand_dims(rgb, axis=0)
        rgb_nchw = rgb_nhwc.transpose((0, 3, 1, 2))

        t = mmcv.imresize(t, input_shape[2:][::-1])
        mean = np.array([0.492, 0.168, 0.430], dtype=np.float32)
        std = np.array([0.317, 0.174, 0.191], dtype=np.float32)
        t = mmcv.imnormalize(t, mean, std, to_rgb=False)

        t_nhwc = np.expand_dims(t, axis=0)
        t_nchw = t_nhwc.transpose((0, 3, 1, 2))
        self.input_data_dict['input1'] = rgb_nchw
        self.input_data_dict['input2'] = t_nchw

    def inference(self, RGBT):
        self.pre_process(RGBT)
        input_names = [self.session.get_inputs()[0].name, self.session.get_inputs()[1].name]
        output_names = [self.session.get_outputs()[0].name, self.session.get_outputs()[1].name]
        input_data = {
            input_names[0]: self.input_data_dict['input1'],
            input_names[1]: self.input_data_dict['input2'],
        }
        out = self.session.run(output_names, input_data)
        num, points = self.post_process(out)

        return num, points

    def post_process(self, out):
        pred_logits = out[0]
        pred_points = out[1]

        softmax_pred_logits_np = self.softmax(np.array(pred_logits))
        selected_elements = softmax_pred_logits_np[..., 1]

        outputs_scores = selected_elements[0]
        outputs_points = pred_points[0]

        points = outputs_points[outputs_scores > self.threshold]
        predict_cnt = int((outputs_scores > self.threshold).sum())

        return predict_cnt, points

    def softmax(self, x):

        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)


if __name__ == '__main__':
    val_path = None
    test_path = '/home/user/RGBTCrowdCounting/test/'
    model = XpuAlgorithm()

    gt_list = sorted(glob.glob(os.path.join(test_path, '*.json')))

    maes = []
    mses = []
    mean_time = 0
    for gt_path in gt_list:
        rgb_path = gt_path.replace('GT', 'RGB').replace('json', 'jpg')
        t_path = gt_path.replace('GT', 'T').replace('json', 'jpg')

        with open(gt_path, 'r') as file:
            data = json.load(file)

        gt_cnt = len(data['points'])
        rgb = cv2.imread(rgb_path)
        t = cv2.imread(t_path)

        start_time = time.time()
        predict_cnt, points = model.inference([rgb, t])
        cost = time.time() - start_time
        mean_time += cost

        print(f'gt: {gt_cnt}, predict: {predict_cnt}')
        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
        maes.append(float(mae))
        mses.append(float(mse))

        for p in points:
            img_to_draw = cv2.circle(rgb, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)
        cv2.putText(rgb, f'GT: {gt_cnt}  Pre:{predict_cnt}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0),
                        1)
        cv2.imwrite(os.path.join('./save', 'pred{}.jpg'.format(gt_path.split('/')[-1].split('.')[0])), rgb)

    mae = np.mean(maes)
    mse = np.sqrt(np.mean(mses))

    print('MAE: {}, MSE: {}'.format(mae, mse))
    print(f'total pic is {len(gt_list)}, mean cost time is {mean_time / len(gt_list)}')
