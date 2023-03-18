from mmdet.apis import init_detector, inference_detector
import mmcv

# モデルの設定ファイルとダウンロードした重みファイルのパスを指定
config_file = '../configs/centernet/centernet_resnet18_140e_coco.py'
checkpoint_file = '../work/weight/centernet_resnet18_140e_coco_20210705_093630-bb5b3bf7.pth'

# モデルを作成
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# テスト
img = 'demo.jpg'
result = inference_detector(model, img)

print(result)

# 結果の保存
model.show_result(img, result, out_file='test_result.jpg')