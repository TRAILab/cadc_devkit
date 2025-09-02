from PIL import Image

# base_path = '/home/trail/workspace/cadc_devkit/viz_det_nolim_scorethres2/val_set/cam0/2018_03_06_0001'
# base_path = '/home/trail/workspace/cadc_devkit/viz_det_nolim_scorethres2/val_set/cam0/2019_02_27_0076'
# filenames = [base_path + '_' + str(i) + '.png' for i in range(100)]
base_path = '/home/trail/workspace/cadc_devkit/out/mot/viz/2019_02_27_0076_lidar'
filenames = [base_path + '_' + str(i+1).zfill(3) + '.png' for i in range(100)]
base_path = '/home/trail/workspace/cadc_devkit/out/mot/viz/2019_02_27_0076_global'
filenames = [base_path + '_' + str(i+1).zfill(3) + '.png' for i in range(10)]

gif_path = base_path + '.gif'
images = [Image.open(filename) for filename in filenames]
images[0].save(gif_path, save_all=True, append_images=images[1:], optimize=False, duration=300, loop=0)
print('GIF saved to', gif_path)