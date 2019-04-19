# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import torch
import os
import glob
import numpy as np
import logging

import argparse
import metrics_acdc
import time
from skimage import transform

from acdc import image_utils, utils
import sys_config
from network import UNet
import _pickle
import sys_config as exp_config
import torch.nn.functional as F
from snake.snake import SnakePytorch
from scipy.ndimage.measurements import center_of_mass

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
from train_edpcnn import get_star_pattern_values, smooth_ind, star_pattern_ind_to_mask


def get_slice(img, nx, ny):
    x, y = img.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    # Crop section of image for prediction
    if x > nx and y > ny:
        cropped_img = img[x_s:x_s + nx, y_s:y_s + ny]
    else:
        cropped_img = np.zeros((nx, ny))
        if x <= nx and y > ny:
            cropped_img[x_c:x_c + x, :] = img[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            cropped_img[:, y_c:y_c + y] = img[x_s:x_s + nx, :]
        else:
            cropped_img[x_c:x_c + x, y_c:y_c + y] = img[:, :]
    return cropped_img, x_s, y_s, x_c, y_c


def score_data(input_folder, output_folder, model_path, args,
               do_postprocessing=False, gt_exists=True, evaluate_all=False):
    num_classes = args.num_cls
    nx, ny = exp_config.image_size[:2]
    batch_size = 1
    num_channels = num_classes + 1

    net = UNet(in_dim=1, out_dim=2).cuda()
    ckpt_path = os.path.join(model_path, 'best_model.pth.tar')
    net.load_state_dict(_pickle.load(open(ckpt_path, 'rb')))
    if args.unet_ckpt:
        pretrained_unet = UNet(in_dim=1, out_dim=4).cuda()
        pretrained_unet.load_state_dict(_pickle.load(open(args.unet_ckpt, 'rb')))

    snake = SnakePytorch(args.delta, 1, args.num_lines, args.radius)

    evaluate_test_set = not gt_exists

    total_time = 0
    total_volumes = 0

    for folder in os.listdir(input_folder):

        folder_path = os.path.join(input_folder, folder)

        if os.path.isdir(folder_path):

            if evaluate_test_set or evaluate_all:
                train_test = 'test'  # always test
            else:
                train_test = 'test' if (int(folder[-3:]) % 5 == 0) else 'train'

            if train_test == 'test':

                infos = {}
                for line in open(os.path.join(folder_path, 'Info.cfg')):
                    label, value = line.split(':')
                    infos[label] = value.rstrip('\n').lstrip(' ')

                patient_id = folder.lstrip('patient')
                ED_frame = int(infos['ED'])
                ES_frame = int(infos['ES'])

                for file in glob.glob(os.path.join(folder_path, 'patient???_frame??.nii.gz')):

                    logging.info(' ----- Doing image: -------------------------')
                    logging.info('Doing: %s' % file)
                    logging.info(' --------------------------------------------')

                    file_base = file.split('.nii.gz')[0]

                    frame = int(file_base.split('frame')[-1])
                    img_dat = utils.load_nii(file)
                    img = img_dat[0].copy()
                    img = image_utils.normalise_image(img)

                    if gt_exists:
                        file_mask = file_base + '_gt.nii.gz'
                        mask_dat = utils.load_nii(file_mask)
                        mask = mask_dat[0]

                    start_time = time.time()

                    pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])
                    scale_vector = (pixel_size[0] / exp_config.target_resolution[0],
                                    pixel_size[1] / exp_config.target_resolution[1])

                    predictions = []

                    for zz in range(img.shape[2]):

                        slice_img = np.squeeze(img[:, :, zz])
                        slice_rescaled = transform.rescale(slice_img,
                                                           scale_vector,
                                                           order=1,
                                                           preserve_range=True,
                                                           multichannel=False,
                                                           mode='constant')

                        x, y = slice_rescaled.shape
                        slice_cropped, x_s, y_s, x_c, y_c = get_slice(slice_rescaled, nx, ny)
                        # GET PREDICTION
                        network_input = np.float32(
                            np.tile(np.reshape(slice_cropped, (nx, ny, 1)), (batch_size, 1, 1, 1)))
                        network_input = np.transpose(network_input, [0, 3, 1, 2])
                        network_input = torch.cuda.FloatTensor(network_input)
                        with torch.no_grad():
                            net.eval()
                            logit = net(network_input)
                            logit = torch.argmax(logit, 1)

                        # get the center
                        if args.unet_ckpt != '':
                            unet_mask = torch.argmax(pretrained_unet(network_input), dim=1).data.cpu().numpy()[0]
                        else:
                            assert gt_exists
                            mask_copy = mask[:, :, zz].copy()
                            unet_mask = get_slice(mask_copy, nx, ny)[0]
                        unet_mask = image_utils.keep_largest_connected_components(unet_mask)
                        from data_iterator import get_center_of_mass
                        if num_classes == 2:
                            lv_center = get_center_of_mass(unet_mask, [3])
                        else:
                            lv_center = get_center_of_mass(unet_mask, [3])
                        lv_center = np.asarray(lv_center)

                        lv_logit, _, _ = get_star_pattern_values(logit, None, lv_center, args.num_lines,
                                                                 args.radius + 1)
                        lv_gs = lv_logit[:, :, 1:] - lv_logit[:, :, :-1]  # compute the gradient
                        # run DP algo
                        # can only put batch with fixed shape into the snake algorithm
                        lv_ind = snake(lv_gs).data.cpu().numpy()
                        lv_ind = np.expand_dims(smooth_ind(lv_ind.squeeze(-1), args.smoothing_window), -1)
                        lv_mask = star_pattern_ind_to_mask(lv_ind, lv_center, nx, ny, args.num_lines, args.radius)
                        if np.isnan(lv_center[0, 0]):
                            lv_mask *= 0

                        if num_classes == 1:
                            pred_mask = lv_mask * 3

                        prediction_cropped = pred_mask.squeeze()
                        # ASSEMBLE BACK THE SLICES
                        prediction = np.zeros((x, y))
                        # insert cropped region into original image again
                        if x > nx and y > ny:
                            prediction[x_s:x_s + nx, y_s:y_s + ny] = prediction_cropped
                        else:
                            if x <= nx and y > ny:
                                prediction[:, y_s:y_s + ny] = prediction_cropped[x_c:x_c + x, :]
                            elif x > nx and y <= ny:
                                prediction[x_s:x_s + nx, :] = prediction_cropped[:, y_c:y_c + y]
                            else:
                                prediction[:, :] = prediction_cropped[x_c:x_c + x, y_c:y_c + y]

                        # RESCALING ON THE LOGITS
                        if gt_exists:
                            prediction = transform.resize(prediction,
                                                          (mask.shape[0], mask.shape[1]),
                                                          order=0,
                                                          preserve_range=True,
                                                          mode='constant')
                        else:  # This can occasionally lead to wrong volume size, therefore if gt_exists
                            # we use the gt mask size for resizing.
                            prediction = transform.rescale(prediction,
                                                           (1.0 / scale_vector[0], 1.0 / scale_vector[1]),
                                                           order=0,
                                                           preserve_range=True,
                                                           multichannel=False,
                                                           mode='constant')

                        # prediction = np.uint8(np.argmax(prediction, axis=-1))
                        prediction = np.uint8(prediction)
                        predictions.append(prediction)

                        gt_binary = (mask[..., zz] == 3) * 1
                        pred_binary = (prediction == 3) * 1
                        from medpy.metric.binary import hd, dc, assd
                        lv_center = lv_center[0]
                        # i=0;  plt.imshow(network_input[0, 0]); plt.plot(lv_center[1], lv_center[0], 'ro'); plt.show(); plt.imshow(unet_mask); plt.plot(lv_center[1], lv_center[0], 'ro'); plt.show(); plt.imshow(logit[0, 0]); plt.plot(lv_center[1], lv_center[0], 'ro'); plt.show(); plt.imshow(lv_logit[0]); plt.show();  plt.imshow(lv_gs[0]); plt.show(); plt.imshow(prediction_cropped); plt.plot(lv_center[1], lv_center[0], 'r.'); plt.show();

                    prediction_arr = np.transpose(np.asarray(predictions, dtype=np.uint8), (1, 2, 0))

                    # This is the same for 2D and 3D again
                    if do_postprocessing:
                        assert num_classes == 1
                        from skimage.measure import regionprops
                        lv_obj = (mask_dat[0] == 3).astype(np.uint8)
                        prop = regionprops(lv_obj)
                        assert len(prop) == 1
                        prop = prop[0]
                        centroid = prop.centroid
                        centroid = (int(centroid[0]), int(centroid[1]), int(centroid[2]))
                        prediction_arr = image_utils.keep_largest_connected_components(prediction_arr, centroid)


                    elapsed_time = time.time() - start_time
                    total_time += elapsed_time
                    total_volumes += 1

                    logging.info('Evaluation of volume took %f secs.' % elapsed_time)

                    if frame == ED_frame:
                        frame_suffix = '_ED'
                    elif frame == ES_frame:
                        frame_suffix = '_ES'
                    else:
                        raise ValueError('Frame doesnt correspond to ED or ES. frame = %d, ED = %d, ES = %d' %
                                         (frame, ED_frame, ES_frame))

                    # Save prediced mask
                    out_file_name = os.path.join(output_folder, 'prediction',
                                                 'patient' + patient_id + frame_suffix + '.nii.gz')
                    if gt_exists:
                        out_affine = mask_dat[1]
                        out_header = mask_dat[2]
                    else:
                        out_affine = img_dat[1]
                        out_header = img_dat[2]

                    logging.info('saving to: %s' % out_file_name)
                    utils.save_nii(out_file_name, prediction_arr, out_affine, out_header)

                    # Save image data to the same folder for convenience
                    image_file_name = os.path.join(output_folder, 'image',
                                                   'patient' + patient_id + frame_suffix + '.nii.gz')
                    logging.info('saving to: %s' % image_file_name)
                    utils.save_nii(image_file_name, img_dat[0], out_affine, out_header)

                    if gt_exists:
                        # Save GT image
                        gt_file_name = os.path.join(output_folder, 'ground_truth',
                                                    'patient' + patient_id + frame_suffix + '.nii.gz')
                        logging.info('saving to: %s' % gt_file_name)
                        utils.save_nii(gt_file_name, mask, out_affine, out_header)

                        # Save difference mask between predictions and ground truth
                        difference_mask = np.where(np.abs(prediction_arr - mask) > 0, [1], [0])
                        difference_mask = np.asarray(difference_mask, dtype=np.uint8)
                        diff_file_name = os.path.join(output_folder,
                                                      'difference',
                                                      'patient' + patient_id + frame_suffix + '.nii.gz')
                        logging.info('saving to: %s' % diff_file_name)
                        utils.save_nii(diff_file_name, difference_mask, out_affine, out_header)

    logging.info('Average time per volume: %f' % (total_time / total_volumes))

    return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Script to evaluate a neural network model on the ACDC challenge data")
    parser.add_argument("EXP_PATH", type=str,
                        help="Path to experiment folder (assuming you are in the working directory)")
    parser.add_argument('--num_class', type=int, default=3)
    parser.add_argument('-t', '--evaluate_test_set', action='store_true')
    parser.add_argument('-a', '--evaluate_all', action='store_true')
    parser.add_argument('-i', '--iter', type=int, help='which iteration to use')
    parser.add_argument('--num_cls', default=1, type=int)
    parser.add_argument('--unet_ckpt', default='', type=str)
    parser.add_argument('--delta', default=1, type=int)
    parser.add_argument('--num_lines', default=25, type=int)
    parser.add_argument('--radius', default=65, type=int)
    parser.add_argument('--smoothing_window', default=1, type=int)

    args = parser.parse_args()

    evaluate_test_set = args.evaluate_test_set
    evaluate_all = args.evaluate_all

    if evaluate_test_set and evaluate_all:
        raise ValueError('evaluate_all and evaluate_test_set cannot be chosen together!')

    use_iter = args.iter
    if use_iter:
        logging.info('Using iteration: %d' % use_iter)

    base_path = sys_config.project_root
    model_path = os.path.join(base_path, args.EXP_PATH)

    if evaluate_test_set:
        logging.warning('EVALUATING ON TEST SET')
        input_path = sys_config.test_data_root
        output_path = os.path.join(model_path, 'predictions_testset')
    elif evaluate_all:
        logging.warning('EVALUATING ON ALL TRAINING DATA')
        input_path = sys_config.data_root
        output_path = os.path.join(model_path, 'predictions_alltrain')
    else:
        logging.warning('EVALUATING ON VALIDATION SET')
        input_path = sys_config.data_root
        output_path = os.path.join(model_path, 'predictions')

    path_pred = os.path.join(output_path, 'prediction')
    path_image = os.path.join(output_path, 'image')
    utils.makefolder(path_pred)
    utils.makefolder(path_image)

    path_gt = os.path.join(output_path, 'ground_truth')
    path_diff = os.path.join(output_path, 'difference')
    path_eval = os.path.join(output_path, 'eval')

    utils.makefolder(path_diff)
    utils.makefolder(path_gt)

    init_iteration = score_data(input_path,
                                output_path,
                                model_path,
                                args,
                                do_postprocessing=True,
                                gt_exists=(not evaluate_test_set),
                                evaluate_all=evaluate_all)

    metrics_acdc.main(path_gt, path_pred, path_eval)
