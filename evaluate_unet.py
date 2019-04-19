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

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def score_data(input_folder, output_folder, model_path, num_classes=3,
               do_postprocessing=False, gt_exists=True, evaluate_all=False):

    nx, ny = exp_config.image_size[:2]
    batch_size = 1
    num_channels = num_classes + 1

    net = UNet(in_dim=1, out_dim=num_classes + 1).cuda()
    ckpt_path = os.path.join(model_path, 'best_model.pth.tar')
    net.load_state_dict(_pickle.load(open(ckpt_path, 'rb')))

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

                        slice_img = np.squeeze(img[:,:,zz])
                        slice_rescaled = transform.rescale(slice_img,
                                                           scale_vector,
                                                           order=1,
                                                           preserve_range=True,
                                                           multichannel=False,
                                                           mode='constant')

                        x, y = slice_rescaled.shape

                        x_s = (x - nx) // 2
                        y_s = (y - ny) // 2
                        x_c = (nx - x) // 2
                        y_c = (ny - y) // 2

                        # Crop section of image for prediction
                        if x > nx and y > ny:
                            slice_cropped = slice_rescaled[x_s:x_s+nx, y_s:y_s+ny]
                        else:
                            slice_cropped = np.zeros((nx,ny))
                            if x <= nx and y > ny:
                                slice_cropped[x_c:x_c+ x, :] = slice_rescaled[:,y_s:y_s + ny]
                            elif x > nx and y <= ny:
                                slice_cropped[:, y_c:y_c + y] = slice_rescaled[x_s:x_s + nx, :]
                            else:
                                slice_cropped[x_c:x_c+x, y_c:y_c + y] = slice_rescaled[:, :]


                        # GET PREDICTION
                        network_input = np.float32(np.tile(np.reshape(slice_cropped, (nx, ny, 1)), (batch_size, 1, 1, 1)))
                        network_input = np.transpose(network_input, [0, 3, 1, 2])
                        network_input = torch.cuda.FloatTensor(network_input)
                        with torch.no_grad():
                            net.eval()
                            logits_out = net(network_input)
                            softmax_out = F.softmax(logits_out, dim=1)
                            # mask_out = torch.argmax(logits_out, dim=1)
                            softmax_out = softmax_out.data.cpu().numpy()
                            softmax_out = np.transpose(softmax_out, [0, 2, 3, 1])
                        # prediction_cropped = np.squeeze(softmax_out[0,...])
                        prediction_cropped = np.squeeze(softmax_out)

                        # ASSEMBLE BACK THE SLICES
                        slice_predictions = np.zeros((x,y,num_channels))
                        # insert cropped region into original image again
                        if x > nx and y > ny:
                            slice_predictions[x_s:x_s+nx, y_s:y_s+ny,:] = prediction_cropped
                        else:
                            if x <= nx and y > ny:
                                slice_predictions[:, y_s:y_s+ny,:] = prediction_cropped[x_c:x_c+ x, :,:]
                            elif x > nx and y <= ny:
                                slice_predictions[x_s:x_s + nx, :,:] = prediction_cropped[:, y_c:y_c + y,:]
                            else:
                                slice_predictions[:, :,:] = prediction_cropped[x_c:x_c+ x, y_c:y_c + y,:]

                        # RESCALING ON THE LOGITS
                        if gt_exists:
                            prediction = transform.resize(slice_predictions,
                                                          (mask.shape[0], mask.shape[1], num_channels),
                                                          order=1,
                                                          preserve_range=True,
                                                          mode='constant')
                        else:  # This can occasionally lead to wrong volume size, therefore if gt_exists
                               # we use the gt mask size for resizing.
                            prediction = transform.rescale(slice_predictions,
                                                           (1.0/scale_vector[0], 1.0/scale_vector[1], 1),
                                                           order=1,
                                                           preserve_range=True,
                                                           multichannel=False,
                                                           mode='constant')

                        # prediction = transform.resize(slice_predictions,
                        #                               (mask.shape[0], mask.shape[1], num_channels),
                        #                               order=1,
                        #                               preserve_range=True,
                        #                               mode='constant')

                        prediction = np.uint8(np.argmax(prediction, axis=-1))
                        if num_classes == 1:
                            prediction[prediction == 1] = 3
                        elif num_classes == 2:
                            prediction[prediction == 2] = 3
                            prediction[prediction == 1] = 2
                        predictions.append(prediction)

                    prediction_arr = np.transpose(np.asarray(predictions, dtype=np.uint8), (1,2,0))

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
                        gt_file_name = os.path.join(output_folder, 'ground_truth', 'patient' + patient_id + frame_suffix + '.nii.gz')
                        logging.info('saving to: %s' % gt_file_name)
                        utils.save_nii(gt_file_name, mask, out_affine, out_header)

                        # Save difference mask between predictions and ground truth
                        difference_mask = np.where(np.abs(prediction_arr-mask) > 0, [1], [0])
                        difference_mask = np.asarray(difference_mask, dtype=np.uint8)
                        diff_file_name = os.path.join(output_folder,
                                                      'difference',
                                                      'patient' + patient_id + frame_suffix + '.nii.gz')
                        logging.info('saving to: %s' % diff_file_name)
                        utils.save_nii(diff_file_name, difference_mask, out_affine, out_header)

    logging.info('Average time per volume: %f' % (total_time/total_volumes))

    return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Script to evaluate a neural network model on the ACDC challenge data")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment folder (assuming you are in the working directory)")
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('-t', '--evaluate_test_set', action='store_true')
    parser.add_argument('-a', '--evaluate_all', action='store_true')
    parser.add_argument('-i', '--iter', type=int, help='which iteration to use')
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
                                num_classes=args.num_classes,
                                do_postprocessing=True,
                                gt_exists=(not evaluate_test_set),
                                evaluate_all=evaluate_all)

    metrics_acdc.main(path_gt, path_pred, path_eval)

