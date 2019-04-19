import numpy as np
import h5py
from acdc.acdc_data import load_and_maybe_process_data
import scipy.ndimage.measurements
from scipy.ndimage.morphology import distance_transform_edt
from skimage.feature import canny
import os


def one_hot_encode(y, num_classes=None):
    if num_classes is None:
        num_classes = y.max() + 1
    y_shape = list(y.shape)
    return np.squeeze(np.eye(num_classes)[y.reshape(-1)]).reshape(y_shape + [-1])


def get_center_of_mass(mask, index):
    # return center of mass
    # output will be in (row, col) order
    center = scipy.ndimage.measurements.center_of_mass(np.ones_like(mask), mask, index)
    center = np.asarray(center)
    return center


def get_distance_transform(img, center):
    img = img.astype(float)
    H, W = img.shape[-2:]
    edges = 1.0 - canny(img.squeeze())
    dt = distance_transform_edt(edges, 0.8)
    dt_original = np.expand_dims(dt.copy(), 0)
    # r, c = int(center[0]), int(center[1])
    # dt[dt > dt[r, c]] = dt[r, c]
    # dt = (dt - np.min(dt)) / (np.max(dt) - np.min(dt))
    dt = np.expand_dims(dt, 0)
    dt_original /= np.sqrt(H ** 2 + W ** 2)
    return dt, dt_original


class DatasetIterator:
    def __init__(self, images, masks, removed_classes=None, center_of_mass_class=3, seed=0, size_limit=10000000,
                 unet=None, remove_nan_centers=True):
        assert len(images) == len(masks)
        self.images = np.asarray(images)
        self.masks = np.asarray(masks)
        self.remove_nan_centers = remove_nan_centers

        # mask the right (and maybe left as well) ventricle as background as we are not working on these
        if removed_classes is not None:
            for c in removed_classes:
                self.masks[self.masks == c] = 0
        # reassign the classes
        classes = np.sort(np.unique(self.masks))
        id_assign = {}
        id_curr = 0
        for c in classes:
            id_assign[c] = id_curr
            id_curr += 1
        for c in classes:
            self.masks[self.masks == c] = id_assign[c]

        # # remove all samples that doesn't have our specific class
        # empty_images = []
        # for j, mask in enumerate(self.masks):
        #     if (mask == id_assign[center_of_mass_class]).astype(int).sum() == 0:
        #         empty_images.append(j)
        # self.images = np.delete(self.images, empty_images, axis=0)
        # self.masks = np.delete(self.masks, empty_images, axis=0)

        # randomize dataset
        self.seed = seed
        self._rng = np.random.RandomState()
        self._seed()
        self.randomize(remove_nan_center=False)
        indices = self._indices_permute[:size_limit]
        self.images = self.images[indices]
        self.masks = self.masks[indices]

        # compute centers of mass
        # center of mass will be in form (row, col)
        self.centers = []
        for mask in self.masks:
            # 3 is the label of the inner circle
            self.centers.append(get_center_of_mass(mask, index=id_assign[center_of_mass_class]))
        self.centers = np.asarray(self.centers)

        # convert masks to one_hot form
        self.onehot_masks = one_hot_encode(self.masks)

        # convert images and onehot_masks to [N, C, H, W] format
        self.images = np.expand_dims(self.images, 1)
        self.onehot_masks = self.onehot_masks.transpose([0, 3, 1, 2])

        # compute distance transform
        # save the original distance_transform as well the modified distance transform
        # this has to be run after removing NaNs from centers
        # run after setting size limit to save computation
        self.dts_modified, self.dts_original = [], []
        for mask, center in zip(self.masks, self.centers):
            dt_modified, dt_original = get_distance_transform(mask, center)
            self.dts_modified.append(dt_modified)
            self.dts_original.append(dt_original)
        self.dts_modified = np.asarray(self.dts_modified)
        self.dts_original = np.asarray(self.dts_original)

        # compute center jitter radius
        # equal distance transform at the center
        H, W = self.images.shape[-2:]
        self.jitter_radius = []
        for center, dt in zip(self.centers, self.dts_original):
            if not np.any(np.isnan(center)):
                self.jitter_radius.append(int(dt[0, int(center[0]), int(center[1])] * np.sqrt(H ** 2 + W ** 2)))
            else:
                self.jitter_radius.append(-1)
        self.jitter_radius = np.asarray(self.jitter_radius)

        self.bboxes = []
        for center in self.centers:
            row, col = center
            bbox = np.asarray([row - 65, row + 65, col - 65, col + 65]).astype(int)
            bbox = np.clip(bbox, 0, 211)
            self.bboxes.append(bbox)
        self.bboxes = np.asarray(self.bboxes)

        # do the inference of UNet here so we won't have to run it again during testing. save time
        if unet is not None:
            import timeit
            import torch
            start = timeit.default_timer()
            self.unet_centers = []
            self.unet_seg = []
            bs = 10
            for j in range(int(np.ceil(len(self.images) / bs))):
                batch = self.images[j * bs:(j + 1) * bs]
                batch = torch.cuda.FloatTensor(batch)
                seg = unet(batch).data.cpu().numpy()
                seg = np.argmax(seg, axis=1)
                self.unet_seg.append(seg)
                seg = (seg > 0).astype(np.float32)
                c = np.asarray([get_center_of_mass(each, 1) for each in seg])
                self.unet_centers.append(c)
            self.unet_centers = np.concatenate(self.unet_centers)
            self.unet_seg = np.concatenate(self.unet_seg)
            stop = timeit.default_timer()
            print("Time takes to compute center using UNet: {:.2f}s".format(stop - start))
            print(np.where(np.isnan(self.unet_centers)))
        else:
            self.unet_centers = None
        self.non_nan_indices = np.unique(np.where(np.invert(np.isnan(self.centers)))[0])
        """
        Additional information:
        ---------------------------
        1 classes:

        max bboxes row/col difference: 
        - train: 59, 62
        -> max radius = 62 / 2 * sqrt(2) = 44

        max jitter radius:
        - train: 21

        =====> max total_radius: 65
        ----------------------------
        2 classes:

        max bboxes row/col difference: 
        - train: 69, 73
        -> max radius = 73 / 2 * sqrt(2) = 52

        max jitter radius:
        - train: 21

        =====> max total_radius: 75
        """
        self.randomize(self.remove_nan_centers)

    def dataset_sz(self):
        if self.remove_nan_centers:
            return len(self.non_nan_indices)
        else:
            return len(self.images)

    def randomize(self, remove_nan_center=True):
        if remove_nan_center:
            _permute = self._rng.permutation(len(self.non_nan_indices))
            self._indices_permute = self.non_nan_indices[_permute]
        else:
            self._indices_permute = self._rng.permutation(len(self.images))
        self.batch_ptr = 0

    def next_batch(self, batch_sz):

        start = self.batch_ptr
        end = self.batch_ptr + batch_sz
        indices = self._indices_permute[start:end]

        images = self.images[indices]
        masks = self.masks[indices]
        one_hot_masks = self.onehot_masks[indices]
        centers = self.centers[indices]
        dts_modified = self.dts_modified[indices]
        dts_original = self.dts_original[indices]
        jitter_radius = self.jitter_radius[indices]
        bboxes = self.bboxes[indices]
        if self.unet_centers is not None:
            unet_centers = self.unet_centers[indices]

        self.batch_ptr += batch_sz
        if self.batch_ptr >= self.dataset_sz():
            extra_sz = self.batch_ptr - self.dataset_sz()
            self.randomize(self.remove_nan_centers)
            if self.unet_centers is not None:
                extra_images, extra_masks, extra_one_hot_masks, extra_centers, extra_dts_modified, extra_dts_original, \
                extra_jitter_radius, extra_bboxes, extra_unet_centers = self.next_batch(extra_sz)
            else:
                extra_images, extra_masks, extra_one_hot_masks, extra_centers, extra_dts_modified, extra_dts_original, \
                extra_jitter_radius, extra_bboxes = self.next_batch(extra_sz)
            images = np.concatenate([images, extra_images], axis=0)
            masks = np.concatenate([masks, extra_masks], axis=0)
            one_hot_masks = np.concatenate([one_hot_masks, extra_one_hot_masks], axis=0)
            centers = np.concatenate([centers, extra_centers], axis=0)
            dts_modified = np.concatenate([dts_modified, extra_dts_modified], axis=0)
            dts_original = np.concatenate([dts_original, extra_dts_original], axis=0)
            jitter_radius = np.concatenate([jitter_radius, extra_jitter_radius], axis=0)
            bboxes = np.concatenate([bboxes, extra_bboxes], axis=0)
            if self.unet_centers is not None:
                unet_centers = np.concatenate([unet_centers, extra_unet_centers])

        if self.unet_centers is not None:
            return images, masks, one_hot_masks, centers, dts_modified, dts_original, jitter_radius, bboxes, \
                   unet_centers
        else:
            return images, masks, one_hot_masks, centers, dts_modified, dts_original, jitter_radius, bboxes

    def _seed(self):
        self._rng.seed(self.seed)


class Dataset:
    def __init__(self,
                 acdc_raw_folder="/home/nhat/ACDC-dataset",
                 preprocessing_folder='preproc_data',
                 train_set_size=1000000, valid_set_size=1000000,
                 num_cls=1, unet=None, remove_nan_center=True):

        data_file_path = os.path.join(preprocessing_folder, "data_2D_size_212_212_res_1.36719_1.36719.hdf5")
        if not os.path.exists(data_file_path):
            load_and_maybe_process_data(acdc_raw_folder, preprocessing_folder, '2D', (212, 212), (1.36719, 1.36719))
        data = h5py.File(data_file_path, "r")

        print("Keys in dataset: ", list(data.keys()))
        self._data = data

        # 1: right ventricle, not segmenting this class
        # 2: left ventricle, outter circle
        # 3: myocardium, inner circle
        # for now only consider class 3, remove both 1 and 2
        if num_cls == 1:
            removed_classes = [1, 2]
        elif num_cls == 2:
            removed_classes = [1]
        else:
            removed_classes = []
        center_of_mass_class = 3
        self.train_set = DatasetIterator(data["images_train"], data["masks_train"],
                                         removed_classes, center_of_mass_class, seed=0, size_limit=train_set_size,
                                         unet=unet, remove_nan_centers=remove_nan_center)
        self.test_set = DatasetIterator(data["images_test"], data["masks_test"],
                                        removed_classes, center_of_mass_class, seed=1, size_limit=valid_set_size,
                                        unet=unet, remove_nan_centers=remove_nan_center)
        # if unet is not None:
        #     assert not np.any(np.isnan(self.test_set.unet_centers))


if __name__ == "__main__":
    d = Dataset()
    print("Finished")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    for j in range(len(d.train_set.images)):
        img = d.train_set.images[j].transpose([2, 0, 1]).squeeze()
        mask = d.train_set.masks[j]
        center = d.train_set.centers[j]
        plt.clf()
        plt.imshow(img, cmap=plt.cm.gray)
        plt.imshow(mask, alpha=1.0)
        plt.scatter(center[1], center[0], color="r", marker="x", s=1)
        plt.savefig("visualize/img{}.jpg".format(j))
        plt.show()
    plt.show()