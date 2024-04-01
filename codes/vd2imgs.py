# coding=utf-8
import cv2
import os
from queue import Queue
import multiprocessing
import fnmatch
import shutil
from skimage.metrics import structural_similarity as ssim


def gen_vdslist(vd_dir, vd_types=['*.mp4']):
    flist = []
    for cur_dir, dirs, files in os.walk(vd_dir):
        fs = []
        for p in vd_types:
            fs.extend(fnmatch.filter(files, p))
        if len(fs)==0:
            continue
        flist.extend([os.path.join(cur_dir, f) for f in fs])
    flist.sort()
    return flist


def get_frames(vfile, img_dir, ssim_threshhold=0.75, num_skip=2, \
                img_ext='.jpg',
                rm_vd=False):
    vs = cv2.VideoCapture(vfile)
    width,height = (int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frames_all = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    # fps = vs.get(cv2.CAP_PROP_FPS)
    vd_name = os.path.basename(vfile).split('.')[0]
    print('processing {}'.format(vd_name))
    _, frame = vs.read()
    num_frame = 0
    cv2.imwrite('{}/{}_{:0>6d}{}'.format(img_dir, vd_name, num_frame, img_ext), frame)
    remove_dup = True if 0<ssim_threshhold<1 else False
    gframe0 = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)/255.0 if remove_dup else None
    while True:
        for i in range(num_skip):
            grabbed = vs.grab()
        if not grabbed:
            break
        grabbed,frame = vs.retrieve()
        if gframe0 is not None:
            gframe1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)/255.0
            # scikit-image 0.22.0 need data_range
            if ssim(gframe0, gframe1, data_range=1) < ssim_threshhold:
                num_frame = num_frame+1
                cv2.imwrite('{}/{}_{:0>6d}{}'.format(img_dir, vd_name, num_frame, img_ext), frame)
                gframe0 = gframe1
        else:
            num_frame = num_frame+1
            cv2.imwrite('{}/{}_{:0>6d}{}'.format(img_dir, vd_name, num_frame, img_ext), frame)
    vs.release()
    if rm_vd:
        os.remove(vfile)


def multiprocess_extract(vd_dir, img_dir, vd_types, ssim_threshhold=0.75, n_work=2):
    pool = multiprocessing.Pool(processes=n_work)
    vd_queue = Queue()
    vd_list = gen_vdslist(vd_dir,vd_types=vd_types)
    print('VD LIST SIZE: ', len(vd_list))
    for vf in vd_list:
        vd_queue.put(os.path.join(vd_dir,vf))
    while not vd_queue.empty():
        vfile = vd_queue.get()
        pool.apply_async(get_frames, (vfile,img_dir,ssim_threshhold,))
    pool.close()
    pool.join()


def group_imgs(source_dir, save_dir, group_sz=20000):
    cnt, num_fd = 0, 0
    imgs = os.listdir(source_dir)
    imgs.sort()
    for img in imgs:
        if cnt%group_sz==0:
            num_fd += 1
            img_dir = os.path.join(save_dir, str(num_fd))
            os.mkdir(img_dir)
        shutil.move(os.path.join(source_dir, img),
                    os.path.join(img_dir, img))
        cnt+=1


if __name__ == '__main__':
    vd_dir = 'dir_of_your_vds'
    img_dir = 'dir_of_your_imgs_want_to_save_in'
    vd_types = ['*.mp4']
    multiprocess_extract(vd_dir, img_dir, vd_types)
    
