import time

from . import get_video_array, get_pose, plot_pose, make_video, filename_maker, config


def make_posevid(video_dir):
    start_time = time.time()
    name = filename_maker(video_dir)
    array = get_video_array(video_dir, config.get_image_size("both"))
    
    print("geting video", time.time() - start_time)
    pred = get_pose(array, device="cpu")
    print("geting get_pose", time.time() - start_time)
    pose_imgs_dir = plot_pose(array, pred, name=name)
    print("geting plot_pose", time.time() - start_time)
    posevid_dir = make_video(pose_imgs_dir, name=name)
    print("saved at ", posevid_dir)
    return posevid_dir

