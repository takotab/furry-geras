import motion
import os

image_files = ["data/roos2.jpg", "data/tako.jpg", "data/roos.jpg", "data/tako2.jpg"]
video_file = os.path.join(os.getcwd(), "test/assets/divera_trend_small.m4v")
video_file = os.path.join(os.getcwd(), "coco/divera_trend.mp4")


def test_get_pose():
    vid2pose = motion.Video2Pose(device="cuda:0")
    array, pred = vid2pose.make_posevid(video_file, False)
    vid2pose._make_plot_pose(array[:2], pred[:2])
    assert True

