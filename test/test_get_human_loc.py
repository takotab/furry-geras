from motion.detect_human import get_human_loc

f = "/home/tako/devtools/furry-geras/test/data/000000050638.jpg"
size = 393, 640
results = [344.43, 65.35, 127.94, 316.04999999999995]


def test_human_loc():
    bbox = get_human_loc(f, size=size)
