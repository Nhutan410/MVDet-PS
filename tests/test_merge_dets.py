"""CPU test of tools/pseudo/merge_dets.py: union of two detectors with cross-detector NMS."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.pseudo.merge_dets import merge_image


def test_merge_image():
    a = (100, 100, 150, 300, 0.9, 0)           # detector 0
    a2 = (102, 98, 151, 303, 0.4, 1)           # same person, detector 1 -> merged, counted once, score 1.0
    b = (400, 100, 450, 300, 0.7, 1)           # only detector 1
    c = (405, 105, 452, 298, 0.6, 1)           # duplicate inside detector 1 -> suppressed, still one source
    out = merge_image([a, a2, b, c], 0.5)
    assert len(out) == 2
    both = [o for o in out if o[5] == 2]
    single = [o for o in out if o[5] == 1]
    assert len(both) == 1 and both[0][4] == 1.0 and both[0][:4] == [100, 100, 150, 300]
    assert len(single) == 1 and abs(single[0][4] - 0.7) < 1e-9
    assert merge_image([], 0.5) == []


if __name__ == '__main__':
    test_merge_image()
    print('PASS test_merge_image')
