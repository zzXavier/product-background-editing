import unittest
from PIL import Image, ImageDraw
from run import metrics


class TestMetrics(unittest.TestCase):
    def test_hard_copy_and_color(self):
        src = Image.new("RGB", (16, 16), "white")
        mask = Image.new("L", src.size)
        ImageDraw.Draw(mask).rectangle((4, 4, 10, 10), fill=255)
        red = Image.composite(src, Image.new("RGB", src.size, "red"), mask)
        m = metrics(src, red, mask, "red")
        self.assertEqual(m["foreground_mae"], 0)
        self.assertEqual(m["target_hue_fraction"], 1)
        self.assertGreater(m["boundary_mae"], 0)
