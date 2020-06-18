#
# Space descriptors define continuous regions
#

from bisect import bisect_left, bisect_right


class RegionOneDimensional:
    def __init__(self, min_value, max_value):
        if not max_value > min_value:
            raise ValueError("Maximum must be > Minimum")
        self.min_value = min_value
        self.max_value = max_value
        self._points = [min_value, max_value]
        self._size = max_value - min_value

    def point_in_region(self, test_point: float) -> bool:
        return not (test_point < self.min_value or test_point > self.max_value)

    def split_location(self, location: float):
        if not self.point_in_region(location):
            raise ValueError("Split location is not in region")
        self._points.insert(bisect_left(self._points, location), location)
        return self

    def split_percent(self, percent: float):
        loc = (self._size * percent) + self.min_value
        return self.split_location(loc)

    def get_size(self):
        return self._size

    def get_num_regions(self):
        return len(self._points) - 1

    def sample_to_region_limits(self, sample: float):
        upper = bisect_left(self._points, sample)
        lower = upper - 1
        return self._points[lower], self._points[upper]

    def sample_to_region_index(self, sample: float):
        return bisect_right(self._points, sample)

    def __len__(self):
        return self.get_size()

    def __getitem__(self, item):
        if item > self.get_num_regions():
            raise IndexError(f"{item} is not a valid region index")
        return self._points[item], self._points[item + 1]
