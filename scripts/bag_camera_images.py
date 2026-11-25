"""Read camera images in timestamp order with bounded memory for pose playback."""
import struct
from pathlib import Path

import cv2
import numpy as np
from rgb_core import decode_mosaic, split_cameras


class BagCameraImages:
    def __init__(self, bag, cameras, quadrants, recorded_times=None):
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import Image
        self.api = rosbag2_py
        self.deserialize = deserialize_message
        self.image_type = Image
        self.bag = Path(bag)
        self.cameras, self.quadrants = cameras, quadrants
        self.reader = None
        self.previous_target = -np.inf
        self.pending = None
        self.last = None
        self.recorded_times = recorded_times
        self.previous_index = -1

    def _reset(self):
        self.reader = self.api.SequentialReader()
        self.reader.open(self.api.StorageOptions(uri=str(self.bag), storage_id='mcap'), self.api.ConverterOptions('', ''))
        self.reader.set_filter(self.api.StorageFilter(topics=['/Data_Cam_Global']))
        self.pending = None
        self.last = None

    def at(self, stamp):
        if self.reader is None or stamp < self.previous_target:
            self._reset()
            self.previous_index = -1
        if self.recorded_times is not None:
            index = int(np.argmin(abs(self.recorded_times[:, 0]-stamp)))
            if (self.previous_index < 0 and index > 0) or index > self.previous_index+1:
                # Jump to the recorded bag time rather than reading every skipped image.
                self.reader.seek(int(self.recorded_times[index, 1]*1e9)-1000000)
                self.pending = self.last = None
            self.previous_index = index
        self.previous_target = stamp
        while True:
            if self.pending is None:
                if not self.reader.has_next(): break
                _, buf, _ = self.reader.read_next()
                endian = '<' if buf[1] == 1 else '>'
                sec, ns = struct.unpack_from(endian+'iI', buf, 4)
                self.pending = (sec+ns*1e-9, buf)
            t, buf = self.pending
            if t > stamp+1e-6: break
            self.last = self.pending
            self.pending = None
            if abs(t-stamp) <= 1e-6: break
        if self.last is None or abs(self.last[0]-stamp) > 1e-6:
            raise ValueError(f'No exact image for pose timestamp {stamp}; refusing to display a different frame')
        message = self.deserialize(self.last[1], self.image_type)
        return message.header.stamp, split_cameras(decode_mosaic(message), self.quadrants, self.cameras)


def labeled_mosaic(tiles):
    """Camera IDs, not raw mosaic quadrants, determine display order."""
    output = []
    for cid in range(4):
        tile = cv2.resize(tiles[cid], (400, 320), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (160, 30), (0, 0, 0), -1)
        cv2.putText(tile, f'Camera {cid}', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1, cv2.LINE_AA)
        output.append(tile)
    return np.vstack([np.hstack(output[:2]), np.hstack(output[2:])])
