import copy
import datetime
import itertools
import multiprocessing as mp
import random
import string
import threading
import time
from collections import defaultdict

import cv2
import numpy as np
from scipy.spatial import distance as dist

from frigate.config import DetectConfig
from frigate.util import intersection_over_union, generate_random_color
from frigate.close_contacts import CloseContact
from typing import Dict, List, Tuple

from frigate.sort_tracker import *
import logging

logger = logging.getLogger(__name__)


class ObjectTracker:
    def __init__(self, config: DetectConfig):
        self.tracked_objects = {}
        self.disappeared = {}
        self.positions = {}
        self.max_disappeared = config.max_disappeared
        self.detect_config = config

    def register(self, index, obj):
        rand_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
        id = f"{obj['frame_time']}-{rand_id}"
        obj["id"] = id
        obj["start_time"] = obj["frame_time"]
        obj["motionless_count"] = 0
        obj["position_changes"] = 0
        # TODO: check if needed
        obj["close_contacts"] = {}
        obj["color"] = generate_random_color()
        self.tracked_objects[id] = obj
        self.disappeared[id] = 0
        self.positions[id] = {
            "xmins": [],
            "ymins": [],
            "xmaxs": [],
            "ymaxs": [],
            "xmin": 0,
            "ymin": 0,
            "xmax": self.detect_config.width,
            "ymax": self.detect_config.height,
        }

    def deregister(self, id):
        # Delete close contacts
        try:
            for contact in self.tracked_objects[id]["close_contacts"].values():
                del self.tracked_objects[contact.id2]["close_contacts"][contact.id1]
            del self.tracked_objects[id]
            del self.disappeared[id]
        except KeyError as e:
            logger.error(f"Unable to delete tracked object {id}")
            logger.error(e)

    # tracks the current position of the object based on the last N bounding boxes
    # returns False if the object has moved outside its previous position
    def update_position(self, id, box):
        position = self.positions[id]
        position_box = (
            position["xmin"],
            position["ymin"],
            position["xmax"],
            position["ymax"],
        )

        xmin, ymin, xmax, ymax = box

        iou = intersection_over_union(position_box, box)

        # if the iou drops below the threshold
        # assume the object has moved to a new position and reset the computed box
        if iou < 0.6:
            self.positions[id] = {
                "xmins": [xmin],
                "ymins": [ymin],
                "xmaxs": [xmax],
                "ymaxs": [ymax],
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
            return False

        # if there are less than 10 entries for the position, add the bounding box
        # and recompute the position box
        if len(position["xmins"]) < 10:
            position["xmins"].append(xmin)
            position["ymins"].append(ymin)
            position["xmaxs"].append(xmax)
            position["ymaxs"].append(ymax)
            # by using percentiles here, we hopefully remove outliers
            position["xmin"] = np.percentile(position["xmins"], 15)
            position["ymin"] = np.percentile(position["ymins"], 15)
            position["xmax"] = np.percentile(position["xmaxs"], 85)
            position["ymax"] = np.percentile(position["ymaxs"], 85)

        return True

    def update_close_contacts(
        self,
        close_objects: list(tuple((string, string, float))),
        non_close_objects: list(tuple((string, string, float))),
        frame_time: datetime.datetime,
    ):
        if close_objects:
            for id1, id2, distance in close_objects:
                # TODO: clean up, could probably use an update method for CloseContact

                # Update last distance if detected in close objects and frame time if rediscovered contact

                try:
                    cc = self.tracked_objects[id1]["close_contacts"][id2]
                    cc.last_distance = distance
                    cc.last_frame_time = frame_time
                    # Do not update if not None so we can track the time from start of the contact
                    if not cc.frame_time:
                        cc.frame_time = frame_time
                except KeyError:
                    self.tracked_objects[id1]["close_contacts"][id2] = CloseContact(
                        id1, id2, distance, frame_time
                    )

                try:
                    cc = self.tracked_objects[id2]["close_contacts"][id1]
                    cc.last_distance = distance
                    cc.last_frame_time = frame_time
                    if not cc.frame_time:
                        cc.frame_time = frame_time
                except KeyError:
                    self.tracked_objects[id2]["close_contacts"][id1] = CloseContact(
                        id2, id1, distance, frame_time
                    )

        if non_close_objects:
            for id1, id2, distance in non_close_objects:
                try:
                    # Calculate the time from last close contact to non contact now
                    cc = self.tracked_objects[id1]["close_contacts"][id2]
                    if cc.frame_time:
                        time_from_last_contact = frame_time - cc.frame_time
                        if cc.contact_time:
                            cc.contact_time += time_from_last_contact
                        else:
                            cc.contact_time = time_from_last_contact

                    # Reset the frame time to None so we know it is not in close contact
                    cc.frame_time = None
                except KeyError:
                    continue

                try:
                    cc = self.tracked_objects[id2]["close_contacts"][id1]
                    if cc.frame_time:
                        time_from_last_contact = frame_time - cc.frame_time
                        if cc.contact_time:
                            cc.contact_time += time_from_last_contact
                        else:
                            cc.contact_time = time_from_last_contact
                    cc.frame_time = None
                except KeyError:
                    continue

    def is_expired(self, id):
        obj = self.tracked_objects[id]
        # get the max frames for this label type or the default
        max_frames = self.detect_config.stationary.max_frames.objects.get(
            obj["label"], self.detect_config.stationary.max_frames.default
        )

        # if there is no max_frames for this label type, continue
        if max_frames is None:
            return False

        # if the object has exceeded the max_frames setting, deregister
        if (
            obj["motionless_count"] - self.detect_config.stationary.threshold
            > max_frames
        ):
            return True

        return False

    def update(self, id, new_obj):
        self.disappeared[id] = 0
        # update the motionless count if the object has not moved to a new position
        if self.update_position(id, new_obj["box"]):
            self.tracked_objects[id]["motionless_count"] += 1
            if self.is_expired(id):
                self.deregister(id)
                return
        else:
            # register the first position change and then only increment if
            # the object was previously stationary
            if (
                self.tracked_objects[id]["position_changes"] == 0
                or self.tracked_objects[id]["motionless_count"]
                >= self.detect_config.stationary.threshold
            ):
                self.tracked_objects[id]["position_changes"] += 1
            self.tracked_objects[id]["motionless_count"] = 0

        self.tracked_objects[id].update(new_obj)

    def update_frame_times(self, frame_time):
        for id in list(self.tracked_objects.keys()):
            self.tracked_objects[id]["frame_time"] = frame_time
            self.tracked_objects[id]["motionless_count"] += 1
            if self.is_expired(id):
                self.deregister(id)

    def match_and_update(self, frame_time, new_objects):
        # group by name
        new_object_groups = defaultdict(lambda: [])
        for obj in new_objects:
            new_object_groups[obj[0]].append(
                {
                    "label": obj[0],
                    "score": obj[1],
                    "box": obj[2],
                    "area": obj[3],
                    "ratio": obj[4],
                    "region": obj[5],
                    "frame_time": frame_time,
                }
            )

        # update any tracked objects with labels that are not
        # seen in the current objects and deregister if needed
        for obj in list(self.tracked_objects.values()):
            if not obj["label"] in new_object_groups:
                if self.disappeared[obj["id"]] >= self.max_disappeared:
                    self.deregister(obj["id"])
                else:
                    self.disappeared[obj["id"]] += 1

        if len(new_objects) == 0:
            return

        # track objects for each label type
        for label, group in new_object_groups.items():
            current_objects = [
                o for o in self.tracked_objects.values() if o["label"] == label
            ]
            current_ids = [o["id"] for o in current_objects]
            current_centroids = np.array([o["centroid"] for o in current_objects])

            # compute centroids of new objects
            for obj in group:
                centroid_x = int((obj["box"][0] + obj["box"][2]) / 2.0)
                centroid_y = int((obj["box"][1] + obj["box"][3]) / 2.0)
                obj["centroid"] = (centroid_x, centroid_y)

            if len(current_objects) == 0:
                for index, obj in enumerate(group):
                    self.register(index, obj)
                continue

            new_centroids = np.array([o["centroid"] for o in group])

            # compute the distance between each pair of tracked
            # centroids and new centroids, respectively -- our
            # goal will be to match each current centroid to a new
            # object centroid
            D = dist.cdist(current_centroids, new_centroids)

            # in order to perform this matching we must (1) find the smallest
            # value in each row (i.e. the distance from each current object to
            # the closest new object) and then (2) sort the row indexes based
            # on their minimum values so that the row with the smallest
            # distance (the best match) is at the *front* of the index list
            rows = D.min(axis=1).argsort()

            # next, we determine which new object each existing object matched
            # against, and apply the same sorting as was applied previously
            cols = D.argmin(axis=1)[rows]

            # many current objects may register with each new object, so only
            # match the closest ones.  unique returns the indices of the first
            # occurrences of each value, and because the rows are sorted by
            # distance, this will be index of the closest match
            _, index = np.unique(cols, return_index=True)
            rows = rows[index]
            cols = cols[index]

            # loop over the combination of the (row, column) index tuples
            for row, col in zip(rows, cols):
                # grab the object ID for the current row, set its new centroid,
                # and reset the disappeared counter
                objectID = current_ids[row]
                self.update(objectID, group[col])

            # compute the row and column indices we have NOT yet examined
            unusedRows = set(range(D.shape[0])).difference(rows)
            unusedCols = set(range(D.shape[1])).difference(cols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    id = current_ids[row]

                    if self.disappeared[id] >= self.max_disappeared:
                        self.deregister(id)
                    else:
                        self.disappeared[id] += 1
            # if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(col, group[col])


# NEEDS REFACTORING


class SortObjectTracker(ObjectTracker):
    def __init__(self, config: DetectConfig, min_hits=5, iou_threshold=0.3):
        super().__init__(config)
        # TODO: move this to configuration file
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

        # self.sort_to_object_tracker_map = {}

    def update_tracked_object(self, id, new_obj):
        self.disappeared[id] = 0
        # update the motionless count if the object has not moved to a new position
        if self.update_position(id, new_obj["box"]):
            self.tracked_objects[id]["motionless_count"] += 1
            if self.is_expired(id):
                self.deregister(id)
                return
        else:
            # register the first position change and then only increment if
            # the object was previously stationary
            if (
                self.tracked_objects[id]["position_changes"] == 0
                or self.tracked_objects[id]["motionless_count"]
                >= self.detect_config.stationary.threshold
            ):
                self.tracked_objects[id]["position_changes"] += 1
            self.tracked_objects[id]["motionless_count"] = 0

        self.tracked_objects[id].update(new_obj)

    def update(self, frame_time, new_objects):

        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        detections = [
            [obj[2][0], obj[2][1], obj[2][2], obj[2][3], obj[1]] for obj in new_objects
        ]
        dets = np.array(detections)
        if len(detections) == 0:
            dets = np.empty((0, 5))

        new_objects = [
            {
                "label": obj[0],
                "score": obj[1],
                "box": obj[2],
                "area": obj[3],
                "ratio": obj[4],
                "region": obj[5],
                "frame_time": frame_time,
                "centroid": (
                    int((obj[2][0] + obj[2][2]) / 2.0),
                    int((obj[2][1] + obj[2][3]) / 2.0),
                ),
            }
            for obj in new_objects
        ]
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.tracked_objects), 5))

        to_del = []
        ret = []
        for t, trk in zip(self.tracked_objects.keys(), trks):
            pos = self.tracked_objects[t]["tracker"].predict()[0]
            # pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                self.deregister(t)

        # used to map associate_detection_to_trackers to the correct object
        tmp_mapper = list(zip(self.tracked_objects.keys(), trks))
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        # for t in reversed(to_del):
        #     self.deregister(self.sort_to_object_tracker_map[t])
        #     self.trackers.pop(t)
        #     del self.sort_to_object_tracker_map[t]
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )
        # update matched trackers with assigned detections
        for m in matched:
            self.tracked_objects[tmp_mapper[m[1]][0]]["tracker"].update(dets[m[0], :])
            self.update_tracked_object(tmp_mapper[m[1]][0], new_objects[m[0]])
            # self.trackers[m[1]].update(dets[m[0], :])
            # try:
            #     self.update_tracked_object(
            #         self.sort_to_object_tracker_map[m[1]], new_objects[m[0]]
            #     )
            # except KeyError:
            #     print("hello")
        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            # trk = KalmanBoxTracker(dets[i, :])
            # self.trackers.append(trk)
            new_objects[i]["tracker"] = KalmanBoxTracker(dets[i, :])
            self.register(-1, new_objects[i])
            # self.sort_to_object_tracker_map[len(self.trackers) - 1] = id_to_map
        # i = len(self.tracked_objects)
        to_del = []
        for key in self.tracked_objects.keys():
            # for trk in reversed(self.trackers):
            # d = trk.get_state()[0]
            # if (trk.time_since_update < 1) and (
            #     trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            # ):
            #     ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            # i -= 1
            # remove dead tracklet
            if (
                self.tracked_objects[key]["tracker"].time_since_update
                > self.max_disappeared
            ):
                to_del.append(key)
                # self.deregister(key)
                # self.trackers.pop(i)
                # del self.sort_to_object_tracker_map[i]
        for key in to_del:
            self.deregister(key)
        # if len(ret) > 0:
        #     return np.concatenate(ret)
        # return np.empty((0, 5))

    def update_frame_times(self, frame_time):
        for id in list(self.tracked_objects.keys()):
            self.tracked_objects[id]["frame_time"] = frame_time
            self.tracked_objects[id]["motionless_count"] += 1
            if self.is_expired(id):
                self.deregister(id)
        self.update(frame_time, [])

    # To make switching between original tracker and SORT easier
    def match_and_update(self, frame_time, new_objects):
        # group by name
        self.update(frame_time, new_objects)
