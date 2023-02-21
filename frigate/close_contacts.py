from frigate.config import CameraConfig
from frigate.util import find_close_bboxes
import datetime


class CloseContact:
    def __init__(self, id1, id2, last_distance, frame_time):
        assert id1 != id2 and frame_time is not None
        self.id1 = min(id1, id2)
        self.id2 = max(id1, id2)
        self.id = f"{self.id1}|{self.id2}"
        self.last_distance = last_distance
        self.first_frame_time = frame_time
        # Used for calculating contact_time
        self.frame_time = frame_time
        self.contact_time = None
        self.last_frame_time = frame_time

    """
    Computes contact time before we are sure the contact ended (we did not detect safe distance between objects)
    It should be used only when we need total contact time before for example deleting the tracked object.
    """

    def update_contact_time(self, frame_time):
        assert frame_time is not None
        if not self.contact_time:
            self.contact_time = frame_time - self.last_frame_time
        else:
            try:
                self.contact_time += frame_time - self.last_frame_time
            # TODO: remove after debug
            except TypeError as t:
                print("ERR")
        return self.contact_time

    def contains_id(self, id):
        return id == self.id1 or id == self.id2

    def to_dict(self):
        return {
            "id": self.id,
            "id1": self.id1,
            "id2": self.id2,
            "last_distance": self.last_distance,
            "first_frame_time": self.first_frame_time,
            "frame_time": self.frame_time,
            "contact_time": self.contact_time,
            "last_frame_time": self.last_frame_time,
        }


class CloseContactsTracker:
    def __init__(self, camera_config: CameraConfig):
        self.camera_config = camera_config
        self.close_contacts = {}

    def detect(self, objects, frame_time):
        close_bboxes, non_close_bboxes = find_close_bboxes(
            [
                (obj["box"], obj["id"])
                for obj in objects.values()
                if obj["frame_time"] == frame_time
            ],
            self.camera_config.calibration.homography_matrix,
            self.camera_config.calibration.scale_factor,
            self.camera_config.close_contacts.distance_threshold,
        )
        return close_bboxes, non_close_bboxes

    # TODO: What happens if detection fps is set higher than camera fps? Maybe we can use EventsPerSecond()?
    # Calculate contact time in seconds based on number of frames and detection fps
    def is_close_contact(self, close_contact: CloseContact):

        return (
            close_contact.frame_count / self.camera_config.detect.fps
            >= self.close_contacts_config.time_threshold
        )

    def create_key(self, id1, id2):
        tmp = f"{min(id1, id2)}|{max(id1, id2)}"
        return tmp

    def register(self, obj1, obj2, frame_time):
        self.close_contacts[self.create_key(obj1.id, obj2.id)] = CloseContact(
            obj1.id, obj2.id, frame_time
        )

    def deregister(self, id, frame_time):
        cc = self.get_close_contacts_for_object(id)

        for close_contact in cc:
            close_contact.update_contact_time(frame_time)
            self.close_contacts.pop(close_contact.id)

    def is_expired(self, id, frame_time):
        try:
            cc = self.close_contacts[id]
        except KeyError:
            # TODO: remove after debug
            print("ERR")
        return (
            frame_time - cc.last_frame_time
            > self.camera_config.close_contacts.max_disappeared
        )

    def update_close_contacts(
        self,
        tracked_objects: dict,
        frame_time: datetime.datetime,
    ):
        close_objects, non_close_objects = self.detect(tracked_objects, frame_time)
        if close_objects:
            for id1, id2, distance in close_objects:

                try:
                    cc = self.close_contacts[self.create_key(id1, id2)]
                    cc.last_distance = distance
                    cc.last_frame_time = frame_time
                    # Do not update if not None so we can track the time from start of the contact
                    if not cc.frame_time:
                        cc.frame_time = frame_time
                except KeyError:
                    self.close_contacts[self.create_key(id1, id2)] = CloseContact(
                        id1, id2, distance, frame_time
                    )

        if non_close_objects:
            for id1, id2, distance in non_close_objects:
                try:
                    # Calculate the time from last close contact to non contact now
                    cc = self.close_contacts[self.create_key(id1, id2)]
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
        to_be_removed = []
        for cc in self.close_contacts.values():

            if self.is_expired(cc.id, frame_time):
                to_be_removed.append((cc.id, frame_time))
        for id, ft in to_be_removed:
            self.deregister(id, ft)

    def get_close_contacts_for_object(self, obj_id: str):
        return [
            close_contact
            for close_contact in self.close_contacts.values()
            if close_contact.contains_id(obj_id)
        ]
