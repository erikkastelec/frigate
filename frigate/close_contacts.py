from frigate.config import CameraConfig
from frigate.util import find_close_bboxes


class CloseContact:
    def __init__(self, id1, id2, last_distance, frame_time):
        self.id1 = id1
        self.id2 = id2
        self.last_distance = last_distance
        self.frame_time = frame_time
        self.contact_time = None
        self.last_frame_time = frame_time

    # So we can delete close contacts if object with self.id2 stops being tracked
    def __hash__(self):
        return hash(self.id2)

    def __eq__(self, other):
        return (self.id1 == other.id1 and self.id2 == other.id2) or (
            self.id1 == other.id2 and self.id2 == other.id1
        )

    """
    Computes contact time before we are sure the contact ended (we did not detect safe distance between objects)
    It should be used only when we need total contact time before for example deleting the tracked object.
    """

    def update_contact_time(self, frame_time):
        if self.contact_time:
            self.contact_time = frame_time - self.frame_time
        else:
            self.contact_time += frame_time - self.frame_time
        return self.contact_time


class CloseContactsDetector:
    def __init__(self, camera_config: CameraConfig):
        self.camera_config = camera_config

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
