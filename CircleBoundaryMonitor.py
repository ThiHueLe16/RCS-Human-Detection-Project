
from datetime import datetime

class CircleBoundaryMonitor:
    def __init__(self, center, outer_radius, inner_radius):
        self.center = center
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius

    def get_zone(self, x, y):
        """
        Determines which zone the point (x, y) is in:
        - 0: Outside both circles
        - 1: Inside outer circle, outside inner circle (Caution zone)
        - 2: Inside inner circle (Alarm zone)
        """
        distance_squared = (x - self.center[0]) ** 2 + (y - self.center[1]) ** 2
        if distance_squared <= self.inner_radius ** 2:
            return 2  # Alarm zone
        elif distance_squared <= self.outer_radius ** 2:
            return 1  # Caution zone
        return 0  # Outside

    def log_event(self, event):
        """
        Logs the event with a timestamp.
        """
        with open("boundary_events_log.txt", "a") as log_file:
            log_file.write(f"{event} at {datetime.now()}\n")
        print(event)


