class RobotCommandProcessor(object):
    def send_command(self, cmd):
        raise NotImplementedError

    def register_state_cb(self, cb):
        raise NotImplementedError


class GripperWrapper(object):
    def sync_set_gripper_position(self, position, effort=50):
        raise NotImplementedError
