import socket
import time
import unittest

from teleop_action_stream import TeleopAction, TeleopActionPublisher, TeleopActionSubscriber


def make_action(seq: int) -> TeleopAction:
    return TeleopAction(
        timestamp=time.time(),
        seq=seq,
        source="test",
        action=(float(seq), 0, 0, 0, 0, 0, 0),
        current_pose=(0, 0, 300, 0, 0, 0),
        target_pose=(float(seq), 0, 300, 0, 0, 0),
        current_joints=(0, 0, 0, 0, 0, 0),
        deadman=True,
        servo_sent=True,
        gripper_command=0.0,
    )


class TeleopActionStreamTest(unittest.TestCase):
    def test_subscriber_keeps_latest_valid_action(self):
        subscriber = TeleopActionSubscriber("127.0.0.1", 0)
        port = subscriber._socket.getsockname()[1]
        publisher = TeleopActionPublisher("127.0.0.1", port)
        try:
            self.assertTrue(publisher.publish(make_action(1)))
            self.assertTrue(publisher.publish(make_action(2)))
            latest = None
            deadline = time.monotonic() + 1.0
            while latest is None and time.monotonic() < deadline:
                latest = subscriber.poll_latest()
                time.sleep(0.005)
            self.assertIsNotNone(latest)
            self.assertEqual(latest.seq, 2)
            self.assertEqual(latest.action[0], 2.0)
        finally:
            publisher.close()
            subscriber.close()

    def test_malformed_packet_is_ignored(self):
        subscriber = TeleopActionSubscriber("127.0.0.1", 0)
        port = subscriber._socket.getsockname()[1]
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sender.sendto(b'{"action":[1,2]}', ("127.0.0.1", port))
            deadline = time.monotonic() + 1.0
            while subscriber.invalid_packets == 0 and time.monotonic() < deadline:
                subscriber.poll_latest()
                time.sleep(0.005)
            self.assertEqual(subscriber.invalid_packets, 1)
            self.assertIsNone(subscriber.latest)
        finally:
            sender.close()
            subscriber.close()
