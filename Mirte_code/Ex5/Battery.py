import rclpy
from rclpy.node import Node
from sensor_msgs.msg import BatteryState


class BatteryMonitor(Node):

    def __init__(self):
        super().__init__('battery_monitor')

        self.subscription = self.create_subscription(
            BatteryState,
            '/io/power/power_watcher',
            self.battery_callback,
            10
        )

        self.get_logger().info(
            'Waiting for battery data from /io/power/power_watcher...'
        )

    def battery_callback(self, msg):
        voltage = msg.voltage
        current = msg.current
        percentage = msg.percentage * 100.0

        if voltage <= 0.0:
            print(
                f'\rBattery: NO VALID READING'
                f' | Voltage: {voltage:.2f} V'
                f' | Current: {current:.2f} A',
                end='',
                flush=True
            )
            return

        print(
            f'\rBattery: {percentage:.0f}%'
            f' | Voltage: {voltage:.2f} V'
            f' | Current: {current:.2f} A',
            end='',
            flush=True
        )


def main(args=None):
    rclpy.init(args=args)

    node = BatteryMonitor()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        print()
        print('Stopping battery monitor...')

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()