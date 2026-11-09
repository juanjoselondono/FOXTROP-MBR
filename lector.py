#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import cv2
import numpy as np

class YoloSubscriber(Node):
    def __init__(self):
        super().__init__('yolo_subscriber')
        
        # 1. El mismo perfil BEST_EFFORT exacto que tiene la Raspberry
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # 2. Suscriptor al tópico de video comprimido
        self.subscription = self.create_subscription(
            CompressedImage,
            '/video_stream/compressed',
            self.listener_callback,
            qos_profile
        )
        self.get_logger().info("Nodo PC listo. Escuchando video en BEST_EFFORT...")

    def listener_callback(self, msg):
        # 3. Convertir los bytes comprimidos (JPEG) de vuelta a una matriz de imagen
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # ==========================================
        # AQUÍ IRÁ TU INFERENCIA DE YOLO (YOLOv8/v11)
        # ==========================================
        
        # 4. Mostrar el video en tiempo real en la PC
        cv2.imshow("Video desde el Robot (OAK-D)", frame)
        cv2.waitKey(1) # Necesario para que OpenCV actualice la ventana

def main(args=None):
    rclpy.init(args=args)
    node = YoloSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()