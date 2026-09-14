#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from std_msgs.msg import Float64
from tf2_ros import Buffer, TransformListener



class track2(Node):
    def __init__(self):
        # Inicializa el nodo
        super().__init__('track2')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Mensaje de inicio
        self.get_logger().info('¡El nodo base ha sido iniciado!')

        # Subscribers
        # self.subscription = self.create_subscription(
        #     Odometry,
        #     '/ekf/odometry',
        #     self.odom_callback,
        #     10
        # )

        self.publishData = self.create_publisher(
                    Float64,
                    '/error_lateral',
                    10
                )
        
        self.publishData2 = self.create_publisher(
                    Float64,
                    '/error_derivativo',
                    10
                )

        self.publishData3 = self.create_publisher(
                            Float64,
                            '/error_angular',
                            10
                        )
        

        self.subscription2 = self.create_subscription(Path, '/planned_path', self.Pathway, 10)

        # Publisher
        self.cmd_pub = self.create_publisher(
            TwistStamped,
            '/diffdrive_controller/cmd_vel',
            10
        )

        # Variables iniciales
        self.current_position = np.array([0.0, 0.0])
        self.current_yaw = 0.0
        self.current_path = None

        self.punto = 0
        self.indice_meta = 1

        self.error_anterior = 0.0
        self.dt = 0.1

        self.v_lineal = 0.0
        self.w_angular = 0.0

        # Timer cada 0.1 segundos
        self.time = self.create_timer(
            0.1,
            self.timer_callback
        )


    # def odom_callback(self, msg):

    #     self.current_position = np.array([
    #         msg.pose.pose.position.x,
    #         msg.pose.pose.position.y
    #     ])

    #     q = msg.pose.pose.orientation

    #     # Quaternion -> yaw
    #     self.current_yaw = np.arctan2(
    #         2.0 * (q.w * q.z + q.x * q.y),
    #         1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    #     )

    #     self.get_logger().info(
    #         f"Posición actual: {self.current_position} | "
    #         f"Yaw actual: {np.degrees(self.current_yaw):.2f} deg"
    #     )



    def Pathway(self, msg):

        # Verifica que el Path tenga puntos
        if len(msg.poses) == 0:
            self.get_logger().warn("El Path recibido está vacío.")
            return

        # Convertir el Path a una lista de puntos [x, y]
        self.current_path = np.array([
            [
                pose.pose.position.x,
                pose.pose.position.y
            ]
            for pose in msg.poses
        ])

        # Reiniciar el recorrido cuando llega un nuevo Path
        self.punto = 0
        self.indice_meta = 1

        # self.get_logger().info(
        #     f"Path recibido con {len(self.current_path)} puntos."
        # )

    def timer_callback(self):

        try:
            # Preguntar a TF la posición del robot en el mapa
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            
            # Actualizar la posición actual
            self.current_position = np.array([
                trans.transform.translation.x,
                trans.transform.translation.y
            ])
            
            # Extraer el Yaw desde el cuaternión
            q = trans.transform.rotation
            self.current_yaw = np.arctan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )
        except Exception as e:
        # Si la transformación aún no existe, no hace nada y espera al siguiente ciclo
          return

        robot = self.current_position

        # ---------------------------------------------
        # Verificar que exista una trayectoria
        # ---------------------------------------------

        if self.current_path is None:
            return

        if len(self.current_path) < 2:
            return

      

        # ---------------------------------------------
        # Buscar el punto de la trayectoria más cercano
        # ---------------------------------------------

        distancias = np.linalg.norm(
            self.current_path - robot,
            axis=1
        )

        punto_cercano = int(np.argmin(distancias))

        # ---------------------------------------------
        # Buscar un punto ligeramente adelantado
        # ---------------------------------------------
        
        lookahead_points = 8

        punto_objetivo = min(
            punto_cercano + lookahead_points,
            len(self.current_path) - 1
        )

        objetivo = self.current_path[punto_objetivo]

        distancia = np.linalg.norm(
            objetivo - robot
        )

        # ---------------------------------------------
        # Obtener segmento de trayectoria
        # ---------------------------------------------

        punto_siguiente = min(
            punto_cercano + 1,
            len(self.current_path) - 1
        )

        w1 = self.current_path[punto_cercano]
        w2 = self.current_path[punto_siguiente]

        vector_path = w2 - w1

        base = np.linalg.norm(vector_path)

        if base < 1e-6:
            return

        # ---------------------------------------------
        # Cross Track Error FIRMADO
        # ---------------------------------------------

        vector_robot = robot - w1

        Area = (
            vector_path[0] * vector_robot[1]
            -
            vector_path[1] * vector_robot[0]
        )

        altura = Area / base

        # ---------------------------------------------
        # Ángulo deseado de la trayectoria
        # ---------------------------------------------

        path_angle = np.arctan2(
            vector_path[1],
            vector_path[0]
        )

        # ---------------------------------------------
        # Error angular
        # ---------------------------------------------

        heading_error = path_angle - self.current_yaw

        # Normalizar entre -pi y pi

        heading_error = np.arctan2(
            np.sin(heading_error),
            np.cos(heading_error)
        )

        # ---------------------------------------------
        # Control angular
        # ---------------------------------------------

        kw = 2.5
        kc = 1.5
        kd_w = 0.15

        derivada_error = (
            altura - self.error_anterior
        ) / self.dt

        w_angular = (
            kw * heading_error
            -
            kc * altura
            +
            kd_w * derivada_error
        )

        # ---------------------------------------------
        # Limitar velocidad angular
        # ---------------------------------------------

        self.w_angular = np.clip(
            w_angular,
            -1.5,
            1.5
        )

        # ---------------------------------------------
        # Velocidad lineal
        # ---------------------------------------------

        kv = 2.2

        v_lineal = kv * distancia

        # Reducir velocidad cuando hay mucho error angular
        if abs(heading_error) > np.radians(45):
            v_lineal *= 0.4

        elif abs(heading_error) > np.radians(25):
            v_lineal *= 0.7

        self.v_lineal = np.clip(
            v_lineal,
            0.0,
            0.5
        )

        # ---------------------------------------------
        # Guardar error
        # ---------------------------------------------

        self.error_anterior = altura

        # ---------------------------------------------
        # Información
        # ---------------------------------------------

        self.get_logger().info(
            f"CTE: {altura:.3f} | "
            f"Heading: {np.degrees(heading_error):.1f} deg | "
            f"Punto: {punto_cercano} | "
            f"Objetivo: {punto_objetivo}"
             f"Posición actual: {self.current_position} | "
            f"Yaw actual: {np.degrees(self.current_yaw):.2f} deg"

       
        )

        # ---------------------------------------------
        # Publicar
        # ---------------------------------------------

        self.publish_cmd(
            self.v_lineal,
            self.w_angular
        )

        self.publishData.publish(
            Float64(data=float(altura))
        )

        self.publishData2.publish(
            Float64(data=float(derivada_error))
        )

        self.publishData3.publish(
            Float64(data=float(heading_error))
        )

    def publish_cmd(self, vx, wz):

        cmd = TwistStamped()

        cmd.header.stamp = (
            self.get_clock().now().to_msg()
        )

        cmd.header.frame_id = 'base_link'

        cmd.twist.linear.x = float(vx)
        cmd.twist.linear.y = 0.0
        cmd.twist.linear.z = 0.0

        cmd.twist.angular.x = 0.0
        cmd.twist.angular.y = 0.0
        cmd.twist.angular.z = float(wz)

        self.cmd_pub.publish(cmd)


def main(args=None):

    # Inicializa ROS 2
    rclpy.init(args=args)

    # Crea el nodo
    nodo = track2()

    try:
        # Mantiene el nodo ejecutándose
        rclpy.spin(nodo)

    except KeyboardInterrupt:

        nodo.get_logger().info(
            'Apagando el nodo de forma segura...'
        )

    finally:

        nodo.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

