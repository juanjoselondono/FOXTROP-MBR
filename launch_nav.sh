#!/bin/bash
WS_DIR="$HOME/Documents/mobileRobotics/mrad_ws_2602_Foxtrot"
SESSION="foxtrot_nav"

# Cerrar sesión previa si existe
tmux kill-session -t $SESSION 2>/dev/null

# 1. Crear sesión y panel principal (Arriba-Izquierda: Gazebo)
tmux new-session -d -s $SESSION -c "$WS_DIR"
P_GZ=$(tmux display-message -p '#{pane_id}')

# 2. Dividir horizontalmente (Arriba-Derecha: Navigation / EKF)
P_NAV=$(tmux split-window -h -P -F '#{pane_id}' -t $P_GZ -c "$WS_DIR")

# 3. Dividir Gazebo verticalmente (Abajo-Izquierda: AMCL)
P_AMCL=$(tmux split-window -v -P -F '#{pane_id}' -t $P_GZ -c "$WS_DIR")

# 4. Dividir Navigation verticalmente (Abajo-Derecha: RViz2)
P_RVIZ=$(tmux split-window -v -P -F '#{pane_id}' -t $P_NAV -c "$WS_DIR")

# Habilitar soporte para ratón (scroll, clic y redimensión)
tmux set-option -g mouse on

# --- Asignar Comandos a cada Panel Específico ---

# Panel 1: Gazebo Simulación
tmux send-keys -t $P_GZ "source /opt/ros/jazzy/setup.bash && source install/setup.bash" C-m
tmux send-keys -t $P_GZ "ros2 launch bringup_2602_foxtrot gz_spawn.launch.py world:=src/gazebo_2602_foxtrot/worlds/exam1_world.sdf gz_mode:=False" C-m

# Panel 2: Navigation (EKF + Trackers)
tmux send-keys -t $P_NAV "source /opt/ros/jazzy/setup.bash && source install/setup.bash" C-m
tmux send-keys -t $P_NAV "sleep 5 && ros2 launch bringup_2602_foxtrot navigation_foxtrot.launch.py use_sim_time:=true" C-m

# Panel 3: AMCL Localization
tmux send-keys -t $P_AMCL "source /opt/ros/jazzy/setup.bash && source install/setup.bash" C-m
tmux send-keys -t $P_AMCL "sleep 7 && ros2 launch bringup_2602_foxtrot amcl_localization.launch.py map:=src/gazebo_2602_foxtrot/maps/map2.yaml use_sim_time:=true" C-m

# Panel 4: RViz2
tmux send-keys -t $P_RVIZ "source /opt/ros/jazzy/setup.bash && source install/setup.bash" C-m
tmux send-keys -t $P_RVIZ "sleep 9 && rviz2" C-m

# Adjuntar sesión en pantalla
tmux attach-session -t $SESSION
