Kineverse Experiment World
==========================
This repository contains a ROS package which implements a number of demos for the [Kineverse](https://github.com/ARoefer/kineverse) articulation framework.

Installation
------------
Clone the package to a ROS workspace.

```bash
cd ros_ws/src
git clone https://github.com/ARoefer/kineverse_experiment_world.git
```

In addition to the main package you will need to install a number of other dependencies.
For the installation of [Kineverse](https://github.com/ARoefer/kineverse), please refer to its installation instructions.

```
# Small robot simulator
git clone https://github.com/ARoefer/iai_bullet_sim.git

# Description of the fetch robot
git clone https://github.com/fetchrobotics/fetch_ros.git
mv fetch_ros/fetch_description ./
rm -rf fetch_ros  # We don't actually need the rest

# IAI robots
git clone https://github.com/code-iai/iai_robots.git

# IAI maps
git clone https://github.com/code-iai/iai_maps.git
```

Rebuild your workspace and source the `devel/setup.bash` file again so ROS discovers all the packages.

Running The Demos
-----------------

The are 

### Animated Kinematics
This demo shows a number of different kinematics implemented in Kineverse and visualized using RVIZ. The demo is a single ROS node and can be run like this:

```bash
# If its not runing already: ROS master
roscore

# The actual node
rosrun kineverse_experiment_world all_kinematics.py

# Fire up RVIZ to see the differen kinematics
rviz -d `rospack find kineverse_experiment_world`/rviz/all_kinematics.rviz
```

### Kitchen Pushing Trajectory Planner
This demo plans trajectories for a PR2 or Fetch to push shut the cabinet doors and drawers of the IAI kitchen. This allows you to view the motions undisturbed by simulation side-effects.

```bash
# If its not runing already: ROS master
roscore

# Fire up RVIZ to see the motions being planned
rviz -d `rospack find kineverse_experiment_world`/rviz/pushing_trajectories.rviz

# The actual node
rosrun kineverse_experiment_world pushing_demo_dry_run.py
```

After first running the node, you might have to refresh the *RobotModel* displays in RVIZ so they load the URDFs from the parameter server.

This demo contains more than one scenario. The options can be displayed by starting it with a `-h` argument:

```bash
~$ rosrun kineverse_experiment_world pushing_demo_dry_run.py -h
usage: pushing_demo_dry_run.py [-h] [--robot ROBOT] [--omni OMNI] [--nav NAV]
                               [--vis-plan VIS_PLAN]

Plans motions for closing doors and drawers in the IAI kitchen environment
using various robots.

optional arguments:
  -h, --help            show this help message and exit
  --robot ROBOT, -r ROBOT
                        Name of the robot to use. [ pr2 | fetch ]
  --omni OMNI           To use an omnidirectional base or not.
  --nav NAV             Heuristic for navigating object geometry. [ cross |
                        linear | cubic ]
  --vis-plan VIS_PLAN   Visualize trajector while planning.
```

* The `-r` option allows you to select a robot. By default this is the `pr2`.
* The `--omni` option allows you to toggle using an omnidirectional or differential drive base for the fetch. By default it is `False`.
* The `--nav` option allows you to select the method by which the gripper navigates to a better push location. By default it is `cross`.
* The `--vis-plan` option allows you to visualize the motion planning as it is happening. This will not play back the trajectory at the correct speed. By default this is false, leading to the trajectories being planned first and then visualized using TF and the robot model displays in RVIZ.

### Object Tracker
In this package is a very simple algorithm for tracking the configuration of an articulated object based on the poses of its parts. This tracker exists in two forms: A benchmarking application and an online tracker.
Both require the kitchen model to be available on a Kineverse server.

#### Running the Benchmark
The benchmark will track larger and larger parts of the kitchen for randomly sampled poses at different noise levels. This benchmark exists to determine the relationship of noise to estimation error and solver time.

```bash
# If its not runing already: ROS master
roscore

# Start a Kineverse server, if it not yet running
rosrun kineverse kineverse_server.py

# Upload the kitchen to the server
rosrun kineverse upload_urdf.py package://iai_kitchen/urdf_obj/IAI_kitchen.urdf

# Run the tracker
rosrun kineverse obj_tracker_statistical_eval.py
```

Running the benchmark with its default configuration can take quite some time, e.g. 21 minutes on an i7-6700. By default, the tracker will create a csv file called `tracker_results_n_dof.csv` in the directory it was started in. Starting the tracker with the `-h` argument reveals that it also has a number of parameters that can changed. 
To just run it quickly, you could try setting `-s 20` and `-mi 5`, which takes around 90 seconds to run on my machine.

#### Running the Tracker on Simulation Data
The simulation environment does not have to be configured manually. Instead there exists a launchfile that takes care of that.

```bash
# If its not runing already: ROS master
roscore

# Start the Kineverse server, simulation, and upload the models to the server
roslaunch kineverse_experiment_world iai_kitchen_robot_scenario.launch use_tracker:=true

# Visualize using RVIZ
rviz -d `rospack find kineverse_experiment_world`/rviz/sim_fetch.rviz
```

The launch file will open a window showing the simulated scene. The scene in this window is interactive and objects can be dragged around in it.

In the RVIZ visualization you should see two kitchens, one solid, one transparent. The transparent one is the ground truth state of the kitchen in the simulation. The solid one is the result of the tracker. The small pose markers at points of the kitchen show which poses are currently observed.

Use your mouse to orient the robot and to move parts of the kitchen. Observe the updates in RVIZ.

### Run Pushing in Simulation
The final demo in this package is a small behavior that reacts to parts of the kitchen being left open. When it observes one of these parts, it will try to push them shut.

#### PR2

```bash
# If its not runing already: ROS master
roscore

# Start the Kineverse server, simulation, and upload the models to the server
roslaunch kineverse_experiment_world iai_kitchen_robot_scenario.launch use_tracker:=true robot:=pr2

# Start the behavior
rosrun kineverse_experiment_world obsessive_demo.py pr2

# Visualize using RVIZ
rviz -d `rospack find kineverse_experiment_world`/rviz/sim_pr2.rviz
```
