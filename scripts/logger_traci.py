#!/usr/bin/env python3

import os
import sys
import math

# For pandas dataframe
import pandas as pd


# Check if SUMO_HOME exists
if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = "/usr/share/sumo"

print(f"SUMO_HOME is set to: {os.environ['SUMO_HOME']}")

# Add SUMO tools to the system path
tools_path = os.path.join(os.environ["SUMO_HOME"], "tools")
if tools_path not in sys.path:
    sys.path.append(tools_path)
print(f"SUMO tools path: {tools_path}")

import traci
import sumolib


# Physics constants from the proposal using Tesla Model 3 as reference.

MASS = 1823.0                 # kg
GRAVITY = 9.81                # m/s²
DRIVETRAIN_EFFICIENCY = 0.90  # eta
ROLLING_COEFF = 0.01          # C_r
AIR_DENSITY = 1.225           # rho, kg/m³
DRAG_COEFF = 0.225             # C_d
FRONTAL_AREA = 2.2            # A, m²
AUXILIARY_POWER = 1000      # P_aux, watts
# ROAD_GRADE_ANGLE = 0.0      theta, radians. Temporary flat-road assumption


def compute_power(speed, acceleration, road_grade_angle):
    """
    Compute instantaneous vehicle power P(t) using the proposal formula:

        P(t) = v(t)/eta * (
            m*a(t)
            + m*g*sin(theta)
            + m*g*C_r*cos(theta)
            + 0.5*rho*A*C_d*v(t)^2
        ) + P_aux

    Units:
        speed        : m/s
        acceleration : m/s²
        theta        : radians
        power        : watts
    """

    inertial_force = MASS * acceleration
    gravitational_force = MASS * GRAVITY * math.sin(road_grade_angle)
    rolling_force = MASS * GRAVITY * ROLLING_COEFF * math.cos(road_grade_angle)
    aerodynamic_force = 0.5 * AIR_DENSITY * FRONTAL_AREA * DRAG_COEFF * speed ** 2

    total_force = (
        inertial_force
        + gravitational_force
        + rolling_force
        + aerodynamic_force
    )

    power = (speed / DRIVETRAIN_EFFICIENCY) * total_force + AUXILIARY_POWER

    return power


def main():
    project_dir = os.path.expanduser("~/Internship/SUMO_Barcelona")

    config_file = os.path.join(project_dir, "simulations", "simulation_2004", "sim.sumocfg")
    output_file = os.path.join(project_dir, "outputs", "vehicle_log.csv")
    dataset_dir = os.path.join(project_dir, "dataset")

    print(f"Using SUMO config: {config_file}")
    print(f"Config exists: {os.path.exists(config_file)}")

    if not os.path.exists(config_file):
        print(f"ERROR: SUMO config file not found: {config_file}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    # Clean previous client files
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".csv"):
            os.remove(os.path.join(dataset_dir, filename))

    sumo_binary = sumolib.checkBinary("sumo")

    SIMULATION_END = 10 * 60 # 10 minutes.
    sumo_cmd = [
        sumo_binary,
        "-c", config_file,
        "--no-step-log", "true",
        "--duration-log.disable", "true",
        "--end", str(SIMULATION_END) 
    ]

    print("Starting SUMO with TraCI...")
    traci.start(sumo_cmd)

    records = []
    step_count = 0

    while (traci.simulation.getMinExpectedNumber() > 0 
    and traci.simulation.getTime() < SIMULATION_END):
        
        traci.simulationStep()
        step_count += 1

        current_time = traci.simulation.getTime()
        vehicle_ids = traci.vehicle.getIDList()

        # Log progress every 10 steps. Used for debugging and monitoring long simulations.
        if step_count % 10 == 0:
            print(
                f"Step {step_count}, "
                f"time={current_time}, "
                f"active vehicles={len(vehicle_ids)}, "
                f"records={len(records)}"
            )

        for veh_id in vehicle_ids:
            speed = traci.vehicle.getSpeed(veh_id)
            acceleration = traci.vehicle.getAcceleration(veh_id)
            slope_degrees = traci.vehicle.getSlope(veh_id)
            road_grade_angle = math.radians(slope_degrees)
            x, y = traci.vehicle.getPosition(veh_id)
            edge_id = traci.vehicle.getRoadID(veh_id)
            lane_id = traci.vehicle.getLaneID(veh_id) 
            # Filtering anomalies
            if speed < 0:
                continue

            if abs(acceleration) > 5.0:
                continue

            signed_power = compute_power(speed, acceleration, road_grade_angle)

            # Energy-consumption target: non-negative power only
            consumption_power = max(signed_power, 0.0)

            records.append({
                "time": current_time,
                "vehicle_id": veh_id,
                "speed": speed,
                "acceleration": acceleration,
                "slope_degrees": slope_degrees,
                "road_grade_angle": road_grade_angle,
                "x": x,
                "y": y,
                "edge_id": edge_id,
                "lane_id": lane_id,
                "signed_power_watts": signed_power,
                "consumption_power_watts": consumption_power
            })

    traci.close()

    print("Simulation finished. Saving data to CSV...")
    print(f"Total simulation steps: {step_count}")
    print(f"Total logged rows: {len(records)}")

    if len(records) == 0:
        print("WARNING: No data collected. No dataset files generated.")
        return

    df = pd.DataFrame(records)

    # Save complete global dataset
    df.to_csv(output_file, index=False)
    print(f"Global CSV saved to: {output_file}")

    # Split by unique vehicle IDs
    unique_vehicle_ids = df["vehicle_id"].unique()

    for client_id, veh_id in enumerate(unique_vehicle_ids):
        vehicle_df = df[df["vehicle_id"] == veh_id].copy()

        client_file = os.path.join(dataset_dir, f"client_{client_id}.csv")
        vehicle_df.to_csv(client_file, index=False)

    print(f"Generated {len(unique_vehicle_ids)} client CSV files in: {dataset_dir}")


if __name__ == "__main__":
    main()