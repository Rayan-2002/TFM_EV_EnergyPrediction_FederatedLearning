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

# ============================================================
# SIMULATION SETTINGS
# ============================================================

STEP_LENGTH = 1.0 # seconds
SIMULATION_END = 20 * 60 # 20 minutes 

PROJECT_DIR = os.path.expanduser("~/Internship/SUMO_Barcelona")

CONFIG_FILE = os.path.join(
    PROJECT_DIR,
    "simulations",
    "simulation_2004",
    "sim.sumocfg"
)

OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")

RAW_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "vehicle_log_raw.csv")
CLEAN_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "vehicle_log.csv")
CLIENT_MAPPING_FILE = os.path.join(DATASET_DIR, "client_mapping.csv")


# Physics constants from the proposal using Tesla Model 3 as reference.

MASS = 1823.0                 # kg
GRAVITY = 9.81                # m/s²
DRIVETRAIN_EFFICIENCY = 0.90  # eta
ROLLING_COEFF = 0.01          # C_r
AIR_DENSITY = 1.225           # rho, kg/m³
DRAG_COEFF = 0.225             # C_d
FRONTAL_AREA = 2.2            # A, m²
AUXILIARY_POWER = 1000      # P_aux, watts
# ROAD_GRADE_ANGLE = 0.0      theta, radians. This will be obtained dynamically from SUMO's getSlope() method.



# ============================================================
# DATA CLEANING SETTINGS
# ============================================================

ACCELERATION_CLIP = 5.0

IDLE_SPEED_THRESHOLD = 0.1 # m/s
IDLE_ACCEL_THRESHOLD = 0.05         # m/s²
MAX_IDLE_SECONDS_PER_STREAK = 10.0  # keep max 10 seconds from each long idle streak

TIME_GAP_TOLERANCE = 1e-6


# Model feature columns

MODEL_FEATURE_COLUMNS = [
    "speed",
    "acceleration",
    "road_grade_angle",
    "consumption_power_watts" # The ML target variable for energy consumption prediction. We include it as a feature here for completeness, but it will be the target variable during model training.
]


# ============================================================
# POWER COMPUTATION
# ============================================================


def compute_power(speed, acceleration, road_grade_angle):
    """
    Compute vehicle power.

    signed_power:
        Can be negative during braking/regeneration.

    consumption_power:
        Non-negative energy-consumption target.
        It never goes below AUXILIARY_POWER.

    Formula:

        traction_power = v / eta * total_force

        signed_power = traction_power + P_aux

        consumption_power = max(traction_power, 0) + P_aux
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

    traction_power = (speed / DRIVETRAIN_EFFICIENCY) * total_force

    signed_power = traction_power + AUXILIARY_POWER

    consumption_power = max(signed_power, 0.0) + AUXILIARY_POWER

    return signed_power, consumption_power


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clip_value(value, min_value, max_value):
    return max(min(value, max_value), min_value)


def clean_previous_dataset_files(dataset_dir):
    """
    Remove previous generated CSV files from the dataset directory.
    """
    if not os.path.exists(dataset_dir):
        return

    for filename in os.listdir(dataset_dir):
        if filename.endswith(".csv"):
            os.remove(os.path.join(dataset_dir, filename))


def reduce_long_idle_streaks(df):
    """
    Reduce long idle periods.

    A row is considered idle when:
        speed < IDLE_SPEED_THRESHOLD
        abs(acceleration) < IDLE_ACCEL_THRESHOLD

    For each vehicle, each continuous idle streak is reduced to at most
    MAX_IDLE_SECONDS_PER_STREAK seconds.

    This prevents the model from being dominated by many repeated rows like:

        speed = 0
        acceleration = 0
        consumption_power_watts = 1000
    """

    max_idle_rows = max(1, int(round(MAX_IDLE_SECONDS_PER_STREAK / STEP_LENGTH)))

    cleaned_groups = []
    total_removed = 0

    for veh_id, group in df.groupby("vehicle_id", sort=False):
        group = group.sort_values("time").copy()

        group["_is_idle"] = (
            (group["speed"] < IDLE_SPEED_THRESHOLD)
            & (group["acceleration"].abs() < IDLE_ACCEL_THRESHOLD)
        )

        # New streak whenever idle/non-idle state changes
        group["_streak_id"] = (
            group["_is_idle"] != group["_is_idle"].shift(fill_value=False)
        ).cumsum()

        group["_position_in_streak"] = group.groupby("_streak_id").cumcount()

        keep_mask = (
            (~group["_is_idle"])
            | (group["_position_in_streak"] < max_idle_rows)
        )

        removed = len(group) - keep_mask.sum()
        total_removed += removed

        group = group.loc[keep_mask].copy()

        group = group.drop(
            columns=[
                "_is_idle",
                "_streak_id",
                "_position_in_streak"
            ]
        )

        cleaned_groups.append(group)

    if len(cleaned_groups) == 0:
        return df.copy(), 0

    cleaned_df = pd.concat(cleaned_groups, ignore_index=True)
    cleaned_df = cleaned_df.sort_values(["vehicle_id", "time"]).reset_index(drop=True)

    return cleaned_df, total_removed


def add_sequence_segments(df):
    """
    Add a segment_id column.

    This is important because reducing long idle streaks can create time gaps.

    Later, when building the PyTorch Dataset, sliding windows should be created
    inside each:

        vehicle_id + segment_id

    and never across two different segments.
    """

    segmented_groups = []

    for veh_id, group in df.groupby("vehicle_id", sort=False):
        group = group.sort_values("time").copy()

        time_diff = group["time"].diff()

        new_segment = (
            time_diff.isna()
            | ((time_diff - STEP_LENGTH).abs() > TIME_GAP_TOLERANCE)
        )

        group["segment_id"] = new_segment.cumsum() - 1

        segmented_groups.append(group)

    segmented_df = pd.concat(segmented_groups, ignore_index=True)
    segmented_df = segmented_df.sort_values(
        ["vehicle_id", "segment_id", "time"]
    ).reset_index(drop=True)

    return segmented_df


def print_diagnostics(df, title):
    """
    Print useful diagnostics after data generation and cleaning.
    """

    print("\n" + "=" * 70)
    print(f"DIAGNOSTICS: {title}")
    print("=" * 70)

    if len(df) == 0:
        print("No rows available.")
        return

    numeric_cols = [
        "speed",
        "acceleration",
        "road_grade_angle",
        "signed_power_watts",
        "consumption_power_watts"
    ]

    numeric_cols = [col for col in numeric_cols if col in df.columns]

    print("\n--- Numeric summary ---")
    print(df[numeric_cols].describe())

    print("\n--- Missing values ---")
    print(df.isna().sum())

    idle_mask = (
        (df["speed"] < IDLE_SPEED_THRESHOLD)
        & (df["acceleration"].abs() < IDLE_ACCEL_THRESHOLD)
    )

    print("\n--- Idle ratio ---")
    print(f"Idle rows: {idle_mask.mean() * 100:.2f}%")

    print("\n--- Auxiliary-only ratio ---")
    aux_only = (df["consumption_power_watts"] - AUXILIARY_POWER).abs() < 1e-6
    print(f"Auxiliary-only rows: {aux_only.mean() * 100:.2f}%")

    print("\n--- Zero consumption ratio ---")
    zero_consumption = df["consumption_power_watts"] == 0
    print(f"Zero consumption rows: {zero_consumption.mean() * 100:.2f}%")

    if "signed_power_watts" in df.columns:
        print("\n--- Negative signed power ratio ---")
        negative_signed = df["signed_power_watts"] < 0
        print(f"Negative signed power rows: {negative_signed.mean() * 100:.2f}%")

    print("\n--- Rows per vehicle ---")
    print(df.groupby("vehicle_id").size().describe())

    if "segment_id" in df.columns:
        print("\n--- Segments per vehicle ---")
        print(df.groupby("vehicle_id")["segment_id"].nunique().describe())

    print("\n--- Most common time differences ---")
    time_diffs = df.groupby("vehicle_id")["time"].diff()
    print(time_diffs.value_counts().head(10))


# ============================================================
# MAIN SCRIPT
# ============================================================


def main():

    print(f"Using SUMO config: {CONFIG_FILE}")
    print(f"Config exists: {os.path.exists(CONFIG_FILE)}")

    if not os.path.exists(CONFIG_FILE):
        print(f"ERROR: SUMO config file not found: {CONFIG_FILE}")
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)

    clean_previous_dataset_files(DATASET_DIR)

    sumo_binary = sumolib.checkBinary("sumo")

    sumo_cmd = [
        sumo_binary,
        "-c", CONFIG_FILE,
        "--no-step-log", "true",
        "--duration-log.disable", "true",
        "--step-length", str(STEP_LENGTH),
        "--end", str(SIMULATION_END) 
    ]

    print("\nStarting SUMO with TraCI...")
    print("SUMO command:")
    print(" ".join(sumo_cmd))

    traci.start(sumo_cmd)

    records = []
    step_count = 0

    acceleration_clip_count = 0
    negative_speed_count = 0

    try:
        while (
            traci.simulation.getMinExpectedNumber() > 0
            and traci.simulation.getTime() < SIMULATION_END
        ):
            traci.simulationStep()
            step_count += 1

            current_time = traci.simulation.getTime()
            vehicle_ids = traci.vehicle.getIDList()

            if step_count % 10 == 0:
                print(
                    f"Step {step_count}, "
                    f"time={current_time:.1f}, "
                    f"active vehicles={len(vehicle_ids)}, "
                    f"records={len(records)}"
                )

            for veh_id in vehicle_ids:
                speed = traci.vehicle.getSpeed(veh_id)
                raw_acceleration = traci.vehicle.getAcceleration(veh_id)

                # Preserve sequence continuity by clipping instead of dropping rows
                if speed < 0:
                    negative_speed_count += 1
                    speed = 0.0

                acceleration = clip_value(
                    raw_acceleration,
                    -ACCELERATION_CLIP,
                    ACCELERATION_CLIP
                )

                if acceleration != raw_acceleration:
                    acceleration_clip_count += 1

                slope_degrees = traci.vehicle.getSlope(veh_id)
                road_grade_angle = math.radians(slope_degrees)

                signed_power, consumption_power = compute_power(
                    speed=speed,
                    acceleration=acceleration,
                    road_grade_angle=road_grade_angle
                )

                records.append({
                    "time": current_time,
                    "vehicle_id": veh_id,

                    # ML features
                    "speed": speed,
                    "acceleration": acceleration,
                    "road_grade_angle": road_grade_angle,
                    "consumption_power_watts": consumption_power,

                    # Debug/diagnostic columns
                    "slope_degrees": slope_degrees,
                    "signed_power_watts": signed_power
                })

    finally:
        traci.close()

    print("\nSimulation finished.")
    print(f"Total simulation steps: {step_count}")
    print(f"Total logged rows before cleaning: {len(records)}")
    print(f"Acceleration values clipped: {acceleration_clip_count}")
    print(f"Negative speed values corrected to zero: {negative_speed_count}")

    if len(records) == 0:
        print("WARNING: No data collected. No dataset files generated.")
        return

    # ========================================================
    # CREATE RAW DATAFRAME
    # ========================================================

    df_raw = pd.DataFrame(records)
    df_raw = df_raw.sort_values(["vehicle_id", "time"]).reset_index(drop=True)

    df_raw.to_csv(RAW_OUTPUT_FILE, index=False)
    print(f"\nRaw global CSV saved to: {RAW_OUTPUT_FILE}")

    print_diagnostics(df_raw, "RAW DATA BEFORE IDLE REDUCTION")

    # ========================================================
    # REDUCE LONG IDLE STREAKS
    # ========================================================

    df_clean, removed_idle_rows = reduce_long_idle_streaks(df_raw)

    print(f"\nRemoved long-idle rows: {removed_idle_rows}")
    print(f"Rows after idle reduction: {len(df_clean)}")

    # Add sequence segments after dropping idle rows
    df_clean = add_sequence_segments(df_clean)

    print_diagnostics(df_clean, "CLEAN DATA AFTER IDLE REDUCTION")

    # ========================================================
    # SAVE CLEAN GLOBAL DATASET
    # ========================================================

    df_clean.to_csv(CLEAN_OUTPUT_FILE, index=False)
    print(f"\nClean global CSV saved to: {CLEAN_OUTPUT_FILE}")

    # ========================================================
    # CREATE ONE CLIENT CSV PER VEHICLE
    # ========================================================

    client_columns = [
        "time",
        "vehicle_id",
        "segment_id"
    ] + MODEL_FEATURE_COLUMNS

    unique_vehicle_ids = df_clean["vehicle_id"].unique()

    mapping_records = []

    for client_id, veh_id in enumerate(unique_vehicle_ids):
        vehicle_df = df_clean[df_clean["vehicle_id"] == veh_id].copy()
        vehicle_df = vehicle_df.sort_values(["segment_id", "time"])

        client_file = os.path.join(DATASET_DIR, f"client_{client_id}.csv")

        vehicle_df[client_columns].to_csv(client_file, index=False)

        idle_mask = (
            (vehicle_df["speed"] < IDLE_SPEED_THRESHOLD)
            & (vehicle_df["acceleration"].abs() < IDLE_ACCEL_THRESHOLD)
        )

        mapping_records.append({
            "client_id": client_id,
            "vehicle_id": veh_id,
            "num_rows": len(vehicle_df),
            "num_segments": vehicle_df["segment_id"].nunique(),
            "start_time": vehicle_df["time"].min(),
            "end_time": vehicle_df["time"].max(),
            "idle_ratio_percent": idle_mask.mean() * 100
        })

    mapping_df = pd.DataFrame(mapping_records)
    mapping_df.to_csv(CLIENT_MAPPING_FILE, index=False)

    print(f"\nGenerated {len(unique_vehicle_ids)} client CSV files in: {DATASET_DIR}")
    print(f"Client mapping saved to: {CLIENT_MAPPING_FILE}")

    print("\nModel feature columns to use later:")
    for col in MODEL_FEATURE_COLUMNS:
        print(f"  - {col}")

    print("\nIMPORTANT:")
    print("Use only MODEL_FEATURE_COLUMNS as model inputs.")
    print("Do not use time, vehicle_id, or segment_id as neural network features.")
    print("Use vehicle_id + segment_id only to create valid sliding windows.")


if __name__ == "__main__":
    main()