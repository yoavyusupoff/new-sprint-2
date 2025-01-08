import numpy as np

def simulate_rocket_impact(
    launch_position,     # [x0, y0, z0] in meters
    launch_angle_deg,    # angle above horizontal (2D case)
    rocket_mass,         # total mass at launch (kg)
    fuel_mass,           # mass of fuel (kg)
    exhaust_velocity,    # v_e in m/s
    burn_time,           # in s
    dt=0.01,             # time-step for numerical integration
    g=9.81               # gravity in m/s^2
):
    """
    Returns an approximate (x, y, z=0) impact point for the rocket,
    under these assumptions:
      - Phase 1: constant thrust from t=0..burn_time
      - Phase 2: ballistic (no thrust)
      - no drag
      - Euler method integration
      - 2D: we treat y=0, so the rocket moves in (x,z).
    """

    # Unpack launch position
    x0, y0, z0 = launch_position

    # Convert angle to radians
    alpha = np.radians(launch_angle_deg)

    # rocket_mass includes fuel_mass, so final (dry) mass = (rocket_mass - fuel_mass)
    if burn_time > 0:
        mdot = fuel_mass / burn_time  # kg/s
    else:
        mdot = 0.0

    # Thrust magnitude (N)
    T = mdot * exhaust_velocity

    # Initial conditions
    t = 0.0
    x = x0
    y = y0
    z = z0
    vx = 0.0
    vy = 0.0
    vz = 0.0
    mass = rocket_mass

    while True:
        # If rocket is below ground AFTER burn, stop
        if (z <= 0.0) and (t > burn_time):
            break

        # Check if rocket is still burning
        if t < burn_time:
            thrust_x = T * np.cos(alpha)
            thrust_z = T * np.sin(alpha)
            dm = -mdot * dt
        else:
            # ballistic
            thrust_x = 0.0
            thrust_z = 0.0
            dm = 0.0

        # accelerations
        ax = thrust_x / mass
        az = thrust_z / mass - g
        # we ignore y, so ay=0

        # Euler update
        vx_new = vx + ax * dt
        vz_new = vz + az * dt

        x_new = x + vx * dt
        z_new = z + vz * dt

        mass_new = mass + dm
        # clamp mass so it doesn't go below (rocket_mass - fuel_mass)
        if mass_new < (rocket_mass - fuel_mass):
            mass_new = (rocket_mass - fuel_mass)

        # increment time
        t += dt

        # update states
        x, z = x_new, z_new
        vx, vz = vx_new, vz_new
        mass = mass_new

        # safety cutoff if it never lands
        if t > burn_time + 300:
            break

    # clamp final z to 0 if below ground
    if z < 0.0:
        z = 0.0

    # ignoring y in this 2D approach
    return (x, 0.0, z)


def identify_rocket_type(
    launch_position,
    angle_deg,
    observed_impact,        # (x_obs, y_obs, z_obs=0)
    rocket_specs_dict,      # dict: rocket_name -> {mass, fuel_mass, exhaust_velocity, burn_time}
    dt=0.01
):
    """
    Simulate each rocket's impact, measure error from observed_impact,
    and return:
      - a dict of {rocket_name: {"impact": (x,y,z), "error": dist_error}}
      - best_rocket, best_error, best_impact
    """
    x_obs, y_obs, z_obs = observed_impact

    results = {}  # to store each rocket's impact + error

    best_rocket = None
    best_error = float('inf')
    best_impact = (None, None, None)

    for rocket_name, specs in rocket_specs_dict.items():
        m0 = specs["mass"]
        mf = specs["fuel_mass"]
        ve = specs["exhaust_velocity"]
        burn = specs["burn_time"]

        # Simulate impact
        impact_x, impact_y, impact_z = simulate_rocket_impact(
            launch_position=launch_position,
            launch_angle_deg=angle_deg,
            rocket_mass=m0,
            fuel_mass=mf,
            exhaust_velocity=ve,
            burn_time=burn,
            dt=dt
        )

        dx = impact_x - x_obs
        dy = impact_y - y_obs
        dz = impact_z - z_obs
        dist_error = np.sqrt(dx*dx + dy*dy + dz*dz)

        results[rocket_name] = {
            "impact": (impact_x, impact_y, impact_z),
            "error": dist_error
        }

        # track best
        if dist_error < best_error:
            best_error = dist_error
            best_rocket = rocket_name
            best_impact = (impact_x, impact_y, impact_z)

    return results, best_rocket, best_error, best_impact


def demo_identification():
    """
    Example usage of identify_rocket_type, printing out:
      - The impact location for each rocket
      - The best rocket match
    """
    # Suppose the rocket was observed impacting at (1000,0,0)
    observed_impact = (1000.0, 0.0, 0.0)

    # Launch from (0,0,0) at angle=45
    launch_position = (0.0, 0.0, 0.0)
    angle_deg = 45.0

    # Hypothetical rocket specs
    rocket_specs = {
        "Qassam4": {
            "mass": 50.0,
            "fuel_mass": 30.0,
            "exhaust_velocity": 1200.0,
            "burn_time": 0.5
        },
        "Grad": {
            "mass": 280.0,
            "fuel_mass": 120.0,
            "exhaust_velocity": 2000.0,
            "burn_time": 2.0
        },
        "M75": {
            "mass": 900.0,
            "fuel_mass": 500.0,
            "exhaust_velocity": 1620.0,
            "burn_time": 4.0
        },
        "R160": {
            "mass": 1670.0,
            "fuel_mass": 1000.0,
            "exhaust_velocity": 3000.0,
            "burn_time": 10.0
        }
    }

    results, best_rocket, best_err, best_impact = identify_rocket_type(
        launch_position=launch_position,
        angle_deg=angle_deg,
        observed_impact=observed_impact,
        rocket_specs_dict=rocket_specs,
        dt=0.01
    )

    # Print each rocket's results
    for rckt, info in results.items():
        print(f"Rocket: {rckt}")
        print(f"  Impact: {info['impact']}")
        print(f"  Error from observed: {info['error']:.2f} m")

    print("\n--- Best match rocket ---")
    print(f"Rocket: {best_rocket}")
    print(f"Impact: {best_impact}, error={best_err:.2f} m")


if __name__ == "__main__":
    demo_identification()
