import numpy as np

def simulate_rocket_impact_2d(
    angle_deg,           # launch angle (above horizontal)
    rocket_mass,         # total mass at launch, including fuel
    fuel_mass,           # fuel mass portion
    exhaust_velocity,    # v_e (m/s)
    burn_time,           # duration of thrust (s)
    dt=0.01,             # integration timestep
    g=9.81,               # gravity (m/s^2)
    C_d=0.5,             # drag coefficient (dimensionless)
    A=0.1,               # frontal area (m^2)
    rho=1.225            # air density (kg/m^3)
):
    """
    Returns an approximate (x, y, z=0) impact point for the rocket,
    under these assumptions:
      - Phase 1: constant thrust from t=0..burn_time
      - Phase 2: ballistic (no thrust)
      - Includes air drag
      - Euler method integration
      - 2D: we treat y=0, so the rocket moves in (x,z).
    """

    # Convert angle to radians
    alpha = np.radians(angle_deg)

    # Mass flow rate
    if burn_time > 0:
        mdot = fuel_mass / burn_time
    else:
        mdot = 0.0

    # Thrust (N)
    T = mdot * exhaust_velocity

    # initial states
    t = 0.0
    x = 0.0
    z = 0.0
    vx = 0.0
    vz = 0.0
    mass = rocket_mass

    # integration
    while True:
        # if rocket is below ground AFTER burn, break
        if (z <= 0.0) and (t > burn_time):
            break

        # check if still burning
        if t < burn_time:
            thrust_x = T * np.cos(alpha)
            thrust_z = T * np.sin(alpha)
            dm = -mdot * dt
        else:
            thrust_x = 0.0
            thrust_z = 0.0
            dm = 0.0

        # Calculate the rocket's velocity magnitude
        velocity = np.sqrt(vx ** 2 + vz ** 2)

        # Drag force
        F_d = 0.5 * rho * C_d * A * velocity ** 2

        # Decompose drag force into components
        if velocity > 0:
            F_d_x = F_d * (vx / velocity)
            F_d_z = F_d * (vz / velocity)
        else:
            F_d_x = 0.0
            F_d_z = 0.0

        # accelerations
        ax = (thrust_x - F_d_x) / mass
        az = (thrust_z - F_d_z - g * mass) / mass

        # update velocity
        vx_new = vx + ax * dt
        vz_new = vz + az * dt

        # update position
        x_new = x + vx * dt
        z_new = z + vz * dt

        # update mass
        mass_new = mass + dm
        # clamp to dry mass (rocket_mass - fuel_mass)
        dry_mass = rocket_mass - fuel_mass
        if mass_new < dry_mass:
            mass_new = dry_mass

        # increment time
        t += dt

        # store updated states
        x, z = x_new, z_new
        vx, vz = vx_new, vz_new
        mass = mass_new

        # safety cutoff
        if t > burn_time + 300:
            break

    # clamp final z to 0 if below ground
    if z < 0:
        z = 0

    return x

def identify_best_rocket(
    angle_deg,                # given launch angle above horizontal
    actual_x,                 # actual observed landing x
    rocket_specs_dict,        # { "RocketName": {"mass":..., "fuel_mass":..., "exhaust_velocity":..., "burn_time":...}, ...}
    dt=0.01
):
    """
    For each rocket, simulate final x (z=0).
    Compare to actual_x => compute absolute error in x.
    Return rocket name with lowest error.
    Also return a dict of {rocket_name: error}.
    """

    errors = {}
    best_rocket = None
    best_error = float('inf')

    for rocket_name, specs in rocket_specs_dict.items():
        # unpack specs
        m0 = specs["mass"]
        fmass = specs["fuel_mass"]
        ve = specs["exhaust_velocity"]
        btime = specs["burn_time"]

        # simulate
        sim_x= simulate_rocket_impact_2d(
            angle_deg=angle_deg,
            rocket_mass=m0,
            fuel_mass=fmass,
            exhaust_velocity=ve,
            burn_time=btime,
            dt=dt
        )
        # error in x
        err_x = abs(sim_x - actual_x)

        errors[rocket_name] = err_x


    return errors


def identify_rocket(angle_deg:float, actual_x:float, rocket_specs_dict:dict, dt=0.01):

    angle_deg = 45.0
    actual_x = 1000.0

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
        # ... add more ...
    }

    errors, best_rocket, best_err = identify_best_rocket(
        angle_deg=angle_deg,
        actual_x=actual_x,
        rocket_specs_dict=rocket_specs,
        dt=0.01
    )

    # Print results
    print("Errors in X for each rocket:")
    for rname, errval in errors.items():
        print(f"  {rname}: {errval:.2f} m")

    print(f"\nBest rocket: {best_rocket}, with error={best_err:.2f} m")

def make_rocket_data():
    rockets_data={}
    rockets_data["Qassam4"] = {
    "mass": 50.0,
    "fuel_mass": 30.0,
    "exhaust_velocity": 1200.0,
    "burn_time": 0.5
    }
    rockets_data["Grad"] = {
        "mass": 300.0,
        "fuel_mass": 140.0,
        "exhaust_velocity": 2000.0,
        "burn_time": 2.0
    }
    rockets_data["M75"] = {
        "mass": 900.0,
        "fuel_mass": 140.0,
        "exhaust_velocity": 2000.0,
        "burn_time": 2.0
    }
    rockets_data["R160"] = {
        "mass": 300.0,
        "fuel_mass": 140.0,
        "exhaust_velocity": 2000.0,
        "burn_time": 2.0
    }
    return rockets_data

if __name__ == "__main__":
    rockets_data=make_rocket_data()


