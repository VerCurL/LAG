"""Portable CAS conversion for the independent controller package."""
import math

def calibrated_airspeed_mps(plane, feet_to_meters):
    """Prefer measured CAS. Otherwise estimate static pressure with ISA from height.

    The official socket table supplies Mach and altitude, but does not list CAS
    or static pressure. This fallback assumes standard atmosphere; it does not
    substitute ground speed or true airspeed for calibrated airspeed.
    """
    if "velocities/vc-fps" in plane:
        return float(plane["velocities/vc-fps"])*feet_to_meters
    if "velocities/vc-kts" in plane:
        return float(plane["velocities/vc-kts"])*1852.0/3600.0
    if "velocities/mach" not in plane:
        raise ValueError("Baseline flight controller needs velocities/vc-fps, vc-kts or mach")
    mach = max(0.0, float(plane["velocities/mach"]))
    height = float(plane["position/h-sl-ft"])*feet_to_meters
    # ISA geopotential height, troposphere and lower isothermal stratosphere.
    height = 6356766.0*height/(6356766.0+height)
    troposphere = min(height, 11000.0)
    temperature = 288.15-0.0065*troposphere
    pressure_ratio = (temperature/288.15)**(9.80665/(287.05287*0.0065))
    if height > 11000.0:
        pressure_ratio *= math.exp(-9.80665*(height-11000.0)/(287.05287*temperature))

    def impact_ratio(value):
        if value <= 1.0:
            return (1.0+0.2*value*value)**3.5-1.0
        # Rayleigh pitot relation includes the normal shock at supersonic speed.
        return (1.2*value*value)**3.5*(2.4/(2.8*value*value-0.4))**2.5-1.0

    ratio = pressure_ratio*impact_ratio(mach)
    if ratio <= impact_ratio(1.0):
        calibrated_mach = math.sqrt(max(0.0, 5.0*((ratio+1.0)**(2.0/7.0)-1.0)))
    else:
        lower, upper = 1.0, max(2.0, mach)
        while impact_ratio(upper) < ratio:
            upper *= 2.0
        for _ in range(48):
            middle = (lower+upper)/2.0
            if impact_ratio(middle) < ratio:
                lower = middle
            else:
                upper = middle
        calibrated_mach = (lower+upper)/2.0
    return calibrated_mach*math.sqrt(1.4*287.05287*288.15)
