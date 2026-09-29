"""Solar geometry over Truist Park during Luzardo's expected outing window.

NOAA solar-position equations (Spencer/Meeus-simplified); accurate to ~0.1 deg,
plenty for shadow reasoning. Truist Park home plate approx 33.8908 N, 84.4678 W.
"""
import math
from datetime import datetime, timedelta, timezone

LAT, LON = 33.8908, -84.4678


def solar_position(dt_utc: datetime):
    doy = dt_utc.timetuple().tm_yday
    hour = dt_utc.hour + dt_utc.minute / 60 + dt_utc.second / 3600
    g = 2 * math.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                       - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
            - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
            - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    tst = hour * 60 + eqtime + 4 * LON  # true solar time, minutes
    ha = math.radians(tst / 4 - 180)
    lat = math.radians(LAT)
    cos_zen = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(ha)
    zen = math.acos(max(-1, min(1, cos_zen)))
    elev = 90 - math.degrees(zen)
    az = math.degrees(math.atan2(math.sin(ha), math.cos(ha) * math.sin(lat) - math.tan(decl) * math.cos(lat))) + 180
    return elev, az % 360, eqtime, math.degrees(decl)


if __name__ == "__main__":
    edt = timezone(timedelta(hours=-4))
    start = datetime(2026, 9, 29, 13, 30, tzinfo=edt)
    print("time EDT | sun elevation | azimuth (deg from N) | shadow length per 10 m of height")
    for m in range(0, 241, 15):
        t = start + timedelta(minutes=m)
        e, a, eq, d = solar_position(t.astimezone(timezone.utc))
        sl = 10 / math.tan(math.radians(e)) if e > 0 else float('inf')
        print(f"{t:%H:%M} | {e:5.1f} | {a:5.1f} | {sl:5.1f} m")
    # solar noon
    best = None
    for m in range(0, 24 * 60):
        t = datetime(2026, 9, 29, 0, 0, tzinfo=edt) + timedelta(minutes=m)
        e, a, *_ = solar_position(t.astimezone(timezone.utc))
        if best is None or e > best[1]:
            best = (t, e)
    print("solar noon ~", best[0].strftime('%H:%M EDT'), f"max elevation {best[1]:.1f}")
