"""MCP weather server backed by Open-Meteo.

Run directly to serve tools over MCP stdio:

    python examples/providers/mcp_weather_server.py
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

try:
    from mcp.server.fastmcp import FastMCP  # type: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - optional dependency path
    FastMCP = None  # type: ignore[assignment]


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

mcp = FastMCP("rlmflow-weather", json_response=True) if FastMCP is not None else None


def mcp_tool(fn):
    if mcp is None:
        return fn
    return mcp.tool()(fn)


WEATHER_CODES = {
    0: "clear",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "rime fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "freezing drizzle",
    57: "heavy freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "heavy freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "rain showers",
    82: "heavy rain showers",
    85: "snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "heavy thunderstorm with hail",
}


def _json_get(url: str, params: dict[str, object]) -> dict[str, Any]:
    query = urlencode(params)
    with urlopen(f"{url}?{query}", timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _normalize_units(units: str) -> tuple[str, str, str, str]:
    normalized = units.lower()
    if normalized in {"f", "fahrenheit"}:
        return "fahrenheit", "mph", "temp_f", "wind_mph"
    if normalized in {"c", "celsius"}:
        return "celsius", "kmh", "temp_c", "wind_kmh"
    raise ValueError("units must be 'fahrenheit' or 'celsius'")


def _condition(code: int | None) -> str:
    if code is None:
        return "unknown"
    return WEATHER_CODES.get(int(code), f"weather code {code}")


def _geocode_city(city: str) -> dict[str, Any]:
    data = _json_get(
        GEOCODING_URL,
        {
            "name": city,
            "count": 1,
            "language": "en",
            "format": "json",
        },
    )
    results = data.get("results") or []
    if not results:
        raise ValueError(f"Could not find coordinates for city: {city}")
    result = results[0]
    return {
        "name": result["name"],
        "country": result.get("country"),
        "latitude": result["latitude"],
        "longitude": result["longitude"],
    }


@mcp_tool
def get_current_weather(city: str, units: str = "fahrenheit") -> dict[str, Any]:
    """Return current weather for a city using Open-Meteo."""

    temp_unit, wind_unit, temp_key, wind_key = _normalize_units(units)
    place = _geocode_city(city)
    data = _json_get(
        FORECAST_URL,
        {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,weather_code,wind_speed_10m",
            "temperature_unit": temp_unit,
            "wind_speed_unit": wind_unit,
            "timezone": "auto",
        },
    )
    current = data.get("current") or {}
    return {
        "city": place["name"],
        "country": place["country"],
        "time": current.get("time"),
        temp_key: current.get("temperature_2m"),
        "condition": _condition(current.get("weather_code")),
        wind_key: current.get("wind_speed_10m"),
    }


@mcp_tool
def get_forecast(city: str, days: int = 3, units: str = "fahrenheit") -> list[dict[str, Any]]:
    """Return a daily forecast for a city using Open-Meteo."""

    if days < 1 or days > 7:
        raise ValueError("days must be between 1 and 7")

    temp_unit, wind_unit, temp_key, wind_key = _normalize_units(units)
    place = _geocode_city(city)
    data = _json_get(
        FORECAST_URL,
        {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,wind_speed_10m_max",
            "forecast_days": days,
            "temperature_unit": temp_unit,
            "wind_speed_unit": wind_unit,
            "timezone": "auto",
        },
    )
    daily = data.get("daily") or {}
    rows: list[dict[str, Any]] = []
    for i, date in enumerate(daily.get("time") or []):
        rows.append(
            {
                "city": place["name"],
                "country": place["country"],
                "date": date,
                f"{temp_key}_high": daily.get("temperature_2m_max", [None])[i],
                f"{temp_key}_low": daily.get("temperature_2m_min", [None])[i],
                "condition": _condition((daily.get("weather_code") or [None])[i]),
                f"{wind_key}_max": daily.get("wind_speed_10m_max", [None])[i],
            }
        )
    return rows


def build_server() -> Any:
    if mcp is None:  # pragma: no cover - exercised by example smoke skips
        raise RuntimeError(
            "The MCP weather server requires the `mcp` package. "
            'Install with `pip install -e ".[mcp]"`.'
        )
    return mcp


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
