# modules/weather_integration.py
# Complete Tomorrow.io Weather API Integration for Ghostline AI
# Supports comprehensive weather monitoring + health-focused tracking for headaches and UV sensitivity
# UPDATED WITH LOCATION DETECTION AND METHOD COMPATIBILITY FIXES

import os
import json
import requests
import datetime
import math
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import timedelta

# Import Ghostline dependencies
try:
    from utils.ghostline_engine import generate_response
    from modules.database import save_conversation_enhanced, save_daily_log_enhanced
    GHOSTLINE_INTEGRATION = True
except ImportError:
    GHOSTLINE_INTEGRATION = False

# Tomorrow.io API configuration
TOMORROW_IO_API_KEY = os.getenv("TOMORROW_IO_API_KEY", "")
TOMORROW_IO_BASE_URL = "https://api.tomorrow.io/v4"

# Default location (you can set this in your environment or modify as needed)
DEFAULT_LOCATION = os.getenv("DEFAULT_WEATHER_LOCATION", "38.8606,-77.2287")  # Merrifield, VA as default

# Weather alert thresholds
PRESSURE_DROP_THRESHOLD = float(os.getenv("PRESSURE_DROP_THRESHOLD", "3.0"))  # mbar drop that might trigger headaches
HIGH_UV_THRESHOLD = int(os.getenv("HIGH_UV_THRESHOLD", "6"))  # UV index considered high
VERY_HIGH_UV_THRESHOLD = int(os.getenv("VERY_HIGH_UV_THRESHOLD", "8"))  # UV index considered very high

# Weather data cache to avoid excessive API calls
_weather_cache = {}
_cache_duration = int(os.getenv("WEATHER_CACHE_DURATION", "1800"))  # 30 minutes in seconds

# Comprehensive weather fields for Tomorrow.io API
COMPREHENSIVE_WEATHER_FIELDS = [
    "temperature",
    "temperatureApparent",
    "pressureSurfaceLevel",
    "uvIndex",
    "uvHealthConcern",
    "humidity",
    "windSpeed",
    "windDirection",
    "windGust",
    "visibility",
    "cloudCover",
    "weatherCode",
    "precipitationIntensity",
    "precipitationProbability",
    "precipitationType"
]

# Comprehensive weather code mappings
WEATHER_CODES = {
    # Clear conditions
    1000: {"desc": "Clear, Sunny", "icon": "☀️", "category": "clear"},
    1100: {"desc": "Mostly Clear", "icon": "🌤️", "category": "clear"},
    1101: {"desc": "Partly Cloudy", "icon": "⛅", "category": "cloudy"},
    1102: {"desc": "Mostly Cloudy", "icon": "☁️", "category": "cloudy"},
    1001: {"desc": "Cloudy", "icon": "☁️", "category": "cloudy"},
    
    # Fog conditions
    2000: {"desc": "Fog", "icon": "🌫️", "category": "fog"},
    2100: {"desc": "Light Fog", "icon": "🌫️", "category": "fog"},
    
    # Rain conditions
    4000: {"desc": "Drizzle", "icon": "🌦️", "category": "rain"},
    4001: {"desc": "Rain", "icon": "🌧️", "category": "rain"},
    4200: {"desc": "Light Rain", "icon": "🌦️", "category": "rain"},
    4201: {"desc": "Heavy Rain", "icon": "⛈️", "category": "rain"},
    
    # Snow conditions
    5000: {"desc": "Snow", "icon": "❄️", "category": "snow"},
    5001: {"desc": "Flurries", "icon": "🌨️", "category": "snow"},
    5100: {"desc": "Light Snow", "icon": "🌨️", "category": "snow"},
    5101: {"desc": "Heavy Snow", "icon": "❄️", "category": "snow"},
    
    # Freezing conditions
    6000: {"desc": "Freezing Drizzle", "icon": "🧊", "category": "freezing"},
    6001: {"desc": "Freezing Rain", "icon": "🧊", "category": "freezing"},
    6200: {"desc": "Light Freezing Rain", "icon": "🧊", "category": "freezing"},
    6201: {"desc": "Heavy Freezing Rain", "icon": "🧊", "category": "freezing"},
    
    # Ice conditions
    7000: {"desc": "Ice Pellets", "icon": "🧊", "category": "ice"},
    7101: {"desc": "Heavy Ice Pellets", "icon": "🧊", "category": "ice"},
    7102: {"desc": "Light Ice Pellets", "icon": "🧊", "category": "ice"},
    
    # Thunderstorm conditions
    8000: {"desc": "Thunderstorm", "icon": "⛈️", "category": "storm"}
}

@dataclass
class ComprehensiveWeatherData:
    """Comprehensive weather data structure for health-aware monitoring"""
    timestamp: datetime.datetime
    temperature: float
    feels_like: float
    pressure_surface_level: float
    uv_index: float
    uv_health_concern: str
    humidity: float
    wind_speed: float
    wind_direction: float
    wind_gust: float
    visibility: float
    cloud_cover: float
    weather_code: int
    precipitation_intensity: float
    precipitation_probability: float
    precipitation_type: int
    location: str
    
    # Derived properties
    weather_description: str = ""
    weather_icon: str = ""
    weather_category: str = ""
    pressure_trend: str = ""
    uv_alert_level: str = ""


class TomorrowIOClient:
    """Tomorrow.io API client with error handling and rate limiting"""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "accept": "application/json",
            "content-type": "application/json"
        })
        if not api_key:
            raise ValueError("Tomorrow.io API key is required")
    
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> requests.Response:
        """Make API request with error handling"""
        url = f"{self.base_url}/{endpoint}"
        params["apikey"] = self.api_key
        
        try:
            print(f"🌐 Tomorrow.io API: Requesting {endpoint}")
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print("🚫 Tomorrow.io: Rate limit exceeded")
                raise Exception("Tomorrow.io API: Rate limit exceeded. Try again later.")
            elif e.response.status_code == 401:
                print("🔑 Tomorrow.io: Invalid API key")
                raise Exception("Tomorrow.io API: Invalid API key. Check your configuration.")
            else:
                print(f"🚨 Tomorrow.io HTTP Error: {e.response.status_code}")
                raise Exception(f"Tomorrow.io API error: {e.response.status_code}")
                
        except requests.exceptions.Timeout:
            print("⏱️  Tomorrow.io: Request timeout")
            raise Exception("Tomorrow.io API: Request timeout. Try again later.")
        except requests.exceptions.ConnectionError:
            print("🌐 Tomorrow.io: Connection failed")
            raise Exception("Tomorrow.io API: Connection failed. Check your internet connection.")
        except Exception as e:
            print(f"🚨 Tomorrow.io: Unexpected error: {e}")
            raise
    
    def get_comprehensive_weather(self, location: str = None) -> dict:
        """Get comprehensive current weather including all the good bits"""
        location = location or DEFAULT_LOCATION
        
        params = {
            "location": location,
            "fields": ",".join(COMPREHENSIVE_WEATHER_FIELDS)
        }
        
        response = self._make_request("weather/realtime", params)
        return response.json()
    
    def get_comprehensive_forecast(self, location: str = None, hours: int = 24) -> dict:
        """Get comprehensive forecast with all weather details"""
        location = location or DEFAULT_LOCATION
        
        params = {
            "location": location,
            "fields": ",".join(COMPREHENSIVE_WEATHER_FIELDS),
            "timesteps": "1h",
            "endTime": (datetime.datetime.now() + datetime.timedelta(hours=hours)).isoformat() + "Z"
        }
        
        response = self._make_request("weather/forecast", params)
        return response.json()


class WeatherMonitor:
    """Weather monitoring system for health-related alerts and comprehensive weather info"""
    
    def __init__(self):
        self.client = TomorrowIOClient(TOMORROW_IO_API_KEY, TOMORROW_IO_BASE_URL)
        self._load_pressure_history()
    
    def _load_pressure_history(self):
        """Load pressure history for trend analysis"""
        try:
            with open("sessions/weather_history.json", "r") as f:
                self.pressure_history = json.load(f)
        except FileNotFoundError:
            self.pressure_history = []
    
    def _save_pressure_history(self):
        """Save pressure history"""
        try:
            os.makedirs("sessions", exist_ok=True)
            with open("sessions/weather_history.json", "w") as f:
                json.dump(self.pressure_history[-100:], f)  # Keep last 100 readings
        except Exception as e:
            print(f"⚠️  Failed to save pressure history: {e}")
    
    def _check_comprehensive_cache(self, location: str) -> Optional[ComprehensiveWeatherData]:
        """Check if we have cached comprehensive weather data"""
        cache_key = f"comprehensive_weather_{location}"
        if cache_key in _weather_cache:
            cached_data, cached_time = _weather_cache[cache_key]
            if (datetime.datetime.now() - cached_time).seconds < _cache_duration:
                return cached_data
        return None
    
    def _cache_comprehensive_weather_data(self, location: str, weather_data: ComprehensiveWeatherData):
        """Cache comprehensive weather data"""
        cache_key = f"comprehensive_weather_{location}"
        _weather_cache[cache_key] = (weather_data, datetime.datetime.now())
    
    def get_comprehensive_conditions(self, location: str = None) -> ComprehensiveWeatherData:
        """Get comprehensive current weather conditions"""
        location = location or DEFAULT_LOCATION
        
        # Check cache first
        cached = self._check_comprehensive_cache(location)
        if cached:
            print("📊 Using cached comprehensive weather data")
            return cached
        
        try:
            # Get comprehensive current conditions
            current_data = self.client.get_comprehensive_weather(location)
            values = current_data["data"]["values"]
            
            # Create comprehensive weather data object
            weather_data = ComprehensiveWeatherData(
                timestamp=datetime.datetime.now(),
                temperature=values.get("temperature", 0),
                feels_like=values.get("temperatureApparent", values.get("temperature", 0)),
                pressure_surface_level=values.get("pressureSurfaceLevel", 0),
                uv_index=values.get("uvIndex", 0),
                uv_health_concern=values.get("uvHealthConcern", "unknown"),
                humidity=values.get("humidity", 0),
                wind_speed=values.get("windSpeed", 0),
                wind_direction=values.get("windDirection", 0),
                wind_gust=values.get("windGust", 0),
                visibility=values.get("visibility", 0),
                cloud_cover=values.get("cloudCover", 0),
                weather_code=values.get("weatherCode", 1000),
                precipitation_intensity=values.get("precipitationIntensity", 0),
                precipitation_probability=values.get("precipitationProbability", 0),
                precipitation_type=values.get("precipitationType", 0),
                location=location
            )
            
            # Add derived properties
            weather_info = WEATHER_CODES.get(weather_data.weather_code, {
                "desc": f"Unknown condition {weather_data.weather_code}",
                "icon": "🌦️",
                "category": "unknown"
            })
            
            weather_data.weather_description = weather_info["desc"]
            weather_data.weather_icon = weather_info["icon"]
            weather_data.weather_category = weather_info["category"]
            
            # Analyze pressure trend
            weather_data.pressure_trend = self._analyze_pressure_trend(weather_data.pressure_surface_level)
            
            # Analyze UV level
            weather_data.uv_alert_level = self._analyze_uv_level(weather_data.uv_index)
            
            # Cache the data
            self._cache_comprehensive_weather_data(location, weather_data)
            
            # Update pressure history
            pressure_entry = {
                "timestamp": weather_data.timestamp.isoformat(),
                "pressure": weather_data.pressure_surface_level
            }
            self.pressure_history.append(pressure_entry)
            self._save_pressure_history()
            
            print(f"✅ Comprehensive weather data updated: {weather_data.weather_description}, {weather_data.temperature:.1f}°C")
            return weather_data
            
        except Exception as e:
            print(f"❌ Failed to get comprehensive weather data: {e}")
            raise
    
    def get_current_conditions(self, location: str = None) -> ComprehensiveWeatherData:
        """Alias for get_comprehensive_conditions for compatibility"""
        return self.get_comprehensive_conditions(location)
    
    def _analyze_pressure_trend(self, current_pressure: float) -> str:
        """Analyze pressure trend for headache prediction"""
        if not self.pressure_history:
            return "No trend data available"
        
        # Get pressure from 3-6 hours ago for trend analysis
        recent_pressures = [
            entry for entry in self.pressure_history[-20:]
            if entry["pressure"] is not None
        ]
        
        if len(recent_pressures) < 2:
            return "Insufficient data for trend analysis"
        
        # Calculate pressure change over recent period
        old_pressure = recent_pressures[0]["pressure"]
        pressure_change = current_pressure - old_pressure
        
        # Classify pressure trends with headache risk assessment
        if pressure_change <= -3.0:
            return f"Dropping rapidly ({pressure_change:.1f}mbar) - High headache risk"
        elif pressure_change <= -1.5:
            return f"Dropping ({pressure_change:.1f}mbar) - Moderate headache risk"
        elif pressure_change >= 3.0:
            return f"Rising rapidly ({pressure_change:.1f}mbar)"
        elif pressure_change >= 1.5:
            return f"Rising ({pressure_change:.1f}mbar)"
        else:
            return f"Stable ({pressure_change:.1f}mbar)"
    
    def _analyze_uv_level(self, uv_index: float) -> str:
        """Analyze UV levels for sun sensitivity"""
        if uv_index >= VERY_HIGH_UV_THRESHOLD:
            return "Very High - Avoid sun exposure"
        elif uv_index >= HIGH_UV_THRESHOLD:
            return "High - Sun protection required"
        elif uv_index >= 3:
            return "Moderate - Some protection advised"
        else:
            return "Low - Minimal risk"
    
    def get_health_alerts(self, weather_data: ComprehensiveWeatherData) -> List[str]:
        """Get health-related weather alerts"""
        alerts = []
        
        # Pressure-based headache alerts
        if "High headache risk" in weather_data.pressure_trend:
            alerts.append(f"🤕 High headache risk due to rapid pressure drop")
        elif "Moderate headache risk" in weather_data.pressure_trend:
            alerts.append(f"⚠️ Moderate headache risk from pressure changes")
        
        # UV-based sun sensitivity alerts
        if weather_data.uv_index >= VERY_HIGH_UV_THRESHOLD:
            alerts.append(f"☀️ Very high UV index ({weather_data.uv_index}) - Avoid prolonged sun exposure")
        elif weather_data.uv_index >= HIGH_UV_THRESHOLD:
            alerts.append(f"🕶️ High UV index ({weather_data.uv_index}) - Use sun protection")
        
        # Weather-specific health alerts
        if weather_data.weather_category == "storm":
            alerts.append("⛈️ Thunderstorm conditions may affect sensitive individuals")
        elif weather_data.humidity > 80 and weather_data.temperature > 25:
            alerts.append("💧 High humidity and temperature may cause discomfort")
        elif weather_data.visibility < 2:  # km
            alerts.append("🌫️ Poor visibility conditions - drive carefully")
        
        return alerts
    
    def format_simple_weather_summary(self, weather_data: ComprehensiveWeatherData) -> str:
        """Format a simple, readable weather summary"""
        temp_f = weather_data.temperature * 9/5 + 32
        feels_like_f = weather_data.feels_like * 9/5 + 32
        wind_mph = weather_data.wind_speed * 2.237
        visibility_miles = weather_data.visibility * 0.621371
        
        summary = f"🌦️ **Current Weather Summary**\n\n"
        summary += f"**{weather_data.weather_icon} {weather_data.weather_description}**\n"
        summary += f"• Temperature: {weather_data.temperature:.1f}°C ({temp_f:.0f}°F)\n"
        summary += f"• Feels like: {weather_data.feels_like:.1f}°C ({feels_like_f:.0f}°F)\n"
        summary += f"• Humidity: {weather_data.humidity:.0f}%\n"
        summary += f"• Wind: {wind_mph:.0f} mph from {self._get_wind_direction(weather_data.wind_direction)}\n"
        summary += f"• Visibility: {visibility_miles:.1f} miles\n\n"
        
        summary += f"**🏥 Health Monitoring:**\n"
        summary += f"• Barometric pressure: {weather_data.pressure_surface_level:.1f} mbar ({weather_data.pressure_trend})\n"
        summary += f"• UV index: {weather_data.uv_index:.1f} ({weather_data.uv_alert_level})\n"
        
        if weather_data.precipitation_probability > 20:
            summary += f"• Precipitation chance: {weather_data.precipitation_probability:.0f}%\n"
        
        return summary
    
    def format_detailed_weather_report(self, weather_data: ComprehensiveWeatherData) -> str:
        """Format a comprehensive weather report"""
        temp_f = weather_data.temperature * 9/5 + 32
        feels_like_f = weather_data.feels_like * 9/5 + 32
        wind_mph = weather_data.wind_speed * 2.237
        wind_gust_mph = weather_data.wind_gust * 2.237
        visibility_miles = weather_data.visibility * 0.621371
        
        report = f"🌦️ **Comprehensive Weather Report**\n\n"
        report += f"📍 **Location**: {weather_data.location}\n"
        report += f"🕐 **Time**: {weather_data.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += f"**🌡️ Temperature & Feel:**\n"
        report += f"• Current: {weather_data.temperature:.1f}°C ({temp_f:.0f}°F)\n"
        report += f"• Feels like: {weather_data.feels_like:.1f}°C ({feels_like_f:.0f}°F)\n"
        report += f"• Humidity: {weather_data.humidity:.0f}%\n\n"
        
        report += f"**{weather_data.weather_icon} Current Conditions:**\n"
        report += f"• {weather_data.weather_description}\n"
        report += f"• Cloud cover: {weather_data.cloud_cover:.0f}%\n"
        report += f"• Visibility: {visibility_miles:.1f} miles\n"
        
        if weather_data.precipitation_probability > 0:
            report += f"• Precipitation: {weather_data.precipitation_probability:.0f}% chance\n"
        
        report += f"\n**🌪️ Wind & Pressure:**\n"
        report += f"• Wind: {wind_mph:.0f} mph from {self._get_wind_direction(weather_data.wind_direction)}\n"
        if wind_gust_mph > wind_mph:
            report += f"• Gusts: up to {wind_gust_mph:.0f} mph\n"
        report += f"• Barometric pressure: {weather_data.pressure_surface_level:.1f} mbar\n"
        report += f"• Pressure trend: {weather_data.pressure_trend}\n\n"
        
        report += f"**🏥 Health Monitoring:**\n"
        report += f"• UV index: {weather_data.uv_index:.1f} ({weather_data.uv_alert_level})\n"
        report += f"• UV concern level: {weather_data.uv_health_concern}\n"
        
        return report
    
    def _get_wind_direction(self, degrees: float) -> str:
        """Convert wind direction degrees to compass direction"""
        directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                     "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        index = int((degrees + 11.25) / 22.5) % 16
        return directions[index]
    
    def answer_simple_weather_questions(self, question: str, weather_data: ComprehensiveWeatherData) -> str:
        """Answer simple weather questions intelligently"""
        q = question.lower()
        
        # Temperature questions
        if any(word in q for word in ["temp", "temperature", "hot", "cold", "warm", "cool"]):
            temp_f = weather_data.temperature * 9/5 + 32
            feels_like_f = weather_data.feels_like * 9/5 + 32
            
            if "feels like" in q or "feel" in q:
                return f"It feels like {feels_like_f:.0f}°F ({weather_data.feels_like:.1f}°C) outside."
            else:
                temp_desc = "warm" if temp_f > 75 else "cool" if temp_f < 60 else "comfortable"
                return f"It's currently {temp_f:.0f}°F ({weather_data.temperature:.1f}°C) - {temp_desc}."
        
        # Rain/precipitation questions
        if any(word in q for word in ["rain", "raining", "wet", "precipitation"]):
            if weather_data.precipitation_probability > 70:
                return f"Yes, there's a high chance of rain ({weather_data.precipitation_probability:.0f}%)."
            elif weather_data.precipitation_probability > 30:
                return f"Possibly - there's a {weather_data.precipitation_probability:.0f}% chance of rain."
            else:
                return f"No rain expected (only {weather_data.precipitation_probability:.0f}% chance)."
        
        # Wind questions
        if any(word in q for word in ["wind", "windy", "breeze"]):
            wind_mph = weather_data.wind_speed * 2.237
            if wind_mph > 20:
                return f"It's quite windy at {wind_mph:.0f} mph from the {self._get_wind_direction(weather_data.wind_direction).lower()}."
            elif wind_mph > 10:
                return f"Moderate winds at {wind_mph:.0f} mph from the {self._get_wind_direction(weather_data.wind_direction).lower()}."
            else:
                return f"Light winds at {wind_mph:.0f} mph - quite calm."
        
        # UV/sun questions
        if any(word in q for word in ["uv", "sun", "sunny", "sunlight"]):
            return f"UV index is {weather_data.uv_index:.1f} ({weather_data.uv_alert_level}). {weather_data.uv_health_concern.title()} concern level."
        
        # Pressure questions
        if any(word in q for word in ["pressure", "headache", "barometric"]):
            return f"Barometric pressure is {weather_data.pressure_surface_level:.1f} mbar. {weather_data.pressure_trend}."
        
        # Visibility questions
        if any(word in q for word in ["visibility", "clear", "see", "fog", "foggy"]):
            visibility_miles = weather_data.visibility * 0.621371
            if visibility_miles < 1:
                return f"Very poor visibility - less than {visibility_miles:.1f} mile. Dense fog present."
            elif visibility_miles < 3:
                return f"Poor visibility - only {visibility_miles:.1f} miles."
            elif visibility_miles < 10:
                return f"Moderate visibility - {visibility_miles:.1f} miles."
            else:
                return "Excellent visibility - crystal clear!"
        
        # General condition questions
        if any(word in q for word in ["weather", "like outside", "conditions", "what's it like"]):
            return self.format_simple_weather_summary(weather_data)
        
        # Default comprehensive response
        return self.format_simple_weather_summary(weather_data)


# Create global weather monitor instance
_weather_monitor = None

def get_weather_monitor() -> WeatherMonitor:
    """Get weather monitor instance with error handling"""
    global _weather_monitor
    try:
        if _weather_monitor is None:
            _weather_monitor = WeatherMonitor()
        return _weather_monitor
    except Exception as e:
        print(f"⚠️  Weather monitor initialization failed: {e}")
        return None


# Command detection functions
def detect_weather_command(user_input: str) -> bool:
    """Detect if user input contains weather-related commands"""
    weather_keywords = [
        "weather", "temperature", "temp", "rain", "raining", "snow", "snowing",
        "wind", "windy", "sunny", "cloudy", "fog", "foggy", "storm", "thunder",
        "pressure", "barometric", "uv", "humidity", "conditions", "outside",
        "headache weather", "weather alerts", "weather patterns"
    ]
    
    user_lower = user_input.lower()
    return any(keyword in user_lower for keyword in weather_keywords)


def handle_comprehensive_weather_command(user_input: str, project: str) -> Dict[str, str]:
    """Handle weather commands with comprehensive responses"""
    try:
        monitor = get_weather_monitor()
        if not monitor:
            return {"SyntaxPrime": "🌦️ Weather monitoring is currently unavailable. Check your Tomorrow.io API key configuration."}
        
        # Get comprehensive weather conditions
        weather_data = monitor.get_comprehensive_conditions()
        
        # Determine response type based on question
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ["detailed", "full", "complete", "report"]):
            response_content = monitor.format_detailed_weather_report(weather_data)
        elif any(word in user_lower for word in ["simple", "quick", "brief"]):
            response_content = monitor.format_simple_weather_summary(weather_data)
        else:
            # Smart response based on question content
            response_content = monitor.answer_simple_weather_questions(user_input, weather_data)
        
        # Add health alerts if present
        alerts = monitor.get_health_alerts(weather_data)
        if alerts:
            response_content += "\n\n🏥 **Health Alerts**:\n" + "\n".join(f"• {alert}" for alert in alerts)
        
        # Save weather context if Ghostline integration available
        if GHOSTLINE_INTEGRATION:
            weather_context = {
                "comprehensive_weather_data": {
                    "temperature": weather_data.temperature,
                    "feels_like": weather_data.feels_like,
                    "condition": weather_data.weather_description,
                    "precipitation_probability": weather_data.precipitation_probability,
                    "wind_speed_mph": weather_data.wind_speed * 2.237,
                    "pressure": weather_data.pressure_surface_level,
                    "pressure_trend": weather_data.pressure_trend,
                    "uv_index": weather_data.uv_index,
                    "visibility_miles": weather_data.visibility * 0.621371,
                    "timestamp": weather_data.timestamp.isoformat()
                },
                "alerts": alerts
            }
            
            try:
                save_conversation_enhanced(
                    project,
                    user_input,
                    {"SyntaxPrime": response_content},
                    {"weather_context": weather_context}
                )
            except Exception as e:
                print(f"⚠️  Failed to save weather context: {e}")
        
        return {"SyntaxPrime": response_content}
        
    except Exception as e:
        error_msg = f"🌦️ Weather monitoring error: {str(e)}"
        print(error_msg)
        return {"SyntaxPrime": error_msg}


def handle_weather_alerts_command(user_input: str, project: str) -> Dict[str, str]:
    """Handle weather alerts and pattern analysis commands"""
    try:
        monitor = get_weather_monitor()
        if not monitor:
            return {"SyntaxPrime": "🌦️ Weather monitoring is currently unavailable."}
        
        # Get current weather data
        weather_data = monitor.get_comprehensive_conditions()
        
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ["alert", "alerts", "warning", "warnings"]):
            # Health alerts analysis
            alerts = monitor.get_health_alerts(weather_data)
            
            analysis = f"🚨 **Weather Health Alerts Analysis**\n\n"
            
            if alerts:
                analysis += f"**Active Alerts ({len(alerts)}):**\n"
                for alert in alerts:
                    analysis += f"• {alert}\n"
                analysis += "\n"
            else:
                analysis += "✅ **No active health alerts**\n\n"
            
            analysis += f"**Current Monitoring Status:**\n"
            analysis += f"• Barometric pressure: {weather_data.pressure_surface_level:.1f}mbar ({weather_data.pressure_trend})\n"
            analysis += f"• UV index: {weather_data.uv_index:.1f} ({weather_data.uv_alert_level})\n"
            analysis += f"• General conditions: {weather_data.weather_description}\n\n"
            
            analysis += f"**Alert Thresholds:**\n"
            analysis += f"• Headache risk: Pressure drops ≥{PRESSURE_DROP_THRESHOLD}mbar\n"
            analysis += f"• UV alert: Index ≥{HIGH_UV_THRESHOLD} (high), ≥{VERY_HIGH_UV_THRESHOLD} (very high)\n"
            
            return {"SyntaxPrime": analysis}
            
        elif any(word in user_lower for word in ["pattern", "patterns", "history", "trend", "analysis"]):
            # Pattern analysis
            analysis = f"📊 **Weather Pattern Analysis**\n\n"
            
            analysis += f"**Current Conditions:**\n"
            analysis += f"• Pressure: {weather_data.pressure_surface_level:.1f}mbar ({weather_data.pressure_trend})\n"
            analysis += f"• UV: {weather_data.uv_index:.1f} ({weather_data.uv_alert_level})\n"
            analysis += f"• Temperature: {weather_data.temperature:.1f}°C\n"
            analysis += f"• Humidity: {weather_data.humidity:.0f}%\n\n"
            
            analysis += f"**Pressure History:**\n"
            if monitor.pressure_history:
                recent_entries = monitor.pressure_history[-5:]
                for entry in recent_entries:
                    timestamp = datetime.datetime.fromisoformat(entry["timestamp"])
                    analysis += f"• {timestamp.strftime('%m/%d %H:%M')}: {entry['pressure']:.1f}mbar\n"
            else:
                analysis += "• No historical data available yet\n"
            
            analysis += f"\n**Health Impact Assessment:**\n"
            analysis += f"• Headache risk: {'High' if 'High headache risk' in weather_data.pressure_trend else 'Moderate' if 'Moderate headache risk' in weather_data.pressure_trend else 'Low'}\n"
            analysis += f"• UV protection needed: {'Yes' if weather_data.uv_index >= HIGH_UV_THRESHOLD else 'No'}\n"
            
            # Add visibility assessment
            visibility_miles = weather_data.visibility * 0.621371
            analysis += f"• Visibility: {visibility_miles:.1f} miles "
            if visibility_miles < 1:
                analysis += "(very poor - fog conditions)\n"
            elif visibility_miles < 3:
                analysis += "(poor - drive carefully)\n"
            else:
                analysis += "(good)\n"
            analysis += f"**\n"
            
            analysis += f"\n💡 **Monitoring Settings**:\n"
            analysis += f"• Headache threshold: {PRESSURE_DROP_THRESHOLD}mbar drop\n"
            analysis += f"• UV alert threshold: {HIGH_UV_THRESHOLD} UV index\n"
            analysis += f"• Data cached for: {_cache_duration//60} minutes\n"
            
            return {"SyntaxPrime": analysis}
            
    except Exception as e:
        error_msg = f"📊 Weather pattern analysis error: {str(e)}"
        print(error_msg)
        return {"SyntaxPrime": error_msg}


def handle_weather_integration(user_input: str, project: str) -> Dict[str, str]:
    """Main weather integration handler"""
    user_lower = user_input.lower().strip()
    
    # Route to appropriate handler
    if any(word in user_lower for word in ["alerts", "patterns", "history", "analysis"]):
        return handle_weather_alerts_command(user_input, project)
    else:
        return handle_comprehensive_weather_command(user_input, project)


def is_weather_configured() -> bool:
    """Check if weather monitoring is properly configured"""
    return bool(TOMORROW_IO_API_KEY)


def get_weather_status() -> dict:
    """Get weather system status for diagnostics"""
    try:
        monitor = get_weather_monitor()
        if not monitor:
            return {
                "configured": False,
                "api_key_present": bool(TOMORROW_IO_API_KEY),
                "error": "Weather monitor initialization failed"
            }
        
        # Test API connection
        try:
            weather_data = monitor.get_comprehensive_conditions()
            return {
                "configured": True,
                "api_key_present": bool(TOMORROW_IO_API_KEY),
                "last_reading": weather_data.timestamp.isoformat(),
                "pressure_history_count": len(monitor.pressure_history),
                "current_pressure": weather_data.pressure_surface_level,
                "current_uv": weather_data.uv_index,
                "current_condition": weather_data.weather_description,
                "status": "operational"
            }
        except Exception as e:
            return {
                "configured": True,
                "api_key_present": bool(TOMORROW_IO_API_KEY),
                "error": str(e),
                "status": "api_error"
            }
            
    except Exception as e:
        return {
            "configured": False,
            "error": str(e),
            "status": "system_error"
        }


# Comprehensive command mappings with natural language support
WEATHER_COMMANDS = {
    # Basic weather
    "weather": handle_comprehensive_weather_command,
    "weather today": handle_comprehensive_weather_command,
    "weather now": handle_comprehensive_weather_command,
    "what's the weather": handle_comprehensive_weather_command,
    "what's it like outside": handle_comprehensive_weather_command,
    "how's the weather": handle_comprehensive_weather_command,
    "current weather": handle_comprehensive_weather_command,
    "conditions": handle_comprehensive_weather_command,
    
    # Temperature
    "temperature": handle_comprehensive_weather_command,
    "temp": handle_comprehensive_weather_command,
    "how hot": handle_comprehensive_weather_command,
    "how cold": handle_comprehensive_weather_command,
    "how warm": handle_comprehensive_weather_command,
    "how cool": handle_comprehensive_weather_command,
    
    # Precipitation
    "is it raining": handle_comprehensive_weather_command,
    "will it rain": handle_comprehensive_weather_command,
    "rain": handle_comprehensive_weather_command,
    "is it snowing": handle_comprehensive_weather_command,
    "snow": handle_comprehensive_weather_command,
    "precipitation": handle_comprehensive_weather_command,
    
    # Wind
    "how windy": handle_comprehensive_weather_command,
    "wind": handle_comprehensive_weather_command,
    "windy": handle_comprehensive_weather_command,
    "breeze": handle_comprehensive_weather_command,
    
    # Visibility
    "visibility": handle_comprehensive_weather_command,
    "fog": handle_comprehensive_weather_command,
    "foggy": handle_comprehensive_weather_command,
    
    # Detailed reports
    "weather report": handle_comprehensive_weather_command,
    "full weather": handle_comprehensive_weather_command,
    "detailed weather": handle_comprehensive_weather_command,
    
    # Health-focused (specialized commands)
    "pressure": handle_comprehensive_weather_command,
    "barometric pressure": handle_comprehensive_weather_command,
    "uv": handle_comprehensive_weather_command,
    "uv index": handle_comprehensive_weather_command,
    "headache weather": handle_comprehensive_weather_command,
    "weather alerts": handle_weather_alerts_command,
    "weather patterns": handle_weather_alerts_command,
    "pressure history": handle_weather_alerts_command
}


# Compatibility aliases for backward compatibility
def handle_weather_command(user_input: str, project: str) -> Dict[str, str]:
    """
    Compatibility alias for handle_comprehensive_weather_command
    DEPRECATED: Use handle_comprehensive_weather_command instead
    """
    print("⚠️  Warning: handle_weather_command is deprecated, use handle_comprehensive_weather_command")
    return handle_comprehensive_weather_command(user_input, project)


def test_weather_integration():
    """Test weather integration for diagnostics"""
    print("🌦️  Testing Tomorrow.io Weather Integration...")
    
    if not TOMORROW_IO_API_KEY:
        print("❌ Tomorrow.io API key not configured")
        return False
    
    try:
        monitor = WeatherMonitor()
        weather_data = monitor.get_comprehensive_conditions()
        print(f"✅ Weather data retrieved successfully")
        print(f"   Temperature: {weather_data.temperature}°C ({weather_data.temperature * 9/5 + 32:.0f}°F)")
        print(f"   Condition: {weather_data.weather_description}")
        print(f"   Pressure: {weather_data.pressure_surface_level} mbar")
        print(f"   UV Index: {weather_data.uv_index}")
        print(f"   Wind: {weather_data.wind_speed * 2.237:.0f} mph")
        print(f"   Pressure trend: {weather_data.pressure_trend}")
        return True
        
    except Exception as e:
        print(f"❌ Weather integration test failed: {e}")
        return False


# Export all available functions for proper imports
__all__ = [
    # Main handler functions
    'handle_comprehensive_weather_command',
    'handle_weather_alerts_command',
    'handle_weather_integration',
    'detect_weather_command',
    
    # Data structures and configuration
    'WEATHER_COMMANDS',
    'ComprehensiveWeatherData',
    'WeatherMonitor',
    'TomorrowIOClient',
    
    # Utility functions
    'get_weather_monitor',
    'is_weather_configured',
    'get_weather_status',
    'test_weather_integration',
    
    # Compatibility (deprecated)
    'handle_weather_command',  # Deprecated alias
]


def verify_weather_integration_exports():
    """Verify all expected functions are available for import"""
    print("🔍 Verifying weather integration exports...")
    
    expected_functions = [
        'handle_comprehensive_weather_command',
        'handle_weather_alerts_command',
        'handle_weather_integration',
        'detect_weather_command',
        'get_weather_monitor',
        'is_weather_configured',
        'get_weather_status'
    ]
    
    current_module = globals()
    missing_functions = []
    
    for func_name in expected_functions:
        if func_name not in current_module:
            missing_functions.append(func_name)
        else:
            if callable(current_module[func_name]):
                print(f"✅ {func_name} - Available")
            else:
                print(f"❌ {func_name} - Not callable")
                missing_functions.append(func_name)
    
    if missing_functions:
        print(f"❌ Missing functions: {missing_functions}")
        return False
    else:
        print("✅ All weather integration functions are available")
        return True


if __name__ == "__main__":
    verify_weather_integration_exports()
    test_weather_integration()
