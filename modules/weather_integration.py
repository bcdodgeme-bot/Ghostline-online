# modules/weather_integration.py
# Complete Tomorrow.io Weather API Integration for Ghostline AI
# Supports comprehensive weather monitoring + health-focused tracking for headaches and UV sensitivity

import os
import json
import requests
import datetime
import math
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import timedelta

# Import Ghostline dependencies
from utils.ghostline_engine import generate_response
from modules.database import save_conversation_enhanced, save_daily_log_enhanced

# Tomorrow.io API configuration
TOMORROW_IO_API_KEY = os.getenv("TOMORROW_IO_API_KEY", "")
TOMORROW_IO_BASE_URL = "https://api.tomorrow.io/v4"

# Default location (you can set this in your environment or modify as needed)
DEFAULT_LOCATION = os.getenv("DEFAULT_WEATHER_LOCATION", "40.7128,-74.0060")  # NYC as default

# Weather alert thresholds
PRESSURE_DROP_THRESHOLD = float(os.getenv("PRESSURE_DROP_THRESHOLD", "3.0"))  # mbar drop that might trigger headaches
HIGH_UV_THRESHOLD = int(os.getenv("HIGH_UV_THRESHOLD", "6"))  # UV index considered high
VERY_HIGH_UV_THRESHOLD = int(os.getenv("VERY_HIGH_UV_THRESHOLD", "8"))  # UV index considered very high

# Weather data cache to avoid excessive API calls
_weather_cache = {}
_cache_duration = int(os.getenv("WEATHER_CACHE_DURATION", "1800"))  # 30 minutes in seconds

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
    4201: {"desc": "Heavy Rain", "icon": "🌧️", "category": "rain"},
    
    # Snow conditions
    5000: {"desc": "Snow", "icon": "❄️", "category": "snow"},
    5001: {"desc": "Flurries", "icon": "🌨️", "category": "snow"},
    5100: {"desc": "Light Snow", "icon": "🌨️", "category": "snow"},
    5101: {"desc": "Heavy Snow", "icon": "❄️", "category": "snow"},
    
    # Freezing conditions
    6000: {"desc": "Freezing Drizzle", "icon": "🧊", "category": "ice"},
    6001: {"desc": "Freezing Rain", "icon": "🧊", "category": "ice"},
    6200: {"desc": "Light Freezing Rain", "icon": "🧊", "category": "ice"},
    6201: {"desc": "Heavy Freezing Rain", "icon": "🧊", "category": "ice"},
    
    # Ice conditions
    7000: {"desc": "Ice Pellets", "icon": "🧊", "category": "ice"},
    7101: {"desc": "Heavy Ice Pellets", "icon": "🧊", "category": "ice"},
    7102: {"desc": "Light Ice Pellets", "icon": "🧊", "category": "ice"},
    
    # Storm conditions
    8000: {"desc": "Thunderstorm", "icon": "⛈️", "category": "storm"}
}

# Weather categories for easy grouping
PRECIPITATION_CODES = [4000, 4001, 4200, 4201, 6000, 6001, 6200, 6201]  # Rain/Drizzle/Freezing
SNOW_CODES = [5000, 5001, 5100, 5101, 7000, 7101, 7102]  # Snow/Ice pellets
STORM_CODES = [8000]  # Thunderstorms
CLEAR_CODES = [1000, 1100]  # Clear/Sunny
CLOUDY_CODES = [1101, 1102, 1001]  # Various cloudy conditions
FOG_CODES = [2000, 2100]  # Fog conditions

# Comprehensive weather fields for API requests
COMPREHENSIVE_WEATHER_FIELDS = [
    "temperature",
    "temperatureApparent",  # Feels like temperature
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

@dataclass
class ComprehensiveWeatherData:
    """Enhanced weather data structure with all the good bits"""
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
    pressure_trend: Optional[str] = None
    uv_alert_level: Optional[str] = None
    weather_description: Optional[str] = None
    weather_icon: Optional[str] = None
    weather_category: Optional[str] = None


class TomorrowIOClient:
    """Tomorrow.io API client with error handling matching OpenRouter patterns"""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        
        if not api_key:
            print("⚠️  Warning: No Tomorrow.io API key configured")
    
    def _make_request(self, endpoint: str, params: dict = None):
        """Make HTTP request to Tomorrow.io API with comprehensive error handling"""
        headers = {
            "accept": "application/json",
            "content-type": "application/json"
        }
        
        if params is None:
            params = {}
        
        params['apikey'] = self.api_key
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            return response
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                print("🔑 Tomorrow.io: Authentication failed")
                raise Exception("Tomorrow.io API: Authentication failed. Check your API key.")
            elif e.response.status_code == 429:
                print("⏱️  Tomorrow.io: Rate limit exceeded")
                raise Exception("Tomorrow.io API: Rate limit exceeded. You've exceeded your daily quota.")
            elif e.response.status_code == 400:
                print("📝 Tomorrow.io: Bad request")
                raise Exception("Tomorrow.io API: Bad request. Check your parameters.")
            elif e.response.status_code == 404:
                print("🌍 Tomorrow.io: Location not found")
                raise Exception("Tomorrow.io API: Location not found. Check your coordinates.")
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
                "icon": "🌤️",
                "category": "unknown"
            })
            
            weather_data.weather_description = weather_info["desc"]
            weather_data.weather_icon = weather_info["icon"]
            weather_data.weather_category = weather_info["category"]
            
            # Analyze trends and alerts
            weather_data.pressure_trend = self._analyze_pressure_trend(weather_data.pressure_surface_level)
            weather_data.uv_alert_level = self._analyze_uv_level(weather_data.uv_index)
            
            # Cache the comprehensive data
            self._cache_comprehensive_weather_data(location, weather_data)
            
            # Save to pressure history
            self.pressure_history.append({
                "timestamp": weather_data.timestamp.isoformat(),
                "pressure": weather_data.pressure_surface_level
            })
            self._save_pressure_history()
            
            return weather_data
            
        except Exception as e:
            print(f"🌦️  Comprehensive weather fetch error: {e}")
            raise Exception(f"Weather monitoring failed: {e}")
    
    def _analyze_pressure_trend(self, current_pressure: float) -> str:
        """Analyze pressure trend for headache prediction"""
        if len(self.pressure_history) < 2:
            return "insufficient_data"
        
        # Get pressure from 3-6 hours ago
        recent_pressures = []
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=6)
        
        for record in self.pressure_history[-20:]:  # Check last 20 readings
            try:
                record_time = datetime.datetime.fromisoformat(record["timestamp"])
                if record_time > cutoff_time:
                    recent_pressures.append(record["pressure"])
            except (ValueError, KeyError):
                continue
        
        if len(recent_pressures) < 2:
            return "insufficient_recent_data"
        
        pressure_change = current_pressure - recent_pressures[0]
        
        if pressure_change <= -PRESSURE_DROP_THRESHOLD:
            return "dropping_significantly"
        elif pressure_change <= -1.0:
            return "dropping_moderately"
        elif pressure_change >= 3.0:
            return "rising_significantly"
        elif pressure_change >= 1.0:
            return "rising_moderately"
        else:
            return "stable"
    
    def _analyze_uv_level(self, uv_index: float) -> str:
        """Analyze UV level for sun sensitivity alerts"""
        if uv_index >= VERY_HIGH_UV_THRESHOLD:
            return "very_high_risk"
        elif uv_index >= HIGH_UV_THRESHOLD:
            return "high_risk"
        elif uv_index >= 3:
            return "moderate_risk"
        elif uv_index >= 1:
            return "low_risk"
        else:
            return "minimal_risk"
    
    def get_health_alerts(self, weather_data: ComprehensiveWeatherData) -> List[str]:
        """Generate health-related alerts based on weather conditions"""
        alerts = []
        
        # Pressure-related headache alerts
        if weather_data.pressure_trend == "dropping_significantly":
            alerts.append("🧠 **Headache Alert**: Barometric pressure is dropping significantly. You may experience increased headache risk.")
        elif weather_data.pressure_trend == "dropping_moderately":
            alerts.append("⚠️ **Pressure Notice**: Barometric pressure is dropping moderately. Monitor for headache symptoms.")
        
        # UV-related sun sensitivity alerts
        if weather_data.uv_alert_level == "very_high_risk":
            alerts.append("☀️ **High UV Warning**: UV index is very high ({:.1f}). Limit sun exposure and use strong sun protection.".format(weather_data.uv_index))
        elif weather_data.uv_alert_level == "high_risk":
            alerts.append("🌞 **UV Caution**: UV index is high ({:.1f}). Use sun protection if going outside.".format(weather_data.uv_index))
        elif weather_data.uv_alert_level == "moderate_risk":
            alerts.append("🌤️ **UV Notice**: UV index is moderate ({:.1f}). Consider sun protection for extended outdoor time.".format(weather_data.uv_index))
        
        return alerts
    
    def _get_wind_direction(self, degrees: float) -> str:
        """Convert wind direction degrees to compass direction"""
        directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                      "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        index = round(degrees / 22.5) % 16
        return directions[index]
    
    def _get_precipitation_intensity_desc(self, intensity: float) -> str:
        """Convert precipitation intensity to description"""
        if intensity == 0:
            return "None"
        elif intensity < 0.25:
            return "Very Light"
        elif intensity < 1.0:
            return "Light"
        elif intensity < 4.0:
            return "Moderate"
        elif intensity < 16.0:
            return "Heavy"
        else:
            return "Very Heavy"
    
    def _get_precipitation_type_desc(self, precip_type: int) -> str:
        """Convert precipitation type code to description"""
        types = {
            0: "None",
            1: "Rain",
            2: "Snow",
            3: "Freezing Rain",
            4: "Ice Pellets"
        }
        return types.get(precip_type, "Unknown")
    
    def format_simple_weather_summary(self, weather_data: ComprehensiveWeatherData) -> str:
        """Format simple, conversational weather summary"""
        temp_f = weather_data.temperature * 9/5 + 32
        feels_f = weather_data.feels_like * 9/5 + 32
        
        # Start with basic condition
        summary = f"{weather_data.weather_icon} **{weather_data.weather_description}**\n"
        summary += f"**{temp_f:.0f}°F** "
        
        # Add feels-like if significantly different
        temp_diff = abs(feels_f - temp_f)
        if temp_diff >= 5:
            summary += f"(feels like {feels_f:.0f}°F)"
        
        summary += "\n"
        
        # Precipitation info
        if weather_data.weather_category in ["rain", "snow", "ice"]:
            if weather_data.precipitation_intensity > 0:
                intensity = "Light" if weather_data.precipitation_intensity < 2 else "Heavy" if weather_data.precipitation_intensity > 10 else "Moderate"
                summary += f"**{intensity} precipitation** currently\n"
            
            if weather_data.precipitation_probability > 20:
                summary += f"**{weather_data.precipitation_probability:.0f}% chance** of precipitation\n"
        
        # Wind info (if notable)
        wind_mph = weather_data.wind_speed * 2.237  # Convert m/s to mph
        if wind_mph > 10:
            wind_dir = self._get_wind_direction(weather_data.wind_direction)
            summary += f"**Wind:** {wind_mph:.0f} mph {wind_dir}"
            
            if weather_data.wind_gust > weather_data.wind_speed * 1.5:
                gust_mph = weather_data.wind_gust * 2.237
                summary += f" (gusts to {gust_mph:.0f} mph)"
            summary += "\n"
        
        # Visibility (if poor)
        visibility_miles = weather_data.visibility * 0.621371  # km to miles
        if visibility_miles < 5:
            summary += f"**Visibility:** {visibility_miles:.1f} miles\n"
        
        # Humidity (if notable)
        if weather_data.humidity > 80 or weather_data.humidity < 30:
            summary += f"**Humidity:** {weather_data.humidity:.0f}%\n"
        
        return summary.strip()
    
    def format_detailed_weather_report(self, weather_data: ComprehensiveWeatherData) -> str:
        """Format detailed weather report with all the good bits"""
        temp_f = weather_data.temperature * 9/5 + 32
        feels_f = weather_data.feels_like * 9/5 + 32
        wind_mph = weather_data.wind_speed * 2.237
        visibility_miles = weather_data.visibility * 0.621371
        
        report = f"{weather_data.weather_icon} **Detailed Weather Report**\n\n"
        
        # Current conditions
        report += f"**Condition:** {weather_data.weather_description}\n"
        report += f"**Temperature:** {temp_f:.1f}°F ({weather_data.temperature:.1f}°C)\n"
        
        if abs(feels_f - temp_f) >= 3:
            report += f"**Feels Like:** {feels_f:.1f}°F ({weather_data.feels_like:.1f}°C)\n"
        
        # Precipitation details
        if weather_data.weather_category in ["rain", "snow", "ice", "storm"]:
            report += f"\n**💧 Precipitation:**\n"
            
            if weather_data.precipitation_intensity > 0:
                intensity_desc = self._get_precipitation_intensity_desc(weather_data.precipitation_intensity)
                report += f"• Current intensity: {intensity_desc}\n"
            
            if weather_data.precipitation_probability > 0:
                report += f"• Probability: {weather_data.precipitation_probability:.0f}%\n"
            
            precip_type = self._get_precipitation_type_desc(weather_data.precipitation_type)
            if precip_type:
                report += f"• Type: {precip_type}\n"
        
        # Wind conditions
        report += f"\n**💨 Wind & Air:**\n"
        if wind_mph > 1:
            wind_dir = self._get_wind_direction(weather_data.wind_direction)
            report += f"• Wind: {wind_mph:.0f} mph {wind_dir}\n"
            
            if weather_data.wind_gust > weather_data.wind_speed * 1.2:
                gust_mph = weather_data.wind_gust * 2.237
                report += f"• Gusts: Up to {gust_mph:.0f} mph\n"
        else:
            report += f"• Wind: Calm\n"
        
        report += f"• Humidity: {weather_data.humidity:.0f}%\n"
        report += f"• Visibility: {visibility_miles:.1f} miles\n"
        report += f"• Cloud Cover: {weather_data.cloud_cover:.0f}%\n"
        
        # Atmospheric conditions
        report += f"\n**🌡️ Atmospheric:**\n"
        report += f"• Pressure: {weather_data.pressure_surface_level:.1f} mbar"
        
        pressure_trend_text = {
            "dropping_significantly": " 📉 (dropping significantly - headache risk)",
            "dropping_moderately": " 📉 (dropping moderately)",
            "rising_significantly": " 📈 (rising significantly)",
            "rising_moderately": " 📈 (rising moderately)",
            "stable": " ➡️ (stable)"
        }.get(weather_data.pressure_trend, "")
        report += pressure_trend_text + "\n"
        
        # UV information
        if weather_data.uv_index > 0:
            report += f"• UV Index: {weather_data.uv_index:.1f} ({weather_data.uv_health_concern})\n"
        
        # Health alerts
        alerts = self.get_health_alerts(weather_data)
        if alerts:
            report += f"\n**🏥 Health Alerts:**\n"
            for alert in alerts:
                clean_alert = alert.replace('🧠 **', '').replace('☀️ **', '').replace('🌞 **', '').replace('🌤️ **', '').replace('**', '')
                report += f"• {clean_alert}\n"
        
        report += f"\n*Last updated: {weather_data.timestamp.strftime('%I:%M %p')}*"
        
        return report
    
    def answer_simple_weather_questions(self, question: str, weather_data: ComprehensiveWeatherData) -> str:
        """Answer simple weather questions in a conversational way"""
        q = question.lower()
        
        # Temperature questions
        if any(word in q for word in ["temperature", "temp", "hot", "cold", "warm", "cool"]):
            temp_f = weather_data.temperature * 9/5 + 32
            feels_f = weather_data.feels_like * 9/5 + 32
            
            response = f"It's {temp_f:.0f}°F ({weather_data.temperature:.0f}°C)"
            if abs(feels_f - temp_f) >= 5:
                response += f", but feels like {feels_f:.0f}°F"
            
            # Add context
            if temp_f >= 80:
                response += " - quite warm!"
            elif temp_f >= 70:
                response += " - nice and comfortable"
            elif temp_f >= 60:
                response += " - pleasantly cool"
            elif temp_f >= 50:
                response += " - getting chilly"
            elif temp_f >= 32:
                response += " - cold out there"
            else:
                response += " - freezing!"
                
            return response
        
        # Rain/precipitation questions
        if any(word in q for word in ["rain", "raining", "wet", "precipitation", "shower"]):
            if weather_data.weather_category == "rain":
                intensity_desc = self._get_precipitation_intensity_desc(weather_data.precipitation_intensity)
                return f"Yes, it's raining - {intensity_desc.lower()} rain currently."
            elif weather_data.precipitation_probability > 50:
                return f"Not currently raining, but {weather_data.precipitation_probability:.0f}% chance of rain."
            elif weather_data.precipitation_probability > 20:
                return f"No rain right now, but there's a {weather_data.precipitation_probability:.0f}% chance later."
            else:
                return "No rain expected - clear skies ahead!"
        
        # Snow questions
        if any(word in q for word in ["snow", "snowing", "snowy", "flurries"]):
            if weather_data.weather_category == "snow":
                return f"Yes! {weather_data.weather_description.lower()} currently."
            elif weather_data.precipitation_type == 2:
                return f"Snow possible - {weather_data.precipitation_probability:.0f}% chance."
            else:
                return "No snow expected."
        
        # Wind questions
        if any(word in q for word in ["wind", "windy", "breeze", "breezy", "gusty"]):
            wind_mph = weather_data.wind_speed * 2.237
            wind_dir = self._get_wind_direction(weather_data.wind_direction)
            
            if wind_mph < 5:
                return "Very light winds - basically calm."
            elif wind_mph < 15:
                return f"Light breeze at {wind_mph:.0f} mph from the {wind_dir.lower()}."
            elif wind_mph < 25:
                return f"Moderately windy - {wind_mph:.0f} mph {wind_dir.lower()} winds."
            else:
                response = f"Quite windy - {wind_mph:.0f} mph {wind_dir.lower()} winds"
                if weather_data.wind_gust > weather_data.wind_speed * 1.3:
                    gust_mph = weather_data.wind_gust * 2.237
                    response += f" with gusts to {gust_mph:.0f} mph"
                return response + "."
        
        # Visibility/fog questions
        if any(word in q for word in ["visibility", "see", "clear", "fog", "foggy", "hazy"]):
            visibility_miles = weather_data.visibility * 0.621371
            
            if weather_data.weather_category == "fog":
                return f"Foggy conditions - visibility is {visibility_miles:.1f} miles."
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
        
        # Save weather context
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
        
        # Load recent weather history for pattern analysis
        recent_weather = []
        for record in monitor.pressure_history[-168:]:  # Last week of hourly data
            recent_weather.append(record)
        
        if len(recent_weather) < 10:
            return {"SyntaxPrime": "📊 Insufficient weather history for pattern analysis. Check back after a few days of monitoring."}
        
        # Analyze pressure patterns
        pressure_drops = []
        for i in range(1, len(recent_weather)):
            try:
                current_pressure = recent_weather[i]["pressure"]
                previous_pressure = recent_weather[i-1]["pressure"]
                change = current_pressure - previous_pressure
                
                if change <= -PRESSURE_DROP_THRESHOLD:
                    pressure_drops.append({
                        "timestamp": recent_weather[i]["timestamp"],
                        "pressure_change": change,
                        "severity": "significant" if change <= -5.0 else "moderate"
                    })
            except (KeyError, TypeError):
                continue
        
        # Generate pattern analysis response
        analysis = f"📊 **Weather Pattern Analysis (Last 7 Days)**\n\n"
        analysis += f"**Pressure Monitoring**: {len(recent_weather)} readings collected\n"
        analysis += f"**Significant Pressure Drops**: {len([d for d in pressure_drops if d['severity'] == 'significant'])}\n"
        analysis += f"**Moderate Pressure Drops**: {len([d for d in pressure_drops if d['severity'] == 'moderate'])}\n\n"
        
        if pressure_drops:
            analysis += "**Recent Pressure Events**:\n"
            for drop in pressure_drops[-5:]:  # Last 5 events
                try:
                    timestamp = datetime.datetime.fromisoformat(drop["timestamp"])
                    analysis += f"• {timestamp.strftime('%m/%d %I:%M%p')}: {drop['pressure_change']:.1f}mbar drop ({drop['severity']})\n"
                except (ValueError, KeyError):
                    continue
        else:
            analysis += "**No significant pressure drops detected in recent history.**\n"
        
        analysis += f"\n💡 **Monitoring Settings**:\n"
        analysis += f"• Headache threshold: {PRESSURE_DROP_THRESHOLD}mbar drop\n"
        analysis += f"• UV alert threshold: {HIGH_UV_THRESHOLD} UV index\n"
        analysis += f"• Data cached for: {_cache_duration//60} minutes\n"
        
        return {"SyntaxPrime": analysis}
        
    except Exception as e:
        error_msg = f"📊 Weather pattern analysis error: {str(e)}"
        print(error_msg)
        return {"SyntaxPrime": error_msg}


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


if __name__ == "__main__":
    test_weather_integration()
