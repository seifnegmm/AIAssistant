"""
Natural language time parsing utilities.
Handles phrases like "in 30 minutes", "tomorrow at 9am", "5pm", etc.
"""

import re
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TimeParser:
    """Parse natural language time expressions into datetime objects."""

    @staticmethod
    def _normalize_arabic_time(time_str: str) -> str:
        """Normalize Arabic time expressions to English equivalents."""
        # Convert Arabic-Indic digits to ASCII
        arabic_digits = "٠١٢٣٤٥٦٧٨٩"
        ascii_digits = "0123456789"
        trans = str.maketrans(arabic_digits, ascii_digits)
        time_str = time_str.translate(trans)

        # Normalize Arabic tokens to English
        replacements = {
            "النهارده": "today",
            "النهاردة": "today",
            "اليوم": "today",
            "بكره": "tomorrow",
            "بكرة": "tomorrow",
            "غداً": "tomorrow",
            "غدا": "tomorrow",
            "مساءً": "pm",
            "مساء": "pm",
            "صباحاً": "am",
            "صباحا": "am",
            "صباح": "am",
            "الساعة": "",
            "الساعه": "",
        }

        for arabic, english in replacements.items():
            time_str = time_str.replace(arabic, english)

        return time_str.strip()

    @staticmethod
    def parse(time_str: str) -> Optional[datetime]:
        """
        Parse a time string into a datetime object.

        Supports:
        - Relative: "in 30 minutes", "in 2 hours", "in 1 day"
        - Today: "5pm", "17:00", "at 3:30pm"
        - Tomorrow: "tomorrow at 9am", "tomorrow 14:00"
        - Absolute: "2026-02-10 14:00", "Feb 10 2pm"
        - Arabic: Automatically normalizes Arabic time expressions

        Args:
            time_str: Natural language time expression (English or Arabic)

        Returns:
            datetime object or None if parsing fails
        """
        # Normalize Arabic time expressions first
        time_str = TimeParser._normalize_arabic_time(time_str)
        time_str = time_str.lower().strip()
        now = datetime.now()

        # Try relative time: "in X minutes/hours/days"
        relative = TimeParser._parse_relative(time_str, now)
        if relative:
            return relative

        # Try today time: "5pm", "17:00", "at 3:30pm"
        today_time = TimeParser._parse_today(time_str, now)
        if today_time:
            return today_time

        # Try tomorrow: "tomorrow at 9am", "tomorrow 14:00"
        tomorrow_time = TimeParser._parse_tomorrow(time_str, now)
        if tomorrow_time:
            return tomorrow_time

        # Try absolute ISO format: "2026-02-10 14:00"
        absolute = TimeParser._parse_absolute(time_str)
        if absolute:
            return absolute

        logger.warning(f"Could not parse time string: {time_str}")
        return None

    @staticmethod
    def _parse_relative(time_str: str, now: datetime) -> Optional[datetime]:
        """Parse relative time like 'in 30 minutes', 'in 2 hours'."""
        # Match: "in X minutes/hours/days/weeks"
        pattern = r"in\s+(\d+)\s+(minute|hour|day|week)s?"
        match = re.search(pattern, time_str)

        if match:
            value = int(match.group(1))
            unit = match.group(2)

            if unit == "minute":
                return now + timedelta(minutes=value)
            elif unit == "hour":
                return now + timedelta(hours=value)
            elif unit == "day":
                return now + timedelta(days=value)
            elif unit == "week":
                return now + timedelta(weeks=value)

        return None

    @staticmethod
    def _parse_today(time_str: str, now: datetime) -> Optional[datetime]:
        """Parse today's time like '5pm', '17:00', 'at 3:30pm'."""
        # Remove 'at' prefix if present
        time_str = re.sub(r"^at\s+", "", time_str)

        # Match: "5pm", "5:30pm", "17:00", "17:30"
        # 12-hour format with am/pm
        match_12h = re.match(r"(\d{1,2}):?(\d{2})?\s*(am|pm)", time_str)
        if match_12h:
            hour = int(match_12h.group(1))
            minute = int(match_12h.group(2)) if match_12h.group(2) else 0
            meridiem = match_12h.group(3)

            if meridiem == "pm" and hour != 12:
                hour += 12
            elif meridiem == "am" and hour == 12:
                hour = 0

            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

            # If time has passed today, schedule for tomorrow
            if target <= now:
                target += timedelta(days=1)

            return target

        # 24-hour format: "17:00", "17:30"
        match_24h = re.match(r"(\d{1,2}):(\d{2})", time_str)
        if match_24h:
            hour = int(match_24h.group(1))
            minute = int(match_24h.group(2))

            if 0 <= hour < 24 and 0 <= minute < 60:
                target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

                # If time has passed today, schedule for tomorrow
                if target <= now:
                    target += timedelta(days=1)

                return target

        return None

    @staticmethod
    def _parse_tomorrow(time_str: str, now: datetime) -> Optional[datetime]:
        """Parse tomorrow's time like 'tomorrow at 9am', 'tomorrow 14:00'."""
        if "tomorrow" not in time_str:
            return None

        # Extract time part after "tomorrow"
        time_part = re.sub(r"tomorrow\s+(at\s+)?", "", time_str).strip()

        # Parse the time using today's parser
        parsed_time = TimeParser._parse_today(time_part, now)

        if parsed_time:
            # Ensure it's tomorrow (not today + 1 if time passed)
            tomorrow = now + timedelta(days=1)
            return parsed_time.replace(
                year=tomorrow.year, month=tomorrow.month, day=tomorrow.day
            )

        return None

    @staticmethod
    def _parse_absolute(time_str: str) -> Optional[datetime]:
        """Parse absolute datetime like '2026-02-10 14:00'."""
        # Try ISO format: YYYY-MM-DD HH:MM
        formats = [
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%Y-%m-%d",  # Just date (will use 00:00)
        ]

        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue

        return None

    @staticmethod
    def format_scheduled_time(dt: datetime) -> str:
        """Format datetime for display in Telegram messages."""
        now = datetime.now()

        # If today
        if dt.date() == now.date():
            return f"today at {dt.strftime('%I:%M %p')}"

        # If tomorrow
        elif dt.date() == (now + timedelta(days=1)).date():
            return f"tomorrow at {dt.strftime('%I:%M %p')}"

        # If within a week
        elif (dt - now).days < 7:
            return dt.strftime("%A at %I:%M %p")  # "Monday at 02:30 PM"

        # Otherwise full date
        else:
            return dt.strftime("%B %d at %I:%M %p")  # "February 10 at 02:30 PM"
