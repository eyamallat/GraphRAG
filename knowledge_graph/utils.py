import  re
import datetime

def sanitize_id(name, prefix):
    """Sanitize an ID by replacing non-alphanumeric characters with underscores."""
    if not isinstance(name, str):
        name = str(name)
    sanitized_name = re.sub(r'\W+', '_', name.lower()).strip('_') if name else "unknown"
    return f"{prefix}_{sanitized_name}"

def compute_period(start, end):
    try:
        start_date = datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.strptime(end, "%Y-%m-%d")
        delta = end_date - start_date
        return f"{abs(delta.days)} days"
    except Exception:
        return "N/A"
