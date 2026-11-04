"""Icon choice by meaning. The planner may name an icon, but a 2-4 word label is the real evidence: 'FOOTBALL' must not get a power button.
`choose_icon` scores every available icon against the label's words with a small hand-written keyword table and picks the best match;
the planner's own pick is kept only when it fits the label too. Pure and deterministic."""
from __future__ import annotations

import re

# icon name -> words that call for it (matched by prefix, so 'engineering' matches 'engineer')
KEYWORDS: dict[str, tuple[str, ...]] = {
    "cog": ("mechanic", "engineer", "machine", "gear", "manufactur", "automation", "process"),
    "wrench": ("maintenance", "repair", "tool", "service", "workshop"), "hammer": ("construct", "civil", "build", "fabricat"),
    "cpu": ("embedded", "processor", "hardware", "electronic", "chip", "vlsi"), "circuit-board": ("circuit", "pcb", "electronic", "embedded"),
    "terminal": ("software", "program", "command", "devops", "coding", "backend"), "code": ("software", "coding", "develop", "web", "app"),
    "monitor": ("computer", "desktop", "online", "virtual", "display"), "laptop": ("laptop", "online test", "digital", "e-learning", "technical"),
    "bot": ("robot", "ai", "assistant", "automat", "chatbot", "machine learning"), "database": ("data", "database", "analytics", "big data"),
    "wifi": ("network", "iot", "wireless", "internet"), "satellite": ("space", "satellite", "aerospace", "communication"),
    "smartphone": ("mobile", "phone", "android", "app"), "calculator": ("account", "calculat", "math", "statistic", "audit", "finance"),
    "zap": ("electric", "power", "energy", "voltage", "spark", "fast"), "battery-charging": ("battery", "charging", "ev", "fuel cell", "storage"),
    "flame": ("fire", "heat", "thermal", "combust", "hydrogen", "fuel"), "sun": ("solar", "sun", "renewable", "daylight"),
    "wind": ("wind", "air", "aero", "turbine"), "droplets": ("water", "hydro", "hydrogen", "fluid", "rain", "liquid"),
    "leaf": ("green", "eco", "environment", "sustainab", "nature", "carbon", "campus"), "sprout": ("agri", "farm", "grow", "plant", "seed", "startup"),
    "tree-pine": ("tree", "forest", "afforest", "green"), "recycle": ("recycl", "waste", "circular", "reuse"),
    "thermometer": ("temperature", "thermal", "climate", "weather"), "mountain": ("trek", "adventure", "hike", "outdoor", "climb"),
    "car": ("car", "automotive", "vehicle", "motor", "racing", "drive"), "truck": ("logistic", "transport", "supply", "delivery", "operation"),
    "plane": ("aviation", "aero", "flight", "travel", "aircraft"), "bike": ("cycl", "bicycle", "bike"),
    "flask-conical": ("chemi", "lab", "science", "experiment", "reaction"), "microscope": ("research", "biolog", "microscop", "study", "science"),
    "atom": ("physic", "atom", "nano", "quantum", "material"), "dna": ("gene", "dna", "bio", "life science"), "brain": ("brain", "cognit", "psycholog", "neural", "mind"),
    "stethoscope": ("medic", "health", "clinic", "nursing", "doctor"), "pill": ("pharma", "drug", "medicine"),
    "graduation-cap": ("eligib", "student", "academic", "degree", "education", "course", "year", "graduate", "admission", "b.e", "b.tech"),
    "book-open": ("book", "read", "library", "literature", "syllabus", "learn", "study"), "scroll": ("certificate", "document", "policy", "scroll"),
    "landmark": ("govern", "heritage", "institution", "law", "public", "civil"), "languages": ("language", "english", "tamil", "translat", "linguist"),
    "newspaper": ("journal", "media", "news", "press", "editorial"), "file-text": ("resume", "document", "report", "paper", "application", "form", "cv"),
    "clipboard-list": ("instruction", "checklist", "process", "procedure", "agenda", "interview", "test"),
    "clipboard-check": ("test", "assessment", "exam", "selection", "screening", "eligible"),
    "users": ("team", "community", "member", "participant", "group", "student", "kabaddi", "human resource", "hr"),
    "user": ("individual", "profile", "personal", "single"), "handshake": ("partner", "collaborat", "mou", "sponsor", "association"),
    "heart-handshake": ("volunteer", "social", "service", "outreach", "ngo", "welfare"), "megaphone": ("market", "publicity", "announce", "promot", "public relation", "campaign"),
    "message-circle": ("communicat", "discussion", "panel", "chat", "feedback", "talk"), "briefcase": ("management", "career", "job", "placement", "business", "recruit", "company", "corporate"),
    "banknote": ("finance", "sponsor", "fee", "prize money", "fund", "money", "budget", "stipend"), "store": ("retail", "shop", "market", "store", "entrepreneur"),
    "building": ("building", "company", "organi", "department", "campus", "infrastructure"), "factory": ("industry", "plant", "manufactur", "production", "factory"),
    "calendar": ("date", "schedule", "calendar", "deadline", "day"), "calendar-days": ("days", "timeline", "event", "schedule"), "clock": ("time", "hour", "duration", "session"),
    "timer": ("athletic", "track", "sprint", "run", "marathon", "relay", "race", "stopwatch"), "map-pin": ("venue", "location", "address", "hall", "ground", "place"),
    "ticket": ("entry", "ticket", "pass", "admission free", "registration"), "gift": ("gift", "reward", "goodie", "hamper"),
    "phone": ("contact", "call", "phone", "helpline"), "mail": ("email", "mail", "write", "contact"), "globe": ("website", "global", "international", "web", "online", "world"),
    "trophy": ("trophy", "winner", "champion", "prize", "cup", "final", "competition", "cricket"), "medal": ("medal", "award", "rank", "podium", "achiev", "badminton", "tennis"),
    "target": ("target", "goal", "aim", "objective", "archery", "shooting", "mission"), "star": ("star", "best", "featured", "highlight", "special"),
    "flag": ("flag", "milestone", "launch", "start", "finish"), "crown": ("royal", "king", "queen", "pageant", "leader"),
    "goal": ("football", "soccer", "goalkeeper", "futsal", "hockey", "penalty"), "dribbble": ("basketball", "hoop", "netball", "dunk"),
    "volleyball": ("volleyball", "beach", "throwball", "handball"), "dumbbell": ("gym", "fitness", "weight", "strength", "powerlift", "body"),
    "weight": ("weight", "powerlifting", "shot put", "heavy"), "swords": ("fencing", "martial", "wrestl", "karate", "judo", "combat", "sword", "duel"),
    "hand-fist": ("kabaddi", "boxing", "punch", "kho", "strength", "fight"), "footprints": ("walk", "trek", "footprint", "path", "run", "cross country"),
    "person-standing": ("yoga", "pose", "athlete", "individual", "stance", "player"), "activity": ("health", "pulse", "fit", "cardio", "activity", "wellness"),
    "music": ("music", "song", "melody", "concert", "band", "singing", "instrument"), "music-2": ("music", "rhythm", "beat", "dj", "dance"),
    "mic": ("speaker", "talk", "debate", "speech", "podcast", "elocution", "mic"), "mic-vocal": ("vocal", "singing", "karaoke", "solo", "voice", "choir"),
    "guitar": ("guitar", "acoustic", "band", "rock", "unplugged"), "piano": ("piano", "keyboard", "classical", "orchestra"),
    "drama": ("drama", "theatre", "theater", "play", "skit", "mime", "acting", "stage"), "film": ("film", "cinema", "movie", "short film", "video", "screening"),
    "camera": ("photograph", "camera", "photo", "shoot", "lens"), "palette": ("art", "paint", "design", "colour", "color", "creative", "sketch"),
    "pen-tool": ("design", "illustrat", "graphic", "vector", "drawing", "calligraph"), "shirt": ("fashion", "ramp", "costume", "cloth", "textile", "apparel", "show"),
    "sparkles": ("festival", "celebrat", "magic", "special", "inaugur", "fest"), "party-popper": ("party", "celebrat", "carnival", "gala", "farewell", "freshers"),
    "lightbulb": ("idea", "innovat", "brainstorm", "ideation", "pitch", "concept", "startup", "hackathon"), "rocket": ("launch", "startup", "space", "venture", "accelerat", "boost"),
    "shield": ("secur", "cyber", "safety", "protect", "defence", "ctf"), "award": ("award", "recogni", "honour", "excellence", "scholarship"),
    "line-chart": ("growth", "trend"), "chart-bar": ("analytic", "statistic", "report", "performance", "data", "market"), "trending-up": ("growth", "trend", "progress", "increase"),
    "scale": ("law", "legal", "justice", "ethic", "moot", "balance"), "layers": ("layer", "stack", "architecture", "material", "design"),
    "gamepad-2": ("game", "gaming", "esport", "play", "arcade", "quiz"), "radio": ("radio", "broadcast", "rj", "podcast", "signal"),
    "ruler": ("measure", "survey", "dimension", "drafting", "cad", "civil"), "drafting-compass": ("draft", "geometry", "architect", "compass", "design", "plan"),
}


def _words(label: str) -> list[str]:
    return re.findall(r"[a-z0-9.]+", label.lower())


def score(icon: str, label: str) -> int:
    """How many of the icon's keywords the label calls for (prefix match; a longer keyword counts more)."""
    text = " ".join(_words(label))
    words = _words(label)
    total = 0
    for k in KEYWORDS.get(icon, ()):
        if " " in k:
            total += 2 if k in text else 0
        elif any(w.startswith(k) or (len(w) >= 6 and k.startswith(w)) for w in words):
            total += 2 if len(k) >= 5 else 1
    return total


def choose_icon(label: str, hint: str | None, available: set[str]) -> str:
    """The icon that best matches `label`; the planner's `hint` wins only when it also fits the label (or nothing fits)."""
    best, best_score = None, 0
    for icon in sorted(available):  # sorted: ties resolve the same way every run
        s = score(icon, label)
        if s > best_score:
            best, best_score = icon, s
    if hint in available and (score(hint, label) >= best_score or best is None):
        return hint
    return best or (hint if hint in available else ("star" if "star" in available else sorted(available)[0]))
