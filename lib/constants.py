"""
lib/constants.py — All MLB model constants.
Single source of truth for park factors, tier thresholds, team maps,
batter hand seeds. Import from here; never redeclare inline.
"""

from typing import Dict, Tuple

TIERS: Dict[str, int] = {
    "🔒 TIER 1": 80,
    "✅ TIER 2": 70,
    "📊 TIER 3": 60,
    "❌ NO PLAY": 0,
}

# (lat, lon, stadium_name, is_dome)
STADIUM_COORDS: Dict[str, Tuple[float, float, str, bool]] = {
    "ARI": (33.4453, -112.0667, "Chase Field", True),
    "ATL": (33.8908, -84.4678, "Truist Park", False),
    "BAL": (39.2839, -76.6216, "Oriole Park", False),
    "BOS": (42.3467, -71.0972, "Fenway Park", False),
    "CHC": (41.9484, -87.6553, "Wrigley Field", False),
    "CWS": (41.8300, -87.6339, "Guaranteed Rate Field", False),
    "CIN": (39.0974, -84.5082, "Great American Ball Park", False),
    "CLE": (41.4962, -81.6853, "Progressive Field", False),
    "COL": (39.7559, -104.9942, "Coors Field", False),
    "DET": (42.3390, -83.0485, "Comerica Park", False),
    "HOU": (29.7573, -95.3555, "Minute Maid Park", True),
    "KC":  (39.0517, -94.4803, "Kauffman Stadium", False),
    "LAA": (33.8003, -117.8827, "Angel Stadium", False),
    "LAD": (34.0739, -118.2400, "Dodger Stadium", False),
    "MIA": (25.7781, -80.2197, "loanDepot park", True),
    "MIL": (43.0280, -87.9712, "American Family Field", True),
    "MIN": (44.9817, -93.2778, "Target Field", False),
    "NYM": (40.7571, -73.8458, "Citi Field", False),
    "NYY": (40.8296, -73.9262, "Yankee Stadium", False),
    "OAK": (37.7516, -122.2005, "Oakland Coliseum", False),
    "PHI": (39.9061, -75.1665, "Citizens Bank Park", False),
    "PIT": (40.4469, -80.0057, "PNC Park", False),
    "SD":  (32.7076, -117.1570, "Petco Park", False),
    "SEA": (47.5914, -122.3325, "T-Mobile Park", True),
    "SF":  (37.7786, -122.3893, "Oracle Park", False),
    "STL": (38.6226, -90.1928, "Busch Stadium", False),
    "TB":  (27.7682, -82.6534, "Tropicana Field", True),
    "TEX": (32.7512, -97.0832, "Globe Life Field", True),
    "TOR": (43.6414, -79.3894, "Rogers Centre", True),
    "WSH": (38.8730, -77.0074, "Nationals Park", False),
    "MEX": (19.4240, -99.0680, "Estadio Alfredo Harp Helú", False),
}

# Composite TB park factors (3-year rolling Statcast, normalized 1.00 = average)
PARK_TB_FACTORS: Dict[str, float] = {
    "ARI": 1.05, "ATL": 0.98, "BAL": 1.02, "BOS": 1.04, "CHC": 1.06,
    "CWS": 0.99, "CIN": 1.08, "CLE": 0.95, "COL": 1.22, "DET": 0.97,
    "HOU": 1.01, "KC":  0.98, "LAA": 0.96, "LAD": 1.00, "MIA": 0.91,
    "MIL": 0.99, "MIN": 1.04, "NYM": 0.97, "NYY": 1.07, "OAK": 0.93,
    "PHI": 1.05, "PIT": 0.96, "SD":  0.92, "SEA": 0.97, "SF":  0.94,
    "STL": 0.99, "TB":  0.95, "TEX": 1.03, "TOR": 1.01, "WSH": 0.97,
    "MEX": 1.18,
}

PARK_HR_FACTORS: Dict[str, float] = {
    "ARI": 1.10, "ATL": 0.99, "BAL": 1.08, "BOS": 1.07, "CHC": 1.15,
    "CWS": 1.05, "CIN": 1.18, "CLE": 0.93, "COL": 1.35, "DET": 0.96,
    "HOU": 1.02, "KC":  0.96, "LAA": 0.94, "LAD": 1.02, "MIA": 0.87,
    "MIL": 1.01, "MIN": 1.09, "NYM": 0.95, "NYY": 1.14, "OAK": 0.90,
    "PHI": 1.10, "PIT": 0.94, "SD":  0.88, "SEA": 0.95, "SF":  0.90,
    "STL": 1.00, "TB":  0.92, "TEX": 1.06, "TOR": 1.03, "WSH": 0.95,
    "MEX": 1.30,
}

# League-average platoon SLG adjustments (points)
PLATOON_ADJ: Dict[str, int] = {
    "RHB_vs_LHP": +33,
    "LHB_vs_RHP": +56,
    "LHB_vs_LHP": -35,
    "RHB_vs_RHP":  0,
}

TEAM_ABB_MAP: Dict[str, str] = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD", "Seattle Mariners": "SEA",
    "San Francisco Giants": "SF", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
    "Athletics": "OAK",
    "AZ": "ARI", "D-backs": "ARI", "Diamondbacks": "ARI",
    "Cubs": "CHC", "Reds": "CIN", "Guardians": "CLE",
    "Rockies": "COL", "Tigers": "DET", "Astros": "HOU",
    "Royals": "KC", "Angels": "LAA", "Dodgers": "LAD",
    "Marlins": "MIA", "Brewers": "MIL", "Twins": "MIN",
    "Mets": "NYM", "Yankees": "NYY", "Phillies": "PHI",
    "Pirates": "PIT", "Padres": "SD", "Mariners": "SEA",
    "Giants": "SF", "Cardinals": "STL", "Rays": "TB",
    "Rangers": "TEX", "Blue Jays": "TOR", "Nationals": "WSH",
    "Braves": "ATL", "Orioles": "BAL", "Red Sox": "BOS",
    "White Sox": "CWS",
}

# Pre-seeded batter handedness (MLBAM ID → L/R/S). Roster fallback for MLB API gaps.
MLBAM_BATTER_HAND: Dict[str, str] = {
    # Switch hitters
    "665742": "S", "669257": "S", "677594": "S", "641355": "S",
    "621566": "S", "663728": "S", "665489": "S", "680757": "S",
    "681481": "S", "671218": "S", "681624": "S",
    # Left-handed batters
    "592450": "L", "518692": "L", "623993": "L", "641313": "L",
    "670541": "L", "666023": "L", "665919": "L", "656976": "L",
    "660271": "L", "665487": "L", "666142": "L", "608384": "L",
    "641645": "L", "677951": "L", "643376": "L", "664702": "L",
    "668939": "L", "682998": "L", "683737": "L", "650490": "L",
    "642133": "L", "660162": "L", "682625": "L", "666301": "L",
    "642708": "L", "664056": "L", "605141": "L", "687695": "L",
    "669032": "L", "691026": "L", "691157": "L", "694538": "L",
}
