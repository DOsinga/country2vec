import json
import os
import threading

import apsw
import numpy as np
import scipy.spatial.distance
import sqlite_vec
from django.http import HttpResponse

from projects.common import Project

DB_PATH = os.path.join(os.path.dirname(__file__), "static", "word2vec.db")

_local = threading.local()


def _conn():
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = apsw.Connection(DB_PATH, flags=apsw.SQLITE_OPEN_READONLY)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        _local.conn = conn
    return conn


def _vec_for_lower(conn, word_lower):
    row = next(
        conn.execute("SELECT embedding FROM embeddings WHERE word = ?", (word_lower,)),
        None,
    )
    return np.frombuffer(row[0], dtype=np.float32) if row else None


# Each value: (Plotly location code, display name). Codes are ISO 3166-1 alpha-3
# for the world map (locationmode="ISO-3") and 2-letter postal codes for the US
# map (locationmode="USA-states").
COUNTRIES = {
    'afghanistan': ('AFG', 'Afghanistan'),
    'albania': ('ALB', 'Albania'),
    'algeria': ('DZA', 'Algeria'),
    'america': ('USA', 'United States'),
    'angola': ('AGO', 'Angola'),
    'antarctica': ('ATA', 'Antarctica'),
    'argentina': ('ARG', 'Argentina'),
    'armenia': ('ARM', 'Armenia'),
    'aruba': ('ABW', 'Aruba'),
    'australia': ('AUS', 'Australia'),
    'austria': ('AUT', 'Austria'),
    'azerbaijan': ('AZE', 'Azerbaijan'),
    'bahamas': ('BHS', 'Bahamas'),
    'bahrain': ('BHR', 'Bahrain'),
    'bangladesh': ('BGD', 'Bangladesh'),
    'barbados': ('BRB', 'Barbados'),
    'belarus': ('BLR', 'Belarus'),
    'belgium': ('BEL', 'Belgium'),
    'belize': ('BLZ', 'Belize'),
    'benin': ('BEN', 'Benin'),
    'bermuda': ('BMU', 'Bermuda'),
    'bhutan': ('BTN', 'Bhutan'),
    'bolivia': ('BOL', 'Bolivia'),
    'bosnia': ('BIH', 'Bosnia and Herzegovina'),
    'botswana': ('BWA', 'Botswana'),
    'brazil': ('BRA', 'Brazil'),
    'brunei': ('BRN', 'Brunei'),
    'bulgaria': ('BGR', 'Bulgaria'),
    'burkina_faso': ('BFA', 'Burkina Faso'),
    'burundi': ('BDI', 'Burundi'),
    'cambodia': ('KHM', 'Cambodia'),
    'cameroon': ('CMR', 'Cameroon'),
    'canada': ('CAN', 'Canada'),
    'cayman_islands': ('CYM', 'Cayman Islands'),
    'chad': ('TCD', 'Chad'),
    'chile': ('CHL', 'Chile'),
    'china': ('CHN', 'China'),
    'colombia': ('COL', 'Colombia'),
    'comoros': ('COM', 'Comoros'),
    'congo': ('COD', 'Dem. Rep. of the Congo'),
    'costa_rica': ('CRI', 'Costa Rica'),
    'croatia': ('HRV', 'Croatia'),
    'cuba': ('CUB', 'Cuba'),
    'cyprus': ('CYP', 'Cyprus'),
    'czech_republic': ('CZE', 'Czech Republic'),
    'denmark': ('DNK', 'Denmark'),
    'djibouti': ('DJI', 'Djibouti'),
    'dominican_republic': ('DOM', 'Dominican Republic'),
    'east_timor': ('TLS', 'Timor-Leste'),
    'ecuador': ('ECU', 'Ecuador'),
    'egypt': ('EGY', 'Egypt'),
    'el_salvador': ('SLV', 'El Salvador'),
    'england': ('GBR', 'United Kingdom'),
    'equatorial_guinea': ('GNQ', 'Equatorial Guinea'),
    'eritrea': ('ERI', 'Eritrea'),
    'estonia': ('EST', 'Estonia'),
    'ethiopia': ('ETH', 'Ethiopia'),
    'fiji': ('FJI', 'Fiji'),
    'finland': ('FIN', 'Finland'),
    'france': ('FRA', 'France'),
    'french_guiana': ('GUF', 'French Guiana'),
    'gabon': ('GAB', 'Gabon'),
    'gambia': ('GMB', 'Gambia'),
    'gaza_strip': ('PSE', 'Palestine'),
    'georgia': ('GEO', 'Georgia'),
    'germany': ('DEU', 'Germany'),
    'ghana': ('GHA', 'Ghana'),
    'goss': ('SSD', 'South Sudan'),
    'greece': ('GRC', 'Greece'),
    'greenland': ('GRL', 'Greenland'),
    'guam': ('GUM', 'Guam'),
    'guatemala': ('GTM', 'Guatemala'),
    'guinea': ('GIN', 'Guinea'),
    'guyana': ('GUY', 'Guyana'),
    'haiti': ('HTI', 'Haiti'),
    'honduras': ('HND', 'Honduras'),
    'hungary': ('HUN', 'Hungary'),
    'iceland': ('ISL', 'Iceland'),
    'india': ('IND', 'India'),
    'indonesia': ('IDN', 'Indonesia'),
    'iran': ('IRN', 'Iran'),
    'iraq': ('IRQ', 'Iraq'),
    'ireland': ('IRL', 'Ireland'),
    'israel': ('ISR', 'Israel'),
    'italy': ('ITA', 'Italy'),
    'ivory_coast': ('CIV', "Côte d'Ivoire"),
    'jamaica': ('JAM', 'Jamaica'),
    'japan': ('JPN', 'Japan'),
    'jersey': ('JEY', 'Jersey'),
    'jordan': ('JOR', 'Jordan'),
    'kazakhstan': ('KAZ', 'Kazakhstan'),
    'kenya': ('KEN', 'Kenya'),
    'kosovo': ('XKX', 'Kosovo'),
    'kuwait': ('KWT', 'Kuwait'),
    'kyrgyzstan': ('KGZ', 'Kyrgyzstan'),
    'laos': ('LAO', 'Laos'),
    'latvia': ('LVA', 'Latvia'),
    'lebanon': ('LBN', 'Lebanon'),
    'lesotho': ('LSO', 'Lesotho'),
    'liberia': ('LBR', 'Liberia'),
    'libya': ('LBY', 'Libya'),
    'lithuania': ('LTU', 'Lithuania'),
    'luxembourg': ('LUX', 'Luxembourg'),
    'macedonia': ('MKD', 'North Macedonia'),
    'madagascar': ('MDG', 'Madagascar'),
    'malawi': ('MWI', 'Malawi'),
    'malaysia': ('MYS', 'Malaysia'),
    'maldives': ('MDV', 'Maldives'),
    'mali': ('MLI', 'Mali'),
    'malta': ('MLT', 'Malta'),
    'mauritania': ('MRT', 'Mauritania'),
    'mauritius': ('MUS', 'Mauritius'),
    'mexico': ('MEX', 'Mexico'),
    'moldova': ('MDA', 'Moldova'),
    'monaco': ('MCO', 'Monaco'),
    'mongolia': ('MNG', 'Mongolia'),
    'montenegro': ('MNE', 'Montenegro'),
    'morocco': ('MAR', 'Morocco'),
    'mozambique': ('MOZ', 'Mozambique'),
    'myanmar': ('MMR', 'Myanmar'),
    'namibia': ('NAM', 'Namibia'),
    'nepal': ('NPL', 'Nepal'),
    'netherlands': ('NLD', 'Netherlands'),
    'new_zealand': ('NZL', 'New Zealand'),
    'nicaragua': ('NIC', 'Nicaragua'),
    'niger': ('NER', 'Niger'),
    'nigeria': ('NGA', 'Nigeria'),
    'north_korea': ('PRK', 'North Korea'),
    'norway': ('NOR', 'Norway'),
    'oman': ('OMN', 'Oman'),
    'pakistan': ('PAK', 'Pakistan'),
    'panama': ('PAN', 'Panama'),
    'papua': ('PNG', 'Papua New Guinea'),
    'paraguay': ('PRY', 'Paraguay'),
    'peru': ('PER', 'Peru'),
    'philippines': ('PHL', 'Philippines'),
    'poland': ('POL', 'Poland'),
    'portugal': ('PRT', 'Portugal'),
    'puerto_rico': ('PRI', 'Puerto Rico'),
    'qatar': ('QAT', 'Qatar'),
    'reunion': ('REU', 'Réunion'),
    'romania': ('ROU', 'Romania'),
    'russia': ('RUS', 'Russia'),
    'rwanda': ('RWA', 'Rwanda'),
    'samoa': ('WSM', 'Samoa'),
    'saudi_arabia': ('SAU', 'Saudi Arabia'),
    'senegal': ('SEN', 'Senegal'),
    'serbia': ('SRB', 'Serbia'),
    'sierra_leone': ('SLE', 'Sierra Leone'),
    'singapore': ('SGP', 'Singapore'),
    'slovakia': ('SVK', 'Slovakia'),
    'slovenia': ('SVN', 'Slovenia'),
    'solomon_islands': ('SLB', 'Solomon Islands'),
    'somalia': ('SOM', 'Somalia'),
    'south_africa': ('ZAF', 'South Africa'),
    'south_korea': ('KOR', 'South Korea'),
    'spain': ('ESP', 'Spain'),
    'sri_lanka': ('LKA', 'Sri Lanka'),
    'sudan': ('SDN', 'Sudan'),
    'suriname': ('SUR', 'Suriname'),
    'svalbard': ('SJM', 'Svalbard'),
    'swaziland': ('SWZ', 'Eswatini'),
    'sweden': ('SWE', 'Sweden'),
    'switzerland': ('CHE', 'Switzerland'),
    'syria': ('SYR', 'Syria'),
    'taiwan': ('TWN', 'Taiwan'),
    'tajikistan': ('TJK', 'Tajikistan'),
    'tanzania': ('TZA', 'Tanzania'),
    'thailand': ('THA', 'Thailand'),
    'togo': ('TGO', 'Togo'),
    'tonga': ('TON', 'Tonga'),
    'tunisia': ('TUN', 'Tunisia'),
    'turkey': ('TUR', 'Turkey'),
    'turkmenistan': ('TKM', 'Turkmenistan'),
    'uae': ('ARE', 'United Arab Emirates'),
    'uganda': ('UGA', 'Uganda'),
    'ukraine': ('UKR', 'Ukraine'),
    'uruguay': ('URY', 'Uruguay'),
    'uzbekistan': ('UZB', 'Uzbekistan'),
    'vatican': ('VAT', 'Vatican City'),
    'venezuela': ('VEN', 'Venezuela'),
    'vietnam': ('VNM', 'Vietnam'),
    'virgin_islands': ('VIR', 'U.S. Virgin Islands'),
    'western_sahara': ('ESH', 'Western Sahara'),
    'yemen': ('YEM', 'Yemen'),
    'zambia': ('ZMB', 'Zambia'),
    'zimbabwe': ('ZWE', 'Zimbabwe'),
}

STATES = {
    'alabama': ('AL', 'Alabama'),
    'alaska': ('AK', 'Alaska'),
    'arizona': ('AZ', 'Arizona'),
    'arkansas': ('AR', 'Arkansas'),
    'california': ('CA', 'California'),
    'colorado': ('CO', 'Colorado'),
    'connecticut': ('CT', 'Connecticut'),
    'delaware': ('DE', 'Delaware'),
    'florida': ('FL', 'Florida'),
    'georgia': ('GA', 'Georgia'),
    'hawaii': ('HI', 'Hawaii'),
    'idaho': ('ID', 'Idaho'),
    'illinois': ('IL', 'Illinois'),
    'indiana': ('IN', 'Indiana'),
    'iowa': ('IA', 'Iowa'),
    'kansas': ('KS', 'Kansas'),
    'kentucky': ('KY', 'Kentucky'),
    'louisiana': ('LA', 'Louisiana'),
    'maine': ('ME', 'Maine'),
    'maryland': ('MD', 'Maryland'),
    'massachusetts': ('MA', 'Massachusetts'),
    'michigan': ('MI', 'Michigan'),
    'minnesota': ('MN', 'Minnesota'),
    'mississippi': ('MS', 'Mississippi'),
    'missouri': ('MO', 'Missouri'),
    'montana': ('MT', 'Montana'),
    'nebraska': ('NE', 'Nebraska'),
    'nevada': ('NV', 'Nevada'),
    'new_hampshire': ('NH', 'New Hampshire'),
    'new_jersey': ('NJ', 'New Jersey'),
    'new_mexico': ('NM', 'New Mexico'),
    'new_york': ('NY', 'New York'),
    'north_carolina': ('NC', 'North Carolina'),
    'north_dakota': ('ND', 'North Dakota'),
    'ohio': ('OH', 'Ohio'),
    'oklahoma': ('OK', 'Oklahoma'),
    'oregon': ('OR', 'Oregon'),
    'pennsylvania': ('PA', 'Pennsylvania'),
    'rhode_island': ('RI', 'Rhode Island'),
    'south_carolina': ('SC', 'South Carolina'),
    'south_dakota': ('SD', 'South Dakota'),
    'tennessee': ('TN', 'Tennessee'),
    'texas': ('TX', 'Texas'),
    'utah': ('UT', 'Utah'),
    'vermont': ('VT', 'Vermont'),
    'virginia': ('VA', 'Virginia'),
    'washington': ('WA', 'Washington'),
    'west_virginia': ('WV', 'West Virginia'),
    'wisconsin': ('WI', 'Wisconsin'),
    'wyoming': ('WY', 'Wyoming'),
}

# Codes for regions Plotly may render but for which we don't have a direct
# Word2Vec lookup. Each entry: (target_code, source_code, display_name).
# When the target is missing from the score map, copy the source's score.
BACKFILL = {
    'world': [
        ('SJM', 'NOR', 'Svalbard'),
        ('COG', 'COD', 'Republic of the Congo'),
        ('CAF', 'TCD', 'Central African Republic'),
    ],
    'usa': [],
}

EXAMPLES = {
    'world': ['cricket', 'vodka', 'crisis', 'carnival', 'asia', 'desert', 'malaria', 'coffee'],
    'usa': ['coastal', 'desert', 'germany', 'heroine', 'hockey', 'mountain', 'soul', 'sunny'],
}

PREVIEWS = {'world': 'preview.jpg', 'usa': 'US-preview.jpg'}
LOCATIONMODE = {'world': 'ISO-3', 'usa': 'USA-states'}
PROJECTION = {'world': 'natural earth', 'usa': 'albers usa'}
MAP_NAMES = {'world': 'world', 'usa': 'U.S. states'}

HI = 240
LO = 80
COLORS = {
    'red': (HI, LO, LO),
    'blue': (LO, LO, HI),
    'yellow': (HI, HI, LO),
    'green': (LO, HI, LO),
}


class MapOf(Project):

    SCORES = {}        # map_id -> {region_key: vec}
    COLOR_SCORES = {}  # color_name -> vec

    def _load_vectors(self, keys):
        conn = _conn()
        out = {}
        for k in keys:
            v = _vec_for_lower(conn, k.lower())
            if v is not None:
                out[k] = v
        return out

    def _region_scores(self, map_id):
        if map_id not in self.SCORES:
            regions = COUNTRIES if map_id == 'world' else STATES
            self.SCORES[map_id] = self._load_vectors(regions.keys())
        return self.SCORES[map_id]

    def _color_scores(self):
        if not self.COLOR_SCORES:
            self.COLOR_SCORES = self._load_vectors(COLORS.keys())
        return self.COLOR_SCORES

    def _score_against(self, word, item_scores, key_to_output, backfill):
        """Returns {output_code: normalized_score} for the given word."""
        word_vec = _vec_for_lower(_conn(), word)
        if word_vec is None:
            return {}
        raw = {
            key_to_output[k]: scipy.spatial.distance.cosine(v, word_vec)
            for k, v in item_scores.items() if k in key_to_output
        }
        for to_code, from_code, _name in backfill:
            if to_code not in raw and from_code in raw:
                raw[to_code] = raw[from_code]
        if not raw:
            return raw
        mx = max(raw.values())
        mn = min(raw.values())
        if mx == mn:
            return {k: 0.5 for k in raw}
        return {k: 1 - (v - mn) / (mx - mn) for k, v in raw.items()}

    def fill_dict(self, request, d):
        map_id = request.GET.get('map', 'world')
        if map_id not in ('world', 'usa'):
            map_id = 'world'
        regions = COUNTRIES if map_id == 'world' else STATES
        key_to_code = {k: v[0] for k, v in regions.items()}
        backfill = BACKFILL[map_id]
        code_to_name = {v[0]: v[1] for v in regions.values()}
        for to_code, _from, name in backfill:
            code_to_name.setdefault(to_code, name)

        word = request.GET.get('word', 'pasta')
        word_processed = word.lower().replace(' ', '_')

        if '-' in word_processed:
            positive, negative = word_processed.split('-', 1)
        else:
            positive, negative = word_processed, ''

        region_scores = self._region_scores(map_id)
        scores = self._score_against(positive, region_scores, key_to_code, backfill)
        if negative:
            neg = self._score_against(negative, region_scores, key_to_code, backfill)
            for k, v in neg.items():
                scores[k] = scores.get(k, 0) - v

        d['word'] = word
        d['map_id'] = map_id
        d['map_name'] = MAP_NAMES[map_id]
        d['other_map_id'] = 'usa' if map_id == 'world' else 'world'
        d['other_map_name'] = MAP_NAMES[d['other_map_id']]
        d['examples'] = EXAMPLES[map_id]
        d['preview'] = PREVIEWS[map_id]
        d['locationmode'] = LOCATIONMODE[map_id]
        d['projection_type'] = PROJECTION[map_id]

        if not scores:
            d['error'] = '"%s" could not be found. Try a different word.' % word
            return

        color_scores = self._score_against(positive, self._color_scores(), {c: c for c in COLORS}, [])
        color = max(color_scores, key=color_scores.get) if color_scores else 'green'
        rgb_to = COLORS[color]
        rgb_from = [int(128 - x / 4) for x in rgb_to]

        sorted_scores = sorted(scores.items(), key=lambda t: t[1], reverse=True)
        d['data'] = sorted_scores
        d['codes_json'] = json.dumps([c for c, _ in sorted_scores])
        d['scores_json'] = json.dumps([round(float(s), 4) for _, s in sorted_scores])
        d['names_json'] = json.dumps([code_to_name.get(c, c) for c, _ in sorted_scores])
        d['color'] = color
        d['rgb_from'] = ''.join('%02x' % c for c in rgb_from)
        d['rgb_to'] = ''.join('%02x' % c for c in rgb_to)

    def handle_request(self, handler, request):
        if handler == 'autocomplete':
            q = request.GET.get('q', '')
            if '-' in q:
                prefix, q = q.split('-', 1)
                prefix += '-'
            else:
                prefix = ''
            if len(q) < 3:
                return HttpResponse('[]')
            q = q.lower().strip().replace(' ', '_')
            seen = set()
            matches = []
            for (word,) in _conn().execute(
                "SELECT word_lower FROM words_lower WHERE word_lower LIKE ? ORDER BY word_lower LIMIT 10",
                (q + '%',),
            ):
                word = word.replace('_', ' ')
                if word in seen:
                    continue
                seen.add(word)
                matches.append(prefix + word)
            return HttpResponse(json.dumps(matches[:5]))
