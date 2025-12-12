"""
Art-Constrained Prompt Mutations for Alien Art Discovery
=========================================================
Only allows mutations that stay within the "painting" schema.
This ensures all outputs are recognizably art-like while still novel.

Key insight from cognitive science:
  True surprise = violating expectations WITHIN a familiar frame
  Not: random weirdness (grocery store landscapes)
  But: unexpected art (a landscape with impossible light)
"""

import random
from typing import List

# =============================================================================
# ART-APPROPRIATE COMPONENTS ONLY
# =============================================================================

# Subjects found in WikiArt
SUBJECTS = [
    # Landscapes (most common in WikiArt)
    "landscape", "seascape", "countryside", "mountain vista", "forest clearing",
    "river valley", "sunset horizon", "garden scene", "pastoral meadow",
    "coastal cliffs", "rolling hills", "autumn forest", "winter scene",
    # Portraits & Figures
    "portrait", "figure study", "self-portrait", "group portrait",
    # Still Life
    "still life", "flowers in vase", "fruit arrangement", "table setting",
    # Architecture
    "cathedral interior", "village street", "ancient bridge", "castle ruins",
    "city skyline", "harbor view", "market square",
    # Nature details
    "flowers", "trees", "clouds", "water reflection", "rocky shore",
    "wheat field", "vineyard", "orchard",
]

# Art-appropriate adjectives
ADJECTIVES = [
    # Traditional art terms
    "impressionist", "romantic", "classical", "pastoral", "serene",
    "dramatic", "luminous", "atmospheric", "delicate", "bold",
    "melancholic", "tranquil", "vibrant", "muted", "golden",
    # Subtle mood shifts
    "dreamlike", "hazy", "ethereal", "mysterious", "timeless",
    "forgotten", "distant", "fading", "emerging", "dissolving",
    "haunting", "contemplative", "intimate", "vast", "quiet",
]

# Major art styles/movements in WikiArt
STYLES = [
    "impressionism", "post-impressionism", "romanticism", "realism",
    "baroque", "renaissance", "expressionism", "symbolism",
    "tonalism", "luminism", "hudson river school", "barbizon school",
    "pre-raphaelite", "naturalism", "academic art",
    # Artist references
    "in the style of Monet", "in the style of Turner", "in the style of Constable",
    "in the style of Corot", "in the style of Caspar David Friedrich",
    "in the style of Bierstadt", "in the style of Cézanne",
    "in the style of van Gogh", "in the style of Renoir",
]

# Time and lighting (crucial for art)
TIMES_OF_DAY = [
    "at dawn", "at sunrise", "in early morning light", "at midday",
    "in afternoon light", "at golden hour", "at sunset", "at dusk",
    "at twilight", "under moonlight", "at midnight",
]

LIGHTING_CONDITIONS = [
    "with soft diffused light", "with dramatic side lighting",
    "with warm golden light", "with cool blue shadows",
    "backlit against the sky", "with dappled sunlight",
    "in chiaroscuro lighting", "with reflected light from water",
    "with atmospheric haze", "in silvery morning mist",
]

WEATHER_CONDITIONS = [
    "in gentle rain", "under stormy skies", "in morning mist",
    "with gathering clouds", "after the storm", "in autumn haze",
    "under clear blue skies", "with threatening clouds",
    "in soft fog", "with snow falling",
]

# Emotional tones (art-appropriate)
MOODS = [
    "with a sense of solitude", "with quiet contemplation",
    "with romantic grandeur", "with melancholic beauty",
    "with serene stillness", "with dramatic tension",
    "with nostalgic warmth", "with mysterious depth",
    "with sublime awe", "with intimate tenderness",
]

# Techniques and media
TECHNIQUES = [
    "oil on canvas", "watercolor", "gouache", "pastel",
    "charcoal study", "ink wash", "tempera", "fresco style",
]

# Compositional elements
COMPOSITIONS = [
    "with deep perspective", "in intimate close-up",
    "as a panoramic view", "from an elevated viewpoint",
    "with strong diagonal lines", "with central focal point",
    "with atmospheric depth", "with layered planes",
]


# =============================================================================
# ART-CONSTRAINED MUTATION OPERATORS
# =============================================================================

def mutate_subject(prompt: str) -> str:
    """Swap the subject for another art-appropriate subject."""
    for subj in SUBJECTS:
        if subj.lower() in prompt.lower():
            new_subj = random.choice([s for s in SUBJECTS if s != subj])
            return prompt.lower().replace(subj.lower(), new_subj, 1)
    # If no subject found, add one
    return f"a painting of a {random.choice(SUBJECTS)}, {prompt}"


def mutate_adjective(prompt: str) -> str:
    """Change or add an art-appropriate adjective."""
    for adj in ADJECTIVES:
        if adj.lower() in prompt.lower():
            new_adj = random.choice([a for a in ADJECTIVES if a != adj])
            return prompt.lower().replace(adj.lower(), new_adj, 1)
    # Add adjective at start
    return f"{random.choice(ADJECTIVES)} {prompt}"


def mutate_style(prompt: str) -> str:
    """Change the artistic style."""
    # Remove existing style references
    result = prompt
    for style in STYLES:
        if style.lower() in result.lower():
            result = result.lower().replace(style.lower(), "").strip(", ")
    # Add new style
    new_style = random.choice(STYLES)
    return f"{result}, {new_style}"


def mutate_lighting(prompt: str) -> str:
    """Change the lighting/time of day."""
    # Remove existing time references
    result = prompt
    for time in TIMES_OF_DAY:
        if time.lower() in result.lower():
            result = result.lower().replace(time.lower(), "").strip(", ")
    # Add new lighting
    if random.random() < 0.5:
        return f"{result} {random.choice(TIMES_OF_DAY)}"
    else:
        return f"{result}, {random.choice(LIGHTING_CONDITIONS)}"


def mutate_weather(prompt: str) -> str:
    """Add or change weather conditions."""
    # Remove existing weather
    result = prompt
    for weather in WEATHER_CONDITIONS:
        if weather.lower() in result.lower():
            result = result.lower().replace(weather.lower(), "").strip(", ")
    return f"{result}, {random.choice(WEATHER_CONDITIONS)}"


def mutate_mood(prompt: str) -> str:
    """Add or change the emotional mood."""
    # Remove existing mood
    result = prompt
    for mood in MOODS:
        if mood.lower() in result.lower():
            result = result.lower().replace(mood.lower(), "").strip(", ")
    return f"{result}, {random.choice(MOODS)}"


def mutate_technique(prompt: str) -> str:
    """Add or change the artistic technique/medium."""
    # Remove existing technique
    result = prompt
    for tech in TECHNIQUES:
        if tech.lower() in result.lower():
            result = result.lower().replace(tech.lower(), "").strip(", ")
    return f"{result}, {random.choice(TECHNIQUES)}"


def mutate_composition(prompt: str) -> str:
    """Add compositional guidance."""
    return f"{prompt}, {random.choice(COMPOSITIONS)}"


def mutate_blend_art_subjects(prompt: str) -> str:
    """Blend two art subjects (still within art domain)."""
    subj1 = random.choice(SUBJECTS)
    subj2 = random.choice([s for s in SUBJECTS if s != subj1])
    adj = random.choice(ADJECTIVES)
    return f"{adj} painting blending {subj1} and {subj2}"


def mutate_simplify(prompt: str) -> str:
    """Simplify by removing a clause (maintains art focus)."""
    parts = [p.strip() for p in prompt.split(",") if p.strip()]
    if len(parts) > 2:
        # Keep at least core subject
        parts.pop(random.randrange(1, len(parts)))
        return ", ".join(parts)
    return prompt


def mutate_intensify(prompt: str) -> str:
    """Intensify the artistic qualities."""
    intensifiers = [
        "deeply", "profoundly", "hauntingly", "strikingly",
        "remarkably", "extraordinarily", "intensely", "subtly",
    ]
    return f"{random.choice(intensifiers)} {prompt}"


# =============================================================================
# MAIN MUTATION FUNCTION
# =============================================================================

def mutate_art_prompt(prompt: str) -> str:
    """
    Apply a random ART-CONSTRAINED mutation.
    All mutations keep the output within the "painting" schema.
    """
    mutations = [
        (mutate_subject, 0.15),
        (mutate_adjective, 0.15),
        (mutate_style, 0.15),
        (mutate_lighting, 0.12),
        (mutate_weather, 0.08),
        (mutate_mood, 0.10),
        (mutate_technique, 0.05),
        (mutate_composition, 0.05),
        (mutate_blend_art_subjects, 0.08),
        (mutate_simplify, 0.04),
        (mutate_intensify, 0.03),
    ]
    
    # Weighted random selection
    total = sum(w for _, w in mutations)
    r = random.random() * total
    cumsum = 0
    for mutator, weight in mutations:
        cumsum += weight
        if r <= cumsum:
            return mutator(prompt)
    
    return mutations[0][0](prompt)


# =============================================================================
# SEED PROMPTS (All art-style)
# =============================================================================

SEED_PROMPTS_CONSTRAINED = [
    "a painting of a landscape",
    "a painting of a sunset over mountains",
    "a painting of a forest clearing",
    "a painting of a river valley",
    "an impressionist landscape with trees",
    "a romantic painting of a stormy sea",
    "a pastoral scene with meadows",
    "a painting of autumn trees by a lake",
    "a misty morning landscape",
    "a painting of clouds over hills",
    "a classical landscape with ruins",
    "a painting of a garden in bloom",
    "a seascape with rocky shore",
    "a painting of a quiet village",
    "a luminous sunset painting",
]


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ART-CONSTRAINED MUTATION TEST")
    print("=" * 60)
    
    base = "a painting of a landscape"
    print(f"\nBase prompt: {base}")
    print("\nMutations:")
    
    for i in range(15):
        mutated = mutate_art_prompt(base)
        # Chain mutations for more variation
        if random.random() < 0.3:
            mutated = mutate_art_prompt(mutated)
        print(f"  {i+1}. {mutated}")
    
    print("\n" + "=" * 60)
    print("All mutations stay within the 'painting' schema!")
