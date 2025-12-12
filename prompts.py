"""
Prompt Bank and Mutation Operators for Alien Art Discovery
===========================================================
Structured prompt generation and evolution for exploring concept space.
"""

import random
from typing import Optional
from dataclasses import dataclass, field

# =============================================================================
# PROMPT COMPONENTS - Curated lists for structured generation
# =============================================================================

SUBJECTS = [
    # Landscapes (WikiArt common)
    "landscape", "seascape", "countryside", "mountain vista", "forest clearing",
    "river valley", "sunset horizon", "garden scene", "pastoral meadow",
    # Portraits & Figures (WikiArt common)
    "portrait", "figure study", "still life", "interior scene",
    # Architecture (WikiArt common)  
    "cathedral", "village street", "bridge", "ruins", "castle",
    # Nature (WikiArt common)
    "flowers", "trees", "clouds", "water reflection", "rocky shore",
]

ADJECTIVES = [
    # Traditional art adjectives (familiar frame)
    "impressionist", "romantic", "classical", "pastoral", "serene",
    "dramatic", "luminous", "atmospheric", "delicate", "bold",
    "melancholic", "tranquil", "vibrant", "muted", "golden",
    # Slightly unusual (for subtle deviation)
    "dreamlike", "hazy", "ethereal", "mysterious", "timeless",
    "forgotten", "distant", "fading", "emerging", "dissolving",
]

MATERIALS = [
    "smoke", "glass", "coral", "circuit boards", "nebula gas",
    "liquid metal", "bone", "vines", "clockwork", "ice",
    "obsidian", "amber", "mycelium", "plasma", "silk",
    "chitin", "marble", "rust", "shadows", "light",
    "mercury", "crystals", "flesh", "stone", "water",
]

STYLES = [
    # Major WikiArt movements (familiar frames)
    "impressionism", "post-impressionism", "romanticism", "realism",
    "baroque", "renaissance", "expressionism", "symbolism",
    "tonalism", "luminism", "hudson river school", "barbizon school",
    # Specific artists in WikiArt
    "in the style of Monet", "in the style of Turner", "in the style of Constable",
    "in the style of Corot", "in the style of Caspar David Friedrich",
    "in the style of Bierstadt", "in the style of Cézanne",
]

PHYSICS_LAWS = [
    "where gravity pulls sideways",
    "where time flows backwards",
    "in non-Euclidean space",
    "where light bends in spirals",
    "at the event horizon",
    "in a world with no shadows",
    "where scale is meaningless",
    "in inverted colors",
    "where matter phases through matter",
    "at absolute zero",
    "in a universe with extra dimensions",
    "where causality is reversed",
]

ENVIRONMENTS = [
    "floating in void", "underwater", "in deep space", "underground",
    "inside a living creature", "between dimensions", "at quantum scale",
    "in a dream", "after the apocalypse", "before time began",
    "inside a computer", "in parallel universe", "at the edge of reality",
]

# =============================================================================
# ART-STYLE COMPONENTS - For WikiArt-comparable familiar frames
# =============================================================================

TIMES_OF_DAY = [
    "dawn", "sunrise", "morning", "midday", "afternoon",
    "golden hour", "sunset", "dusk", "twilight", "night",
    "moonrise", "midnight",
]

WEATHER_CONDITIONS = [
    "mist", "fog", "rain", "storm", "sunshine",
    "overcast skies", "clearing clouds", "snow", "haze",
    "gentle breeze", "still air", "approaching storm",
]

LIGHTING_CONDITIONS = [
    "soft light", "dramatic lighting", "backlighting", "diffused light",
    "warm sunlight", "cool shadows", "dappled light", "golden light",
    "silvery moonlight", "reflected light", "ambient glow",
]

EMOTIONS = [
    "nostalgic", "melancholic", "peaceful", "unsettling",
    "sublime", "intimate", "vast", "lonely", "hopeful",
    "mysterious", "forgotten", "eternal",
]

# =============================================================================
# WOW-FACTOR COMPONENTS - Mundane anchors for cognitive dissonance
# =============================================================================

MUNDANE_ANCHORS = [
    "office cubicle", "grocery store", "parking lot", "kitchen sink",
    "laundry basket", "suburban house", "dentist waiting room",
    "highway overpass", "vending machine", "bathroom tile",
    "filing cabinet", "school cafeteria", "gas station",
    "elevator interior", "hotel hallway", "airport terminal",
    "bus stop bench", "storage closet", "garage door",
    "shopping cart", "office printer", "ceiling fan",
    "stairwell", "washing machine", "refrigerator interior",
]

EMOTIONAL_CONTRADICTIONS = [
    "uncomfortably beautiful",
    "nostalgically futuristic",
    "aggressively peaceful",
    "intimately vast",
    "warmly terrifying",
    "precisely chaotic",
    "silently deafening",
    "comfortingly disturbing",
    "joyfully melancholic",
    "violently serene",
    "anciently modern",
    "impossibly familiar",
]

BANALITY_INJECTIONS = [
    "but it's somehow also a kitchen appliance",
    "with a 'for sale' sign",
    "that your grandmother would recognize",
    "in the style of a corporate PowerPoint",
    "as seen in a home improvement catalog",
    "with a motivational poster aesthetic",
    "that feels like a dentist's waiting room",
    "with an 'out of order' sign",
    "that belongs in a hotel lobby",
    "rendered as office furniture",
    "with a price tag still attached",
    "that smells like a new car",
    "sponsored by a fast food chain",
    "with IKEA assembly instructions",
]

# =============================================================================
# NEGATIVE PROMPTS (for evolution)
# =============================================================================

NEGATIVE_PROMPTS = [
    "blurry, low quality, text, watermark",
    "realistic, photographic, normal, ordinary",
    "humans, faces, animals, recognizable objects",
    "symmetry, repetition, patterns",
    "bright colors, cheerful, pleasant",
]

NEGATIVE_CONCEPTS = [
    "no symmetry",
    "no straight lines",
    "no recognizable forms",
    "no humans or faces",
    "no repetition",
    "without color",
    "without perspective",
    "devoid of structure",
    "lacking familiar objects",
    "absent of realism",
]


# =============================================================================
# LLM-BASED MUTATION (Optional - requires API)
# =============================================================================

def mutate_with_llm(prompt: str, api_key: Optional[str] = None) -> str:
    """
    Use Claude API to generate a stranger version of the prompt.
    Falls back to heuristic mutation if no API key.
    """
    if api_key is None:
        # Fallback to heuristic
        return mutate_prompt(prompt)
    
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        
        system = """You are a creative mutation engine for generating strange, novel art prompts.
Given an input prompt, make it STRANGER in one of these ways:
- Combine with an unrelated scientific field
- Add impossible physics
- Merge with an unexpected concept
- Make it more abstract or alien

Output ONLY the new prompt, nothing else. Keep it concise (under 20 words)."""
        
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            system=system,
            messages=[
                {"role": "user", "content": f"Make this stranger: {prompt}"}
            ]
        )
        
        return message.content[0].text.strip()
    
    except Exception as e:
        print(f"LLM mutation failed: {e}, falling back to heuristic")
        return mutate_prompt(prompt)


def mutate_with_llm_batch(prompts: list[str], api_key: Optional[str] = None) -> list[str]:
    """Batch LLM mutation for efficiency."""
    if api_key is None:
        return [mutate_prompt(p) for p in prompts]
    
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        
        system = """You are a creative mutation engine. Given multiple prompts, make each one STRANGER.
Combine with unrelated fields, add impossible physics, or make more alien/abstract.
Output one mutated prompt per line, nothing else. Keep each under 20 words."""
        
        prompt_list = "\n".join([f"{i+1}. {p}" for i, p in enumerate(prompts)])
        
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system=system,
            messages=[
                {"role": "user", "content": f"Mutate these prompts:\n{prompt_list}"}
            ]
        )
        
        # Parse response
        lines = message.content[0].text.strip().split("\n")
        results = []
        for line in lines:
            # Remove numbering if present
            clean = line.strip()
            if clean and clean[0].isdigit():
                clean = clean.split(".", 1)[-1].strip()
            if clean:
                results.append(clean)
        
        # Pad with heuristic mutations if needed
        while len(results) < len(prompts):
            results.append(mutate_prompt(prompts[len(results)]))
        
        return results[:len(prompts)]
    
    except Exception as e:
        print(f"Batch LLM mutation failed: {e}, falling back to heuristic")
        return [mutate_prompt(p) for p in prompts]


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

TEMPLATES = [
    # Basic art prompts (familiar frames)
    "a painting of a {subject}",
    "a {adjective} painting of a {subject}",
    "a {adjective} {subject}, {style}",
    
    # With atmosphere/mood
    "a {adjective} painting of a {subject} at {time}",
    "a painting of a {subject} in {weather}",
    "a {adjective} {subject} with {lighting}",
    
    # Compositional variations
    "a painting of a {subject} with {subject2} in the distance",
    "a {adjective} view of a {subject}",
    "a painting looking across a {subject}",
    
    # Style emphasis
    "{style} painting of a {subject}",
    "a {subject} rendered in {style}",
    
    # Subtle strangeness (deviation within frame)
    "a {adjective} painting of a {subject} that feels {emotion}",
    "a painting of a {subject} as if seen in a dream",
    "a {subject} painted from memory",
]


# =============================================================================
# PROMPT GENERATION
# =============================================================================

def generate_random_prompt() -> str:
    """Generate a random prompt from templates."""
    template = random.choice(TEMPLATES)
    
    # Fill in the template with art-style components
    prompt = template.format(
        adjective=random.choice(ADJECTIVES),
        subject=random.choice(SUBJECTS),
        subject2=random.choice(SUBJECTS),
        material=random.choice(MATERIALS),
        style=random.choice(STYLES),
        physics=random.choice(PHYSICS_LAWS),
        environment=random.choice(ENVIRONMENTS),
        mundane=random.choice(MUNDANE_ANCHORS),
        contradiction=random.choice(EMOTIONAL_CONTRADICTIONS),
        # New art-style components
        time=random.choice(TIMES_OF_DAY),
        weather=random.choice(WEATHER_CONDITIONS),
        lighting=random.choice(LIGHTING_CONDITIONS),
        emotion=random.choice(EMOTIONS),
    )
    
    return prompt


def generate_prompt_batch(n: int) -> list[str]:
    """Generate n unique random prompts."""
    prompts = set()
    while len(prompts) < n:
        prompts.add(generate_random_prompt())
    return list(prompts)


# =============================================================================
# MUTATION OPERATORS
# =============================================================================

def _parse_prompt_parts(prompt: str) -> dict:
    """Loosely parse a prompt into components (best effort)."""
    # This is fuzzy - we just try to identify what might be swappable
    parts = {
        "adjectives": [],
        "subjects": [],
        "materials": [],
        "styles": [],
    }
    
    words = prompt.lower().split()
    
    for adj in ADJECTIVES:
        if adj.lower() in prompt.lower():
            parts["adjectives"].append(adj)
    
    for subj in SUBJECTS:
        if subj.lower() in prompt.lower():
            parts["subjects"].append(subj)
            
    for mat in MATERIALS:
        if mat.lower() in prompt.lower():
            parts["materials"].append(mat)
            
    for style in STYLES:
        if style.lower() in prompt.lower():
            parts["styles"].append(style)
    
    return parts


def mutate_substitute_word(prompt: str) -> str:
    """Substitute one component with another from the same category."""
    parts = _parse_prompt_parts(prompt)
    
    # Try substitution in order of preference
    if parts["adjectives"] and random.random() < 0.4:
        old = random.choice(parts["adjectives"])
        new = random.choice([a for a in ADJECTIVES if a != old])
        return prompt.replace(old, new, 1)
    
    if parts["subjects"] and random.random() < 0.4:
        old = random.choice(parts["subjects"])
        new = random.choice([s for s in SUBJECTS if s != old])
        return prompt.replace(old, new, 1)
    
    if parts["materials"] and random.random() < 0.3:
        old = random.choice(parts["materials"])
        new = random.choice([m for m in MATERIALS if m != old])
        return prompt.replace(old, new, 1)
    
    # Fallback: add a random adjective
    return f"{random.choice(ADJECTIVES)} {prompt}"


def mutate_add_modifier(prompt: str) -> str:
    """Add a new modifier to the prompt."""
    modifiers = [
        f", {random.choice(STYLES)}",
        f" {random.choice(PHYSICS_LAWS)}",
        f" {random.choice(ENVIRONMENTS)}",
        f", made of {random.choice(MATERIALS)}",
        f", {random.choice(ADJECTIVES)} and {random.choice(ADJECTIVES)}",
    ]
    return prompt + random.choice(modifiers)


def mutate_blend_concepts(prompt: str) -> str:
    """Create a hybrid by blending with another concept."""
    parts = _parse_prompt_parts(prompt)
    
    if parts["subjects"]:
        # "X that is also Y"
        new_subject = random.choice([s for s in SUBJECTS if s not in parts["subjects"]])
        return f"{prompt}, merged with {new_subject}"
    else:
        # Generate fresh hybrid
        return f"{random.choice(ADJECTIVES)} hybrid of {random.choice(SUBJECTS)} and {random.choice(SUBJECTS)}"


def mutate_change_physics(prompt: str) -> str:
    """Add or change the physics/environment."""
    # Remove existing physics if present
    for phys in PHYSICS_LAWS:
        if phys.lower() in prompt.lower():
            prompt = prompt.replace(phys, "")
    
    # Add new physics
    return f"{prompt.strip()} {random.choice(PHYSICS_LAWS)}"


def mutate_change_style(prompt: str) -> str:
    """Change or add artistic style."""
    # Remove existing style if present
    for style in STYLES:
        if style.lower() in prompt.lower():
            prompt = prompt.replace(style, "")
            prompt = prompt.replace(", ,", ",").replace("  ", " ").strip(", ")
    
    # Add new style
    return f"{prompt.strip()}, {random.choice(STYLES)}"


def mutate_simplify(prompt: str) -> str:
    """Simplify by removing a clause (for diversity)."""
    # Split on commas and remove one part
    parts = [p.strip() for p in prompt.split(",") if p.strip()]
    if len(parts) > 1:
        parts.pop(random.randrange(len(parts)))
        return ", ".join(parts)
    return prompt


def mutate_negate(prompt: str) -> str:
    """Add a negation or 'without' clause."""
    negations = [
        "without symmetry",
        "without recognizable forms",
        "devoid of life",
        "with no straight lines",
        "lacking color",
        "without perspective",
        "formless yet structured",
    ]
    return f"{prompt}, {random.choice(negations)}"


def mutate_concept_fusion(prompt: str) -> str:
    """Fuse two unrelated concepts for conceptual leaps."""
    fusions = [
        ("quantum mechanics", "gothic architecture"),
        ("deep sea biology", "space exploration"),
        ("microscopic organisms", "cosmic structures"),
        ("ancient mythology", "cybernetics"),
        ("musical notation", "topological spaces"),
        ("neural networks", "geological formations"),
        ("time dilation", "organic growth"),
        ("fluid dynamics", "crystalline structures"),
        ("electromagnetic fields", "dream logic"),
        ("evolutionary biology", "impossible geometry"),
    ]
    concept_a, concept_b = random.choice(fusions)
    return f"{prompt}, fusion of {concept_a} and {concept_b}"


def mutate_scientific_twist(prompt: str) -> str:
    """Add a scientific/mathematical twist."""
    twists = [
        "rendered as a phase diagram",
        "visualized through electron microscopy",
        "mapped onto a hyperbolic plane",
        "expressed as a quantum superposition",
        "following fibonacci spiral patterns",
        "undergoing entropy dissolution",
        "in thermodynamic equilibrium",
        "as seen through x-ray crystallography",
        "modeled by strange attractors",
        "obeying non-commutative geometry",
    ]
    return f"{prompt}, {random.choice(twists)}"


def mutate_perspective_shift(prompt: str) -> str:
    """Shift the perspective radically."""
    perspectives = [
        "from the perspective of a photon",
        "as perceived by an AI",
        "viewed from inside a black hole",
        "through the eyes of a microorganism",
        "from beyond the observable universe",
        "as remembered by a dying star",
        "witnessed by probability itself",
        "from the substrate of reality",
    ]
    return f"{prompt}, {random.choice(perspectives)}"


def mutate_banality_injection(prompt: str) -> str:
    """Inject mundane elements to create cognitive dissonance and wow-factor."""
    return f"{prompt}, {random.choice(BANALITY_INJECTIONS)}"


def mutate_emotional_contradiction(prompt: str) -> str:
    """Add emotional contradiction for unsettling beauty."""
    return f"{random.choice(EMOTIONAL_CONTRADICTIONS)} {prompt}"


def mutate_mundane_collision(prompt: str) -> str:
    """Collide the prompt with something utterly mundane."""
    collisions = [
        f"{prompt}, but it's actually a {random.choice(MUNDANE_ANCHORS)}",
        f"{random.choice(MUNDANE_ANCHORS)} that dreams of being {prompt}",
        f"{prompt} found inside a {random.choice(MUNDANE_ANCHORS)}",
        f"{prompt} designed by someone who only knows {random.choice(MUNDANE_ANCHORS)}s",
    ]
    return random.choice(collisions)


def mutate_prompt(prompt: str) -> str:
    """Apply a random mutation to a prompt."""
    mutations = [
        (mutate_substitute_word, 0.12),
        (mutate_add_modifier, 0.10),
        (mutate_blend_concepts, 0.10),
        (mutate_change_physics, 0.08),
        (mutate_change_style, 0.08),
        (mutate_simplify, 0.04),
        (mutate_negate, 0.04),
        (mutate_concept_fusion, 0.08),
        (mutate_scientific_twist, 0.04),
        (mutate_perspective_shift, 0.04),
        # WOW-FACTOR mutations (28% combined weight)
        (mutate_banality_injection, 0.10),   # Mundane twists
        (mutate_emotional_contradiction, 0.09),  # Emotional dissonance
        (mutate_mundane_collision, 0.09),    # Mundane collisions
    ]
    
    # Weighted random selection
    r = random.random()
    cumulative = 0
    for mutation_fn, prob in mutations:
        cumulative += prob
        if r < cumulative:
            return mutation_fn(prompt)
    
    # Fallback
    return mutate_substitute_word(prompt)


def mutate_prompt_multiple(prompt: str, n_mutations: int = 1) -> str:
    """Apply multiple mutations in sequence."""
    for _ in range(n_mutations):
        prompt = mutate_prompt(prompt)
    return prompt


# =============================================================================
# SEED PROMPTS FOR INITIALIZATION
# =============================================================================

SEED_PROMPTS = [
    # Classic landscape paintings (familiar frame - WikiArt comparable)
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
    "a painting of mountains at dawn",
    "a winter landscape with snow",
    "a painting of a winding path",
]


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PROMPT GENERATION TEST")
    print("=" * 60)
    
    print("\n--- Random Prompts ---")
    for i in range(5):
        print(f"  {i+1}. {generate_random_prompt()}")
    
    print("\n--- Mutation Examples ---")
    base = "a painting of a landscape"
    print(f"  Base: {base}")
    for i in range(5):
        mutated = mutate_prompt(base)
        print(f"  Mutation {i+1}: {mutated}")
    
    print("\n--- Seed Prompts ---")
    for p in SEED_PROMPTS[:5]:
        print(f"  • {p}")