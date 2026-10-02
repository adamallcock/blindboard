"""Embedded word and trivia lists for the static finger-typing probe.
Uppercase A-Z only; trivia answers are unambiguous single words."""

WORDS = [
    "CAT", "DOG", "SUN", "MAP", "BOX", "KEY", "JAR", "FOX", "ZIP", "HAT",
    "WIND", "LAMP", "FISH", "GOLD", "TREE", "BLUE", "SNOW", "MILK", "ROAD", "BIRD",
    "STONE", "PLANT", "HOUSE", "WATCH", "BREAD", "CLOUD", "TRAIN", "SMILE", "GRAPE", "CHAIR",
    "MONKEY", "SILVER", "GARDEN", "PENCIL", "WINDOW", "ROCKET", "BASKET", "VIOLIN", "JUNGLE", "MARBLE",
    "CRANE", "PLUMB", "QUART", "FJORD", "GLYPH", "WALTZ", "ZEBRA", "QUICK", "VIXEN", "JUMBO",
    "STORM", "FLAME", "BRICK", "CHESS", "DRUMS", "EAGLE", "FROST", "GHOST", "HONEY", "IVORY",
    "KNIFE", "LEMON", "MOUSE", "NURSE", "OCEAN", "PIANO", "QUEEN", "RIVER", "SHEEP", "TIGER",
    "UNCLE", "VAPOR", "WHALE", "XENON", "YACHT", "ZESTY", "ACORN", "BADGE", "CIDER", "DAISY",
]

# (question, accepted answers) — first accepted answer is canonical.
TRIVIA = [
    ("What is the capital of France?", ["PARIS"]),
    ("What metal has the chemical symbol Fe?", ["IRON"]),
    ("What is the largest planet in our solar system?", ["JUPITER"]),
    ("What is the opposite of hot?", ["COLD"]),
    ("How many days are in a week? Answer with the word.", ["SEVEN"]),
    ("What color is a ripe banana?", ["YELLOW"]),
    ("What is the common name for H2O?", ["WATER"]),
    ("What is the currency of Japan?", ["YEN"]),
    ("What is the first month of the year?", ["JANUARY"]),
    ("Which animal is known as man's best friend?", ["DOG"]),
    ("What is the capital of Italy?", ["ROME"]),
    ("Which planet is known as the Red Planet?", ["MARS"]),
    ("Which animal is called the king of the jungle?", ["LION"]),
    ("What is frozen water called?", ["ICE"]),
    ("How many legs does a spider have? Answer with the word.", ["EIGHT"]),
    ("What is the capital of England?", ["LONDON"]),
    ("In which compass direction does the sun rise?", ["EAST"]),
    ("What is a baby dog called?", ["PUPPY", "PUP"]),
    ("What is the largest ocean on Earth?", ["PACIFIC"]),
    ("What is the fastest land animal?", ["CHEETAH"]),
    ("What is the national bird of the United States?", ["EAGLE"]),
    ("Which season comes after winter?", ["SPRING"]),
    ("What is the capital of Germany?", ["BERLIN"]),
    ("What do bees make?", ["HONEY"]),
]
