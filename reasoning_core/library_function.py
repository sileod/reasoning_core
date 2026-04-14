import inspect
import pkgutil
import importlib
import sqlite3
from typing import get_type_hints

DB_NAME = "functions.db"
visited_modules = set()

# -------------------------------
# LIST OF LIBRARIES TO SCAN
# -------------------------------
LIBRARIES = [
    "sklearn",
    "pandas",
    "scipy",
    "jax",
    "numpy",
    # "matplotlib",
    "seaborn",
    "statsmodels",
    "tensorflow",
    "torch",
    "sympy",
    "xgboost",
    # "lightgbm",
    "cv2",
    "PIL",
    "nltk",
    # "spacy",
    # "gensim",
    "joblib",
    # "h5py",
    "requests",
    "flask",
    # "fastapi",
    # "django",
    # "sqlalchemy",
    # "beautifulsoup4",
    # "lxml",
    "pyyaml",
    # "pytest",
    # "tqdm",
    # "plotly",
    # "dash",
    # "pydantic",
    "skimage",
    # "openpyxl",
    # "pymongo",
    # "psycopg2",
    "pytorch_lightning",
    # "cffi",
    # "cryptography",
    # "click",
    # "rich",
    # "pdfminer",
    # "torchvision",
    # "torchaudio",
    "transformers",
    # "accelerate",
    # "sentencepiece",
    # "huggingface_hub",
    "datasets", 
    "networkx",
    "shapelys"
]

# -------------------------------
# DB SETUP
# -------------------------------
def create_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS functions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        library TEXT,
        module TEXT,
        function_name TEXT,
        inputs TEXT,
        outputs TEXT
    )
    """)
    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_func
        ON functions(module, function_name, inputs, outputs)
    """)
    conn.commit()
    return conn


# -------------------------------
# TYPE EXTRACTION
# -------------------------------
def format_type(t):
    try:
        return t.__name__
    except AttributeError:
        return str(t)


def extract_types(func):
    try:
        hints = get_type_hints(func)
    except Exception:
        return None  # skip if hints break

    # skip functions with no signature (built-in/C functions)
    try:
        signature = inspect.signature(func)
    except (ValueError, TypeError):
        return None

    inputs = []
    for name, param in signature.parameters.items():
        if name not in hints:
            return None  # strict: skip if ANY param missing type
        inputs.append(f"{name}: {format_type(hints[name])}")

    if "return" not in hints:
        return None

    output = format_type(hints["return"])
    return ", ".join(inputs), output


# -------------------------------
# SCAN AND STORE
# -------------------------------

BAD_NAME_KEYWORDS = [
    "test", "compat", "deprecated", "deprecate"
    # removed: "util", "version", "validate", "stack", "copy", "dtype", "engine"
    # these were killing legitimate functions like cross_validate, clone, etc.
]

# ALLOWED_PREFIXES = [
#     # sklearn
#     "sklearn.",

#     # scipy
#     "scipy.signal",
#     "scipy.stats",
#     "scipy.fft",
#     "scipy.optimize",
#     "scipy.linalg",

#     # statsmodels
#     "statsmodels.tsa",
#     "statsmodels.stats",

#     # pandas
#     "pandas.core",

#     # torch
#     "torch.nn.functional",
#     "torch.optim",
#     "torch.utils.data",
#     "torch.compiler",

#     # transformers
#     "transformers.modeling_utils",
#     "transformers.tokenization_utils",

#     # jax
#     "jax.numpy",
#     "jax.scipy",

#     # sympy
#     "sympy.core",
#     "sympy.functions",

#     # nltk
#     "nltk.translate",
#     "nltk.metrics",
#     "regex",
#     "re",

#     # PIL
#     "PIL.Image",
#     "PIL.ImageFilter",
#     "PIL.ImageEnhance",

#     # xgboost
#     "xgboost.",

#     # requests
#     "requests.",

#     # networkx
#     "networkx.",

#     # shapely
#     "shapely.",

#     # skimage
#     "skimage.",

#     # joblib
#     "joblib.cpu_count",
#     "joblib.dump",
#     "joblib.hash",
#     "joblib.Memory",
#     "joblib.Parallel",

#     # numpy
#     "numpy.",
# ]

ALLOWED_PREFIXES = [
    # core typed ML / tensor ecosystem
    "jax",
    "jaxlib",
    "flax",
    "optax",
    "orbax",
    "equinox",
    "chex",
    "dm_tree",

    # HuggingFace ecosystem (partial but useful)
    "transformers",
    "datasets",
    "tokenizers",
    "accelerate",
    "huggingface_hub",
    "diffusers",
    "peft",
    "trl",
    "evaluate",
    "optimum",
    "sentencepiece",
    "safetensors",
    "bitsandbytes",

    # modern ML / data
    "sklearn",
    "xgboost",
    "lightgbm",
    "ray",
    "ray.tune",
    "ray.rllib",

    "lightning",
    "torchmetrics",
    "skorch",

    "catboost",
    "autogluon",
    "pycaret",

    # numeric / scientific (limited yield but still useful)
    "numpy",
    "scipy",
    "pandera",
    "great_expectations",
    "evidently",
    "dvc",
    "mlflow",
    "wandb",
    "clearml",

    # graph / geometry
    "networkx",
    "shapely",

    # image
    "skimage",
    "PIL",

    # NLP
    "nltk",
    "spacy",

    # web / infra (VERY typed-friendly)
    "fastapi",
    "pydantic",
    "requests",

    # utility / orchestration
    "joblib",
    "typing_extensions",
    "einops",
    "timm",
    "fairscale",
    "apex",
    "detectron2",

    # LLM app framework
    "langchain",
    "langchain_core",
    "langchain_community",
    "langchain_openai",

    "llama_index",
    "llama_cpp",
    "autogen",
    "guidance",
    "outlines",
    "vllm",
    "deepspeed",

    # Probabilistic / Bayesian ML
    "pyro",
    "pyro_ppl",
    "numpyro",
    "tensorflow_probability",
]




def is_bad_function(name, module_name):
    full = f"{module_name}.{name}".lower()
    return any(k in full for k in BAD_NAME_KEYWORDS)

def is_allowed(module_name):
    return any(
        module_name == p or module_name.startswith(p + ".")
        for p in ALLOWED_PREFIXES
    )

def scan_and_store(library_name, conn):
    cur = conn.cursor()
    inserted = 0

    try:
        package = importlib.import_module(library_name)
    except ModuleNotFoundError:
        print(f"Library {library_name} not installed, skipping...")
        return 0

    seen = set()  # global dedup inside this library

    for _, module_name, _ in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):

        # -------------------------------
        # HARD FILTERS (fast skips)
        # -------------------------------
        if any(x in module_name for x in [
            ".tests", ".test_", "conftest",
            "._", "internals", "compat",
            "testing", "util", "utils",
            "private", "_libs"
        ]):
            continue

        if not any(module_name.startswith(p) for p in ALLOWED_PREFIXES):
            continue
        # if not is_allowed(module_name):
        #     continue

        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        if module in visited_modules:
            continue
        visited_modules.add(module)

        try:
            functions = inspect.getmembers(module, inspect.isfunction)
        except Exception:
            continue

        for name, func in functions:

            # -------------------------------
            # NAME FILTERS
            # -------------------------------
            if name.startswith("_"):
                continue

            if is_bad_function(name, module_name):
                continue

            # kill generic junk functions
            if name in {
                "int_like", "pprint_thing", "is_url",
                "get_console_size", "handle_data_source",
                "array_namespace"
            }:
                continue

            # -------------------------------
            # TYPE EXTRACTION
            # -------------------------------
            result = extract_types(func)
            if result is None:
                continue

            inputs, outputs = result

            # skip empty / trivial signatures
            if not inputs.strip():
                continue

            # -------------------------------
            # DOC QUALITY FILTER
            # -------------------------------
            doc = inspect.getdoc(func)
            if not doc or len(doc.split()) < 20:
                continue

            # -------------------------------
            # DEDUPLICATION (strong)
            # -------------------------------
            key = (name, inputs, outputs)

            if key in seen:
                continue
            seen.add(key)

            # -------------------------------
            # INSERT
            # -------------------------------
            cur.execute("""
                INSERT OR IGNORE INTO functions 
                (library, module, function_name, inputs, outputs)
                VALUES (?, ?, ?, ?, ?)
            """, (library_name, module_name, name, inputs, outputs))

            inserted += 1

    conn.commit()
    return inserted


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    conn = create_db()
    total_inserted = 0

    for lib in LIBRARIES:
        print(f"Scanning {lib}...")
        count = scan_and_store(lib, conn)
        print(f"Inserted {count} typed functions from {lib}\n")
        total_inserted += count

    conn.close()
    print(f"Done. Total typed functions inserted: {total_inserted}")