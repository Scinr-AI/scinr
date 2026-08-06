# Custom Themes

Create and register custom extraction themes to extend scinr with domain-specific models. A theme is a collection of extraction models organized under a common domain description. Themes are auto-discovered by `ThemeRegistry` and used by the extraction LLM (Stage 1) to classify document sections.

---

## What is a Theme?

A **theme** is a Python package containing two required files:

| File | Purpose |
| :--- | :--- |
| `catalog.py` | Declares `THEME_DESCRIPTION` (for LLM classification) and `SELECTABLE_MODELS` (the models available for this theme). |
| `models.py` | Defines the Pydantic extraction model classes. |

Themes live in a directory tree. The `ThemeRegistry` scans directories for `catalog.py` files and builds a tree of themes. A folder is a theme if and only if it contains a `catalog.py`.

### How Themes Are Used

During Stage 1 (`"extraction"`), the extraction LLM reads every registered theme's `THEME_DESCRIPTION` to decide which thematic domain each structural node belongs to. Each node is classified independently — a single document may have nodes classified to multiple different themes. Once classified, Stage 3 (`"annotation"`) selects a model from that theme's `SELECTABLE_MODELS` to extract structured entities.

```
Stage 1: Document section ──► LLM reads THEME_DESCRIPTIONs ──► picks best theme for this node
Stage 3: Classified node ──► LLM reads SELECTABLE_MODELS ──► picks best model
Stage 4: Selected model ──► extracts structured entities
```

### Auto-Discovery

Themes are discovered automatically. There is no manual registration step. The `ThemeRegistry` scans:

1. The built-in `scinr.newton.models/` directory at import time.
2. Any additional directories passed via `extra_models_paths` in `configure()`.
3. Packages registered via the `scinr.newton.models` entry-point group.

---

## Theme Structure

A minimal theme has the following layout:

```
my_custom_theme/
├── __init__.py              # REQUIRED (may be empty)
├── catalog.py               # THEME_DESCRIPTION + SELECTABLE_MODELS
└── models.py                # Extraction model definitions
```

A theme with sub-themes:

```
my_custom_theme/
├── __init__.py              # REQUIRED (may be empty)
├── catalog.py               # Parent catalog (aggregates all sub-themes)
├── models.py                # Parent-level models
└── sub_theme/               # Optional sub-theme
    ├── __init__.py          # REQUIRED
    ├── catalog.py           # Sub-theme catalog
    └── models.py            # Sub-theme models
```

**Every directory that contains `.py` files must have an `__init__.py`.** This is not optional — without it, Python will not treat the directory as a package and relative imports will fail.

---

## Creating `catalog.py`

The `catalog.py` file is the heart of a theme. It declares two module-level variables:

| Variable | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `THEME_DESCRIPTION` | `str` | Yes | One to three sentences describing the document types this theme covers. Used by the annotation LLM to classify sections. |
| `SELECTABLE_MODELS` | `list[type]` | Yes | List of Pydantic model classes the annotation agent can select. Must be subclasses of `ExtractionModel` (or `BaseModel`). |

### Complete Example

```python
"""Catalog for the clinical trials theme."""
from __future__ import annotations

from .models import (
    TrialProtocolModel,
    TrialProtocolModelList,
    AdverseEventModel,
    AdverseEventModelList,
    DosageModel,
)

THEME_DESCRIPTION: str = (
    "Clinical trial protocol documents and adverse event reports. "
    "Covers trial phases (I-IV), inclusion/exclusion criteria, "
    "adverse event grading (CTCAE), dosage regimens, and endpoints. "
    "Use for sections describing trial design, patient populations, "
    "safety data, and dosing schedules. "
    "Distinct from regulatory variation guidelines and manufacturing records."
)

SELECTABLE_MODELS: list[type] = [
    TrialProtocolModelList,    # multi-trial sections (most specific first)
    TrialProtocolModel,        # single trial section
    AdverseEventModelList,     # multi-event sections
    AdverseEventModel,         # single event
    DosageModel,               # dosage regimen
]
```

### Writing an Effective `THEME_DESCRIPTION`

The extraction LLM (Stage 1) reads `THEME_DESCRIPTION` to decide whether a structural node belongs to this theme. A good description:

- **Names the document types** it covers (`"Clinical trial protocol documents"`, `"EMA Best Practice Guidelines"`)
- **Names regulatory standards** when applicable (`"CTCAE grading"`, `"ICH E6 GCP"`, `"EC Regulation 1234/2008"`)
- **Distinguishes** from adjacent themes that could be confused (`"Distinct from regulatory variation guidelines..."`)
- **Gives examples** of the entities it captures (`"trial phases (I-IV)"`, `"adverse event grading"`)

```python
# GOOD — specific, technical, distinguishable
THEME_DESCRIPTION: str = (
    "EU pharmaceutical variation guidelines (Official Journal, EC Regulation 1234/2008). "
    "Covers variation type classification (IA, IAIN, IB, II, A, BA), variation codes "
    "(e.g. Q.I.a.1, B.II.b.1), conditions, documentation requirements, and procedural rules. "
    "Distinct from BPG (best practice guidelines) and Q&A documents."
)

# BAD — too vague, LLM will misclassify
THEME_DESCRIPTION: str = "Pharmaceutical regulatory documents."

# BAD — only one sentence, no distinguishing information
THEME_DESCRIPTION: str = "Documents about clinical trials."
```

### `SELECTABLE_MODELS` Ordering

Order models from most to least specific. The LLM tends to prefer earlier entries when confidence between models is similar:

1. **List wrapper models** for multi-instance sections (most specific)
2. **Main models** for single-instance sections
3. **Supporting/complementary models**

```python
SELECTABLE_MODELS: list[type] = [
    VariationCodeWithDocsAndConditionModelList,   # 1. multi-code sections with inline data
    VariationCodeModel,                           # 2. single-code sections
    DocumentationModelList,                       # 3. sections listing >=2 documentation requirements
    DocumentationModel,                           # 4. single documentation requirement
    ConditionModelList,                           # 5. sections listing >=2 conditions
    ConditionModel,                               # 6. single condition
    ProcedureTypeModelList,                       # 7. sections defining >=2 procedure types
    ProcedureTypeModel,                           # 8. single procedure type
]
```

> **Note:** Both `XxxModel` and `XxxModelList` must appear in `SELECTABLE_MODELS` if you create a list wrapper. If only the list wrapper is present, the agent cannot select the single-instance model directly.

---

## Creating `models.py`

Model definition follows the patterns described in [Custom Models](custom-models.md). Extraction models are recommended to inherit from `ExtractionModel` (imported from `scinr.newton.models.base`), which enforces `extra="forbid"` to prevent silent LLM hallucinations. Any valid Pydantic `BaseModel` subclass will work, but `ExtractionModel` is the recommended base.

### Minimal Complete Example

```python
"""Clinical trial extraction models."""
from __future__ import annotations

from enum import Enum
from pydantic import Field
from scinr.newton.models.base import ExtractionModel


class TrialPhase(str, Enum):
    """Clinical trial phase."""

    PHASE_1 = "Phase I"
    PHASE_2 = "Phase II"
    PHASE_3 = "Phase III"
    PHASE_4 = "Phase IV"


class TrialProtocolModel(ExtractionModel):
    """A single clinical trial protocol entry."""

    trial_id: str = Field(
        ...,
        description=(
            "Unique trial identifier (e.g. 'NCT01234567', 'Study-2024-001'). "
            "Include the full prefix as written in the document."
        ),
        json_schema_extra={"entity_label": "TrialID", "instance_key": True},
    )
    phase: TrialPhase | None = Field(
        default=None,
        description=(
            "Clinical trial phase: Phase I through Phase IV. "
            "None if the phase is not stated in the document."
        ),
    )
    indication: str = Field(
        ...,
        description=(
            "Disease or condition being studied in this trial. "
            "Use the exact terminology from the document."
        ),
    )
    inclusion_criteria: list[str] = Field(
        default_factory=list,
        description=(
            "Patient inclusion criteria as bullet points or numbered items. "
            "Each element is one criterion. Empty list if not stated."
        ),
    )
    exclusion_criteria: list[str] = Field(
        default_factory=list,
        description=(
            "Patient exclusion criteria as bullet points or numbered items. "
            "Each element is one criterion. Empty list if not stated."
        ),
    )


class TrialProtocolModelList(ExtractionModel):
    """Use when the section defines TWO OR MORE trial protocol entries."""

    items: list[TrialProtocolModel] = Field(
        default_factory=list,
        description=(
            "List of trial protocol entries. Each element represents one "
            "distinct trial. Use TrialProtocolModel when there is only one trial. "
            "Use this model when the section is a table or list with two or more trials."
        ),
    )
```

### Key Rules

| Rule | Why |
| :--- | :--- |
| Inherit from `ExtractionModel` | Enforces `extra="forbid"` to catch LLM hallucinations |
| Every field has `description=` with >= 15 words | The LLM uses the description as its primary extraction signal |
| List fields use `default_factory=list` | Never use `default=[]` — mutable default shared across instances |
| Optional scalars use `default=None` | Explicit opt-in for None values |
| Enums use `str` as base class | Guarantees JSON-serializable values and correct Neo4j storage |
| Relative imports for internal modules | `from .models import ...` — never bare `from models import ...` |

For detailed model creation patterns (enums, validators, `entity_label`, `instance_relationships`, list wrappers, etc.), see [Custom Models](custom-models.md) and the [Model Creation Guide](https://github.com/scinr-ai/scinr/blob/main/src/scinr/newton/model-creation/AGENTS.md).

---

## Registering Themes

There are four ways to make a custom theme available to scinr.

### Method 1: `extra_models_paths` in `configure()`

Pass filesystem paths to `configure()`. The `ThemeRegistry` scans each path recursively for `catalog.py` files.

```python
from scinr.newton import configure

configure(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
    extra_models_paths=["/path/to/my_custom_theme"],
)
```

Multiple paths:

```python
configure(
    extra_models_paths=[
        "/home/user/projects/clinical_trials",
        "/opt/scinr/models/device_safety",
        "./local_models",  # relative paths resolve from the working directory
    ],
)
```

**How it works:**

1. `configure()` stores the paths in `ScinrConfig.extra_models_paths`.
2. On first access, `get_theme_registry()` creates a `ThemeRegistry` with these paths.
3. `ThemeRegistry._scan_extra_roots()` walks each directory recursively.
4. For each folder containing `catalog.py`, the catalog is imported and registered.

> **Important:** The path you pass must be the **parent directory** of your theme folders. If your theme is at `/path/to/my_custom_theme/catalog.py`, pass `/path/to/` — not `/path/to/my_custom_theme`. The registry scans the passed directory and discovers `my_custom_theme` as a child folder.
>
> However, if you pass `/path/to/my_custom_theme` directly and it contains `catalog.py`, the registry will also find it as the root-level theme named `my_custom_theme`.

### Method 2: Environment Variable

Set `SCINR_EXTRA_MODELS_PATHS` as a colon-separated list of paths:

```bash
# .env file
SCINR_EXTRA_MODELS_PATHS=/path/to/theme1:/path/to/theme2
```

```python
from scinr.newton import configure

# No extra_models_paths arg needed — read from environment
configure(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
)
```

The environment variable is read during `configure()` and converted to a list of `Path` objects. Explicit `extra_models_paths` in `configure()` takes precedence and replaces the environment variable entirely.

### Method 3: `enabled_user_themes`

Use `enabled_user_themes` to whitelist specific user themes. This is useful when you have many themes in `extra_models_paths` but only want to activate a subset.

```python
from scinr.newton import configure

configure(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
    extra_models_paths=["/path/to/my_models"],
    enabled_user_themes=["clinical_trials", "device_safety"],
)
```

With `enabled_user_themes`, only the listed themes from user paths are active. All other user themes discovered in `extra_models_paths` are silently excluded. Built-in themes are unaffected (use `enabled_base_themes` for those).

**Combining with `enabled_base_themes`:**

```python
configure(
    # Only these built-in themes are active
    enabled_base_themes=["default", "pharmaceutical_quality"],
    # Only these user themes are active
    enabled_user_themes=["clinical_trials"],
    extra_models_paths=["/path/to/my_models"],
)
```

> **Validation:** If a theme name in `enabled_user_themes` does not exist among the discovered user themes, `configure()` raises a `ConfigurationError`. Similarly, an empty list raises an error — pass `None` to activate all themes.

### Method 4: Entry Points (Packaged Themes)

Distribute a theme as a Python package with an entry point. This is the recommended approach for sharing themes across projects or publishing them to PyPI.

```toml
# pyproject.toml of your theme package
[project]
name = "scinr-clinical-trials"
version = "0.1.0"

[project.entry-points."scinr.newton.models"]
clinical_trials = "scinr_clinical_trials.catalog"
```

The entry point value (`"scinr_clinical_trials.catalog"`) is the dotted import path to your `catalog` module. The `ThemeRegistry` discovers entry points from the `scinr.newton.models` group and imports them automatically.

**Directory layout for a packaged theme:**

```
scinr-clinical-trials/
├── pyproject.toml
├── src/
│   └── scinr_clinical_trials/
│       ├── __init__.py
│       ├── catalog.py
│       ├── models.py
│       └── adverse_events/
│           ├── __init__.py
│           ├── catalog.py
│           └── models.py
└── README.md
```

Install the package and the theme is automatically available — no `extra_models_paths` needed:

```bash
pip install scinr-clinical-trials
```

```python
from scinr.newton import configure

# No extra_models_paths — entry point discovery handles it
configure(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
)
```

> **Precedence:** Built-in themes take precedence over external package themes with the same name. If a conflict is detected, a warning is logged and the built-in theme is used. Rename your external theme to avoid conflicts.

---

## Verifying Theme Registration

After setting up your theme, verify it was discovered correctly:

```python
from scinr.newton.utils.theme_registry import get_theme_registry

registry = get_theme_registry()

# List all registered theme paths
all_paths = registry.get_all_theme_paths()
print("Registered themes:")
for path in all_paths:
    theme = registry._themes[path]
    print(f"  {path}: {len(theme.models)} models")

# Inspect a specific theme
theme = registry.find_best_theme("clinical_trials")
print(f"\nTheme: {theme.path}")
print(f"Description: {theme.description}")
print(f"Models: {[m.__name__ for m in theme.models]}")

# See the catalog block the LLM will receive
catalog_block = registry.build_catalog_block(theme)
print(f"\nLLM catalog block:\n{catalog_block}")
```

### The Theme Classification Prompt

You can inspect what the annotation LLM sees for theme classification:

```python
from scinr.newton.utils.theme_registry import get_theme_registry

registry = get_theme_registry()

# The full theme list injected into the classification prompt
print(registry.get_theme_list_for_prompt())
```

Example output:

```
- clinical_trials: Clinical trial protocol documents and adverse event reports...
- default: Generic fallback for content that does not fit a specific thematic domain
- pharmaceutical_quality: Pharmaceutical drug development documents following ICH CTD Module 3...
```

---

## Sub-Themes

A **sub-theme** is a nested folder with its own `catalog.py`, representing a specialized subset of a parent theme. Use sub-themes when the domain has clearly differentiated document types with incompatible model sets.

### Structure

```
my_theme/
├── __init__.py
├── catalog.py               # Parent catalog (aggregates all sub-themes)
├── phase1_trials/
│   ├── __init__.py
│   ├── catalog.py           # Sub-theme catalog
│   └── models.py
└── adverse_events/
    ├── __init__.py
    ├── catalog.py
    └── models.py
```

### Parent Catalog

The parent `catalog.py` aggregates models from sub-themes:

```python
"""Catalog for the clinical trials parent theme."""
from __future__ import annotations

from .phase1_trials.models import Phase1TrialModel, Phase1TrialModelList
from .adverse_events.models import AdverseEventModel, AdverseEventModelList

THEME_DESCRIPTION: str = (
    "Clinical trial documents across all phases and safety reporting. "
    "Covers Phase I first-in-human trials, adverse event reports (CTCAE grading), "
    "and related safety documentation. "
    "Distinct from regulatory variation guidelines and manufacturing records."
)

SELECTABLE_MODELS: list[type] = [
    Phase1TrialModelList,
    Phase1TrialModel,
    AdverseEventModelList,
    AdverseEventModel,
]
```

### Sub-Theme Catalog

Each sub-theme has its own `catalog.py` with a narrower `THEME_DESCRIPTION`:

```python
# phase1_trials/catalog.py
"""Catalog for Phase I clinical trials."""
from __future__ import annotations

from .models import Phase1TrialModel, Phase1TrialModelList

THEME_DESCRIPTION: str = (
    "Phase I first-in-human clinical trial protocols and reports. "
    "Covers single-ascending dose (SAD), multiple-ascending dose (MAD), "
    "food effect, and drug-drug interaction studies. "
    "Focuses on safety, tolerability, and pharmacokinetics. "
    "Distinct from Phase II-IV trials and adverse event narratives."
)

SELECTABLE_MODELS: list[type] = [
    Phase1TrialModelList,
    Phase1TrialModel,
]
```

```python
# adverse_events/catalog.py
"""Catalog for adverse event reporting."""
from __future__ import annotations

from .models import AdverseEventModel, AdverseEventModelList

THEME_DESCRIPTION: str = (
    "Adverse event reports and safety narratives from clinical trials. "
    "Covers CTCAE grading (Grade 1-5), Serious Adverse Events (SAE), "
    "adverse drug reactions (ADR), and causality assessments. "
    "Distinct from trial protocol design and pharmacokinetic data."
)

SELECTABLE_MODELS: list[type] = [
    AdverseEventModelList,
    AdverseEventModel,
]
```

### When to Use Sub-Themes

| Use sub-themes when... | Keep in same theme when... |
| :--- | :--- |
| Document types are clearly differentiated | Models are complementary and used together |
| Model sets are incompatible (different field structures) | Sections often mix entity types from both domains |
| Each sub-domain has its own regulatory standards | The distinction is artificial or marginal |
| You want finer-grained theme classification | A single `THEME_DESCRIPTION` covers the domain well |

### How Sub-Themes Are Discovered

The `ThemeRegistry` registers both the parent and each sub-theme independently:

```
my_theme/                    → registered as "my_theme"
├── catalog.py
├── phase1_trials/           → registered as "my_theme/phase1_trials"
│   └── catalog.py
└── adverse_events/          → registered as "my_theme/adverse_events"
    └── catalog.py
```

During classification, the LLM receives all three theme descriptions. If a section matches the narrow sub-theme description, it gets classified to the sub-theme path. If it matches only the parent description, it gets the parent path. The `find_best_theme()` method resolves from most specific to least specific:

```python
# If the LLM classifies a section as "my_theme/phase1_trials":
registry.find_best_theme("my_theme/phase1_trials")  # exact match → sub-theme

# If the LLM classifies a section as "my_theme":
registry.find_best_theme("my_theme")  # exact match → parent theme
```

---

## Theme Discovery Flow

Understanding the complete flow helps with debugging:

```
1. configure(extra_models_paths=["/path/to/models"])
   │
2. ThemeRegistry.__init__()
   │
   ├── Scans built-in models/ for catalog.py files
   │   └── Imports via importlib: "scinr.newton.models.<path>.catalog"
   │
   ├── Applies enabled_base_themes filter (if set)
   │
   ├── Discovers entry-point packages (scinr.newton.models group)
   │
   ├── Scans extra_models_paths for catalog.py files
   │   └── Imports via importlib.util (works for any filesystem path)
   │       ├── Package layout: uses __init__.py chain + importlib.import_module
   │       └── Standalone layout: uses spec_from_file_location + sys.path manipulation
   │
   └── Applies enabled_user_themes filter (if set)
   │
3. Stage 1 (extraction) begins
    │
    ├── get_theme_list_for_prompt() → theme descriptions for classification
    │
4. For each structural node (Stage 1):
    │
    ├── LLM reads theme descriptions → picks best theme for this node
    ├── find_best_theme(detected_path) → resolves to ThemeNode
    │
5. Stage 3 (annotation) begins
    │
    ├── build_catalog_block(theme) → model catalog for annotation LLM
    ├── LLM selects model from SELECTABLE_MODELS
    │
6. Stage 4 (extraction)
    │
    └── Selected model → extracts structured entities
```

### Package Layout vs. Standalone Layout

When loading user themes from `extra_models_paths`, the registry supports two layouts:

**Package layout** (has `__init__.py` files):
```
my_models/
├── __init__.py
└── clinical_trials/
    ├── __init__.py
    ├── catalog.py          ← imported as "my_models.clinical_trials.catalog"
    └── models.py
```
Relative imports (`from .models import ...`) work correctly because Python's normal package machinery handles them. Pass `/path/to/` (parent of `my_models`) as `extra_models_paths`.

**Standalone layout** (no `__init__.py`):
```
my_models/
└── clinical_trials/
    ├── catalog.py          ← loaded via spec_from_file_location
    └── models.py
```
The catalog's directory is temporarily added to `sys.path` so bare sibling imports (`from models import ...`) resolve. Pass `/path/to/my_models` as `extra_models_paths`.

> **Recommendation:** Always use the package layout with `__init__.py` files. It is more robust and avoids `sys.path` manipulation.

---

## Troubleshooting

| Problem | Cause | Fix |
| :--- | :--- | :--- |
| Theme not discovered | Missing `__init__.py` in a directory | Add empty `__init__.py` to every directory containing `.py` files |
| Theme not discovered | Path in `extra_models_paths` is wrong | Pass the parent directory of the theme folder, not the theme folder itself (unless the theme folder has `catalog.py` directly) |
| Import errors on startup | Wrong relative import in `catalog.py` | Use dots: `from .models import ...` not `from models import ...` |
| Import errors on startup | Wrong relative import in `models.py` | Count dots from the file's directory: `from ..base import ...` for parent, `from ...base import ...` for grandparent |
| Model not selectable by annotation agent | Model not in `SELECTABLE_MODELS` | Add the model class to `SELECTABLE_MODELS` in `catalog.py` |
| Wrong model applied to sections | Vague `THEME_DESCRIPTION` | Be specific: name document types, regulatory standards, distinguish from adjacent themes |
| Wrong model applied to sections | Models in wrong order in `SELECTABLE_MODELS` | Order from most specific (list wrappers) to least specific (single-instance) |
| `ConfigurationError: not a subclass of pydantic.BaseModel` | Model in `SELECTABLE_MODELS` doesn't inherit from `BaseModel` | Ensure all models inherit from `ExtractionModel` or any `BaseModel` subclass |
| Built-in theme silently overridden | User theme has same name as built-in theme | Rename your theme folder or adjust `extra_models_paths` |
| Entry-point theme not loading | Package not installed or entry-point misconfigured | Verify `pip show <package>` and check `[project.entry-points."scinr.newton.models"]` in `pyproject.toml` |

### Debug Logging

Enable debug logging to see the theme discovery process:

```python
import logging
from scinr.newton import configure

logging.basicConfig(level=logging.DEBUG)

configure(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
    extra_models_paths=["/path/to/my_models"],
)
```

Look for lines like:

```
DEBUG:scinr.newton.utils.theme_registry:ThemeRegistry: registered theme 'clinical_trials' with 5 models
DEBUG:scinr.newton.utils.theme_registry:ThemeRegistry: discovered 8 themes: ['clinical_trials', 'default', ...]
```

### Common Import Errors

```python
# catalog.py — CORRECT
from .models import TrialProtocolModel

# catalog.py — WRONG (bare import, depends on sys.path)
from models import TrialProtocolModel

# models.py — CORRECT (one level up)
from ..baseModels import NormalizedBaseModel

# models.py — CORRECT (installed package, absolute is fine)
from scinr.newton.models.base import ExtractionModel
```

### Verifying a Specific Theme

```python
from scinr.newton.utils.theme_registry import get_theme_registry

registry = get_theme_registry()

# Check if a theme exists
if "clinical_trials" in registry._themes:
    theme = registry._themes["clinical_trials"]
    print(f"Found: {theme.path}")
    print(f"  Description: {theme.description}")
    print(f"  Models: {[m.__name__ for m in theme.models]}")
    print(f"  Children: {list(theme.children.keys())}")
else:
    print("Theme 'clinical_trials' not found.")
    print("Available themes:", list(registry._themes.keys()))
```

---

## Complete Example: End-to-End Theme

Here is a complete custom theme from scratch, ready to use.

### Directory Layout

```
own_models/
├── __init__.py
└── clinical_trials/
    ├── __init__.py
    ├── catalog.py
    └── models.py
```

### `own_models/__init__.py`

Empty file. Required to make `own_models` a Python package.

### `own_models/clinical_trials/__init__.py`

Empty file. Required to make `clinical_trials` a Python package.

### `own_models/clinical_trials/models.py`

```python
"""Clinical trial extraction models."""
from __future__ import annotations

from enum import Enum
from pydantic import Field
from scinr.newton.models.base import ExtractionModel


class TrialPhase(str, Enum):
    """Clinical trial phase."""

    PHASE_1 = "Phase I"
    PHASE_2 = "Phase II"
    PHASE_3 = "Phase III"
    PHASE_4 = "Phase IV"


class AdverseEventGrade(str, Enum):
    """CTCAE adverse event severity grade."""

    GRADE_1 = "Grade 1"
    GRADE_2 = "Grade 2"
    GRADE_3 = "Grade 3"
    GRADE_4 = "Grade 4"
    GRADE_5 = "Grade 5"


class TrialProtocolModel(ExtractionModel):
    """A single clinical trial protocol entry."""

    trial_id: str = Field(
        ...,
        description=(
            "Unique trial identifier (e.g. 'NCT01234567', 'Study-2024-001'). "
            "Include the full prefix as written in the document."
        ),
        json_schema_extra={"entity_label": "TrialID", "instance_key": True},
    )
    phase: TrialPhase | None = Field(
        default=None,
        description=(
            "Clinical trial phase: Phase I through Phase IV. "
            "None if the phase is not stated in the document."
        ),
    )
    indication: str = Field(
        ...,
        description=(
            "Disease or condition being studied in this trial. "
            "Use the exact terminology as written in the document."
        ),
    )
    primary_endpoint: str | None = Field(
        default=None,
        description=(
            "Primary efficacy endpoint of the trial. "
            "None if not explicitly stated in the section."
        ),
    )


class TrialProtocolModelList(ExtractionModel):
    """Use when the section defines TWO OR MORE trial protocol entries."""

    items: list[TrialProtocolModel] = Field(
        default_factory=list,
        description=(
            "List of trial protocol entries. Each element represents one "
            "distinct trial. Use TrialProtocolModel for a single trial. "
            "Use this model when the section is a table or list with two or more trials."
        ),
    )


class AdverseEventModel(ExtractionModel):
    """A single adverse event entry from a clinical trial report."""

    event_term: str = Field(
        ...,
        description=(
            "Preferred term for the adverse event as written in the document "
            "(e.g. 'headache', 'nausea', 'anaphylactic reaction'). "
            "Use the exact terminology from the source."
        ),
        json_schema_extra={"entity_label": "AdverseEventTerm"},
    )
    grade: AdverseEventGrade | None = Field(
        default=None,
        description=(
            "CTCAE severity grade: Grade 1 (mild) through Grade 5 (death). "
            "None if the grade is not stated."
        ),
    )
    causality: str | None = Field(
        default=None,
        description=(
            "Assessed causality relationship to the study drug "
            "(e.g. 'related', 'possibly related', 'unrelated'). "
            "None if causality is not assessed."
        ),
    )
    outcome: str | None = Field(
        default=None,
        description=(
            "Outcome of the adverse event (e.g. 'resolved', 'not resolved', "
            "'resolved with sequelae', 'fatal'). None if not stated."
        ),
    )


class AdverseEventModelList(ExtractionModel):
    """Use when the section defines TWO OR MORE adverse event entries."""

    items: list[AdverseEventModel] = Field(
        default_factory=list,
        description=(
            "List of adverse event entries. Each element represents one "
            "distinct event. Use AdverseEventModel for a single event. "
            "Use this model when the section is a table or list with two or more events."
        ),
    )
```

### `own_models/clinical_trials/catalog.py`

```python
"""Catalog for the clinical trials theme."""
from __future__ import annotations

from .models import (
    TrialProtocolModel,
    TrialProtocolModelList,
    AdverseEventModel,
    AdverseEventModelList,
)

THEME_DESCRIPTION: str = (
    "Clinical trial protocol documents and adverse event reports. "
    "Covers trial phases (I-IV), inclusion/exclusion criteria, "
    "adverse event grading (CTCAE), dosage regimens, and endpoints. "
    "Use for sections describing trial design, patient populations, "
    "safety data, and dosing schedules. "
    "Distinct from regulatory variation guidelines and manufacturing records."
)

SELECTABLE_MODELS: list[type] = [
    TrialProtocolModelList,    # multi-trial sections (most specific first)
    TrialProtocolModel,        # single trial section
    AdverseEventModelList,     # multi-event sections
    AdverseEventModel,         # single event
]
```

### Usage

```python
import asyncio
from pathlib import Path
from scinr.newton import configure, run_pipeline

async def main():
    configure(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
        extra_models_paths=[str(Path(__file__).parent / "own_models")],
    )

    # Verify theme registration
    from scinr.newton.utils.theme_registry import get_theme_registry
    registry = get_theme_registry()
    print("Themes:", registry.get_all_theme_paths())

    # Run pipeline with custom theme
    result = await run_pipeline(input_raw="./raw_docs")
    print(f"Pipeline result: {result.success}")

asyncio.run(main())
```

---

## See Also

- **[Custom Models](custom-models.md)** — Defining domain-specific Pydantic extraction models.
- **[Configuration](../configuration.md)** — Complete reference for `configure()`, including `extra_models_paths`, `enabled_base_themes`, and `enabled_user_themes`.
- **[Running the Pipeline](running-pipeline.md)** — Orchestrating the full ingestion pipeline with custom themes.
- **[Architecture](../architecture.md)** — Detailed walkthrough of Stage 1 extraction (theme classification) and Stage 3 annotation (model selection).
- **[Model Creation Guide](https://github.com/scinr-ai/scinr/blob/main/src/scinr/newton/model-creation/AGENTS.md)** — Comprehensive guide for AI agents creating extraction models.
