from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import backoff
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

# --- env loader ---
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# --------------------- Globals / Debug ---------------------
DEBUG = "True" #os.getenv("DEBUG", "True").lower() == "true"
INFER_TEXT_LIMIT_CHARS = int(os.getenv("INFER_TEXT_LIMIT_CHARS", "4000"))
INFER_TEMPERATURE = float(os.getenv("INFER_TEMPERATURE", "0.1"))


class EntityField(BaseModel):
    """Represents a field in an entity.

    Attributes:
        name (str): The name of the field.
        type (str): The type of the field, which can be one of 'string', 'number', 'boolean', 'date', or 'array<...>'.
        description (Optional[str]): An optional description of the field.
        required (bool): Whether the field is required or not.
    """

    name: str
    type: str  # 'string' | 'number' | 'boolean' | 'date' | 'array<...>'
    description: Optional[str] = None
    required: bool = True


def _log(*args: object) -> None:
    """
    Log a message to the console.

    Parameters:
        *args (object): The arguments to log.

    Returns:
        None
    """
    if DEBUG:
        print("[extractor]", *args, flush=True)


# --------------------- Models ---------------------
class EntityField(BaseModel):
    """Represents a field in an entity.

    Attributes:
        name (str): The name of the field.
        type (str): The type of the field, which can be one of 'string', 'number', 'boolean', 'date', or 'array<...>'.
        description (Optional[str]): An optional description of the field.
        required (bool): Whether the field is required or not.
    """

    name: str
    type: str  # 'string' | 'number' | 'boolean' | 'date' | 'array<...>'
    description: Optional[str] = None
    required: bool = True


class ExtractionSchema(BaseModel):
    """
    This class represents the schema for extracting data from a database.

    Attributes:
        fields (List[EntityField]): A list of EntityFields that define the data to be extracted.

    Returns:
        List[dict]: A list of dictionaries, where each dictionary represents a row of data extracted from the database.
    """

    fields: List[EntityField]


class ExtractRequest(BaseModel):
    """Represents a request to extract information from text.

    Attributes:
        model_config (ConfigDict): The configuration parameters for the extraction model.
        text (str): The input text to be extracted.
        extraction_schema (ExtractionSchema): The schema defining the structure of the extracted information.
        language (Optional[str]): The language of the input text. Defaults to None.
        instructions (Optional[str]): Any additional instructions for the extraction process. Defaults to None.
        temperature (Optional[float]): The temperature parameter for the extraction model. Defaults to 0.0.
        return_confidence (bool): Whether to include confidence scores in the extracted information. Defaults to True.
    """

    model_config = ConfigDict(populate_by_name=True)
    text: str
    extraction_schema: ExtractionSchema = Field(alias="schema")
    language: Optional[str] = None
    instructions: Optional[str] = None
    temperature: Optional[float] = 0.0
    return_confidence: bool = True


class ExtractResponse(BaseModel):
    """Represents a response from an entity extraction request.

    Attributes:
        request_id (str): The unique ID of the request that generated this response.
        model (str): The name of the entity extraction model used to generate this response.
        provider (str): The name of the entity extraction provider used to generate this response.
        latency_ms (int): The time it took for the entity extraction request to be processed, in milliseconds.
        entities (Dict[str, Any]): A dictionary containing the extracted entities and their corresponding values.
        confidence (Optional[float]): An optional confidence score between 0 and 1 indicating the accuracy of the extracted entities.
        warnings (List[str]): A list of any warnings or errors that occurred during the entity extraction process.
        raw (Optional[Dict[str, Any]]): An optional dictionary containing additional information about the response from the provider.
    """

    request_id: str
    model: str
    provider: str
    latency_ms: int
    entities: Dict[str, Any]
    confidence: Optional[float] = None
    warnings: List[str] = []
    raw: Optional[Dict[str, Any]] = None


class InferRequest(BaseModel):
    """Represents a request to perform inference on a text input.

    Attributes:
        text (str): The text input to be inferred.
        max_fields (int, optional): The maximum number of fields to include in the inference output. Defaults to 15.
        min_fields (int, optional): The minimum number of fields to include in the inference output. Defaults to 8.
        prefer_common_fields (bool, optional): Whether to prefer common fields over rare ones. Defaults to True.

    Returns:
        dict: A dictionary containing the inferred fields and their corresponding probabilities.
    """

    text: str
    max_fields: int = 15
    min_fields: int = 8
    prefer_common_fields: bool = True


class InferResponse(BaseModel):
    """
    Represents the response from an infer request.

    Attributes:
        fields (List[EntityField]): A list of entity fields returned by the infer request.
    """

    fields: List[EntityField]


class AutoExtractRequest(BaseModel):
    """Represents a request to extract structured data from unstructured text.

    Attributes:
        text (str): The unstructured text to extract data from.
        max_fields (int, optional): The maximum number of fields to extract. Defaults to 15.
        min_fields (int, optional): The minimum number of fields to extract. Defaults to 8.
        temperature (float, optional): The temperature to use for the extraction process. Defaults to 0.0.

    Returns:
        List[Dict[str, str]]: A list of dictionaries representing the extracted data. Each dictionary contains field names as keys and field values as values.
    """

    text: str
    max_fields: int = 15
    min_fields: int = 8
    temperature: Optional[float] = 0.0


class AutoExtractResponse(BaseModel):
    """
    Response object returned by the auto-extract API.

    Attributes:
        model_config (ConfigDict): Configuration parameters for the extraction model.
        inferred_schema (ExtractionSchema): Schema inferred from the input data.
        extraction (ExtractResponse): Extracted data and metadata.
    """

    model_config = ConfigDict(populate_by_name=True)
    inferred_schema: ExtractionSchema = Field(alias="schema")
    extraction: ExtractResponse


class PickRequest(BaseModel):
    """
    Represents a request to pick items from a list based on a given schema.

    Attributes:
        model_config (ConfigDict): Configuration parameters for the picking model.
        text (str): The input text that will be used to generate the picking candidates.
        input_schema (ExtractionSchema): The schema that defines the structure of the items to be picked.
        candidates (Dict): A dictionary of candidate items, where each key is a category and each value is a list of items in that category.
        temperature (int): The temperature parameter for the picking model.
    """

    model_config = ConfigDict(populate_by_name=True)
    text: str
    input_schema: ExtractionSchema = Field(alias="schema")
    candidates: Dict[str, List[str]]
    temperature: Optional[float] = 0.0


class PickResponse(BaseModel):
    """Represents a response from the pick API.

    Attributes:
        request_id (str): The ID of the request that was made.
        model (str): The name of the model used to generate the response.
        provider (str): The name of the provider that generated the response.
        latency_ms (int): The amount of time it took for the API to respond, in milliseconds.
        entities (Dict[str, Any]): A dictionary containing information about the entities returned by the API.
        selection (Dict[str, Optional[int]]): A dictionary containing the selected entities and their corresponding indices.
        warnings (List[str]): A list of warnings generated by the API.
    """

    request_id: str
    model: str
    provider: str
    latency_ms: int
    entities: Dict[str, Any]
    selection: Dict[str, Optional[int]]
    warnings: List[str] = []


# --------------------- JSON helpers ---------------------
try:
    import orjson  # type: ignore

    def _json_dumps(v: Any) -> str:
        return orjson.dumps(
            v, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS
        ).decode()

except Exception:

    def _json_dumps(v: Any) -> str:
        return json.dumps(v, indent=2)


JSON_BLOCK_RE = re.compile(r"```(?:json)?\n(.*?)\n```", re.DOTALL)
JSON_TAG_RE = re.compile(r"<json>\s*(\{[\s\S]*?\})\s*</json>")

# --------------------- Utils ---------------------
PRIMITIVES = {"string", "number", "boolean", "date"}

def _shorten_for_infer(text: str) -> str:
    t = text.strip()
    return t if len(t) <= INFER_TEXT_LIMIT_CHARS else t[:INFER_TEXT_LIMIT_CHARS]

def _schema_hint(schema: ExtractionSchema) -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    required = []
    for f in schema.fields:
        t = f.type.lower()
        if t.startswith("array<") and t.endswith(">"):
            inner = t[6:-1]
            if inner not in PRIMITIVES:
                inner = "string"
            p = {
                "type": "array",
                "items": {
                    "type": "string"
                    if inner == "date"
                    else (
                        "number"
                        if inner == "number"
                        else ("boolean" if inner == "boolean" else "string")
                    )
                },
                "description": f.description or "",
            }
        else:
            p = {
                "type": "string"
                if t in {"string", "date"}
                else (
                    "number"
                    if t == "number"
                    else ("boolean" if t == "boolean" else "string")
                ),
                "description": f.description or "",
            }
        props[f.name] = p
        if f.required:
            required.append(f.name)
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


def _build_prompt(
    text: str, schema: ExtractionSchema, instructions: Optional[str]
) -> str:
    rules = [
        "Return ONLY valid JSON (no markdown, no prose).",
        "Wrap the output JSON between <json> and </json> tags; output nothing else.",
        "If value is missing/uncertain, use null (or [] for arrays). Do NOT invent values.",
        "For each field, output the SHORTEST exact substring from TEXT that matches the definition.",
        "Do NOT copy whole sentences or unrelated fragments into fields.",
        "Normalize dates to YYYY-MM-DD; numbers to plain numeric (no symbols).",
        "Do not reuse the same substring for different fields unless identical by definition.",
    ]
    fields_desc = "\n".join(
        f"- {f.name} ({f.type}): {f.description or ''}" for f in schema.fields
    )
    return (
        "You are an accurate information extraction engine.\n"
        f"Target fields:\n{fields_desc}\n\n"
        "Rules:\n- "
        + "\n- ".join(rules)
        + "\n"
        + (f"Extra instructions: {instructions}\n" if instructions else "")
        + "Now extract entities from the TEXT and output strict JSON that validates the provided schema.\n\n"
        f"TEXT:\n{text}\n\n<json>\n"
    )


def _build_pick_prompt(
    text: str, schema: ExtractionSchema, candidates: Dict[str, List[str]]
) -> str:
    fields_desc = "\n".join(
        f"- {f.name} ({f.type}): {f.description or ''}" for f in schema.fields
    )
    return (
        "Candidate selection mode. Choose the best index per field from the candidate list, or null if absent.\n"
        'Return ONLY JSON of the form {"selection": {field_name: index|null}}.\n'
        "Wrap the output JSON between <json> and </json> tags; output nothing else.\n"
        "Guidelines: prefer exact matches in TEXT; break ties by field semantics (date looks like a date, etc.).\n\n"
        f"Fields:\n{fields_desc}\n\n"
        f"CANDIDATES JSON:\n{_json_dumps(candidates)}\n\n"
        f'TEXT:\n{text}\n\n<json>\n{{\n  "selection": {{}}\n}}\n</json>\n'
    )


# --------------------- LLM (watsonx) ---------------------
class WatsonxClient:
    def __init__(self, model: Optional[str] = None):
        self.model = model or os.getenv("MODEL", "meta-llama/llama-3-3-70b-instruct")
        self.base = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        self.version = os.getenv("WATSONX_VERSION", "2025-02-11")
        self.api_key = os.getenv("WATSONX_API_KEY")
        self.project = os.getenv("WATSONX_PROJECT_ID")
        if not self.api_key or not self.project:
            raise RuntimeError("WATSONX_API_KEY and WATSONX_PROJECT_ID must be set")

    @backoff.on_exception(backoff.expo, httpx.HTTPError, max_tries=3)
    async def _token(self) -> str:
        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post(
                "https://iam.cloud.ibm.com/identity/token",
                data={
                    "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                    "apikey": self.api_key,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            r.raise_for_status()
            return r.json()["access_token"]

    async def complete(
        self,
        prompt: str,
        schema_hint: Optional[Dict[str, Any]] = None,  # <- accept but ignore
        temperature: float = 0.0,
    ) -> str:
        token = await self._token()
        payload = {
            "model_id": self.model,
            "project_id": self.project,
            "input": prompt,
            "parameters": {
                "decoding_method": "greedy",
                "temperature": temperature or 0.0,
                "max_new_tokens": 900,
            },
        }
        endpoint = f"{self.base.rstrip('/')}/ml/v1/text/generation?version={self.version}"
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
            return data.get("results", [{}])[0].get("generated_text", "")


def get_client() -> WatsonxClient:
    return WatsonxClient()


# --------------------- Core helpers ---------------------


def _repair_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = JSON_TAG_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m = JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    try:
        i, j = text.index("{"), text.rindex("}")
        return json.loads(text[i : j + 1])
    except Exception:
        pass
    cleaned = re.sub(r",\s*([}\]])", lambda m: m.group(1), text)
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
    try:
        return json.loads(cleaned)
    except Exception:
        return None


def _coerce_types(data: Dict[str, Any], schema: ExtractionSchema) -> Dict[str, Any]:
    from datetime import datetime

    def to_date(s: Any) -> Optional[str]:
        if s in (None, "", []):
            return None
        if isinstance(s, (int, float)):
            return None
        candidates = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%m/%d/%Y",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
        ]
        for fmt in candidates:
            try:
                dt = datetime.strptime(str(s).strip(), fmt)
                return dt.strftime("%Y-%m-%d")
            except Exception:
                continue
        return str(s)

    out: Dict[str, Any] = {}
    for f in schema.fields:
        v = data.get(f.name)
        t = f.type.lower()
        if v is None:
            out[f.name] = [] if t.startswith("array<") else None
            continue
        if t == "number":
            try:
                out[f.name] = float(v)
            except Exception:
                m = re.search(r"[-+]?[0-9]*\.?[0-9]+", str(v))
                out[f.name] = float(m.group(0)) if m else None
        elif t == "boolean":
            out[f.name] = str(v).lower() in {"true", "yes", "1"}
        elif t == "date":
            out[f.name] = to_date(v)
        elif t.startswith("array<") and t.endswith(">"):
            inner = t[6:-1]
            arr = v if isinstance(v, list) else [v]
            coerced = []
            for item in arr:
                if inner == "number":
                    try:
                        coerced.append(float(item))
                    except Exception:
                        m = re.search(r"[-+]?[0-9]*\.?[0-9]+", str(item))
                        if m:
                            coerced.append(float(m.group(0)))
                elif inner == "boolean":
                    coerced.append(str(item).lower() in {"true", "yes", "1"})
                elif inner == "date":
                    d = to_date(item)
                    if d is not None:
                        coerced.append(d)
                else:
                    coerced.append(str(item))
            out[f.name] = coerced
        else:
            out[f.name] = str(v)
    return out


# Validators & substring gate
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?:(?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})")
_TRACKING_RE = re.compile(r"\b[A-Z0-9]{2,4}[- ]?[A-Z0-9]{3,4}[- ]?[A-Z0-9]{3,4}\b")
_ORDER_RE = re.compile(
    r"\b[0-9A-Za-z]{3,}[-][0-9A-Za-z]{2,}\b|\bORD(?:ER)?[\s:#-]*[0-9A-Za-z-]+\b",
    re.IGNORECASE,
)
_ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")
_DATE_HINT_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})\b")


def _enforce_substrings(
    text: str, entities: Dict[str, Any], schema: ExtractionSchema
) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    out = dict(entities)
    for f in schema.fields:
        if f.type.startswith("array<"):
            arr = out.get(f.name)
            if isinstance(arr, list):
                kept = [s for s in arr if isinstance(s, str) and s in text]
                if len(kept) != len(arr):
                    warnings.append(f"{f.name}: removed non-substrings from array")
                out[f.name] = kept
            continue
        if f.type == "string":
            v = out.get(f.name)
            if v not in (None, "") and not (isinstance(v, str) and v in text):
                out[f.name] = None
                warnings.append(f"{f.name}: not an exact substring -> null")
    return out, warnings


def _apply_validators(
    text: str, entities: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    out = dict(entities)
    for k, v in list(out.items()):
        if v is None:
            continue
        s, kl = str(v), k.lower()
        if "email" in kl and not _EMAIL_RE.search(s):
            out[k] = None
            warnings.append(f"{k}: invalid email -> null")
        if any(t in kl for t in ["phone", "tel", "mobile"]) and not _PHONE_RE.search(s):
            out[k] = None
            warnings.append(f"{k}: invalid phone -> null")
        if "tracking" in kl and not _TRACKING_RE.search(s):
            out[k] = None
            warnings.append(f"{k}: invalid tracking -> null")
        if any(t in kl for t in ["order_id", "order", "po_number"]) and not (
            _ORDER_RE.search(s) or _TRACKING_RE.search(s)
        ):
            out[k] = None
            warnings.append(f"{k}: invalid order -> null")
        if "address" in kl and not _ZIP_RE.search(s):
            out[k] = None
            warnings.append(f"{k}: weak address (no zip) -> null")
        if "date" in kl and not _DATE_HINT_RE.search(s):
            out[k] = None
            warnings.append(f"{k}: not a date -> null")
        if len(s.split()) > 16 and any(
            t in kl for t in ["name", "status", "issue", "address", "description"]
        ):
            warnings.append(f"{k}: suspiciously long; likely sentence dump")
    return out, warnings


# --------------------- LLM calls ---------------------
async def _extract_with_llm(req: ExtractRequest) -> Dict[str, Any]:
    client = get_client()
    prompt = _build_prompt(req.text, req.extraction_schema, req.instructions)
    schema_hint = _schema_hint(req.extraction_schema)

    @backoff.on_exception(backoff.expo, (httpx.HTTPError, RuntimeError), max_tries=3)
    async def _call() -> str:
        return await client.complete(prompt, schema_hint, req.temperature or 0.0)

    raw_text = await _call()
    parsed = _repair_json(raw_text)
    if parsed is None:
        raise ValueError("Model did not return valid JSON.")
    return parsed


async def _repair_with_llm(
    text: str, schema: ExtractionSchema, bad_fields: List[str], previous: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    client = get_client()
    keep = {k: v for k, v in previous.items() if k not in bad_fields}
    fix_list = "\n".join(f"- {bf}" for bf in bad_fields)
    fields_desc = "\n".join(
        f"- {f.name} ({f.type}): {f.description or ''}" for f in schema.fields
    )
    rules = [
        "Return ONLY JSON with the full schema keys.",
        "Keep the provided values for fields not listed below.",
        "For listed fields, replace with the SHORTEST exact substring from TEXT that matches the field definition; otherwise null.",
        "No prose, no markdown.",
    ]
    prompt = (
        "Schema repair request.\n"
        f"Fields:\n{fields_desc}\n\n"
        f"Rules:\n- " + "\n- ".join(rules) + "\n\n"
        f"Fields to fix:\n{fix_list}\n\n"
        f"Current values (JSON):\n{_json_dumps(keep)}\n\n"
        f"TEXT:\n{text}\n"
    )
    raw = await client.complete(prompt, _schema_hint(schema), temperature=0.0)
    return _repair_json(raw)


# --------------------- Heuristic schema fallback ---------------------


def _heuristic_schema_from_text(text: str, max_fields: int = 15) -> ExtractionSchema:
    slots: List[EntityField] = []

    def add(name: str, t: str, desc: str = "", required: bool = False):
        if all(f.name != name for f in slots):
            slots.append(
                EntityField(name=name, type=t, description=desc, required=required)
            )

    if _ORDER_RE.search(text):
        add("order_id", "string", "Customer order identifier", True)
    if _TRACKING_RE.search(text):
        add("tracking_number", "string", "Carrier tracking number", True)
    if _EMAIL_RE.search(text):
        add("contact_email", "string", "Customer contact email")
    if _PHONE_RE.search(text):
        add("contact_phone", "string", "Customer phone")
    if _ZIP_RE.search(text):
        add("delivery_address", "string", "Destination address")
        add("pickup_address", "string", "Origin address")
    if _DATE_HINT_RE.search(text):
        add("promised_delivery_date", "date", "Promised delivery date")
        add("last_scan_date", "date", "Last carrier scan date")

    # Common shipping fields – include some even without explicit patterns
    add("customer_name", "string", "Customer full name")
    add("package_status", "string", "Shipment status")
    add("issue_type", "string", "Issue category (late, lost, damaged)")

    if not slots:
        slots = [
            EntityField(
                name="customer_name",
                type="string",
                description="Customer full name",
                required=False,
            ),
            EntityField(
                name="order_id",
                type="string",
                description="Order identifier",
                required=False,
            ),
            EntityField(
                name="tracking_number",
                type="string",
                description="Tracking number",
                required=False,
            ),
        ]
    return ExtractionSchema(fields=slots[:max_fields])


def _prefer_common_guidance(prefer: bool) -> str:
    return (
        (
            "Prefer normalized names: customer_name, order_id, tracking_number, contact_email, contact_phone, "
            "delivery_address, pickup_address, last_scan_date, package_status, issue_type, item_description, "
            "declared_value, promised_delivery_date, refund_requested.\n"
        )
        if prefer
        else ""
    )


async def _infer_schema_with_llm(
    text: str,
    min_fields: int = 8,
    max_fields: int = 15,
    prefer_common: bool = True,
) -> ExtractionSchema:
    # ensure sensible bounds
    if max_fields < 1: max_fields = 1
    if min_fields < 1: min_fields = 1
    if max_fields < min_fields: max_fields = min_fields

    client = get_client()
    guidance = [
        "Propose concise fields capturing key entities.",
        f"Propose AT LEAST {min_fields} fields and at most {max_fields} fields.",
        "Each field: name (snake_case), type (string|number|boolean|date|array<...>), short description, required boolean.",
        _prefer_common_guidance(prefer_common),
        "Return ONLY JSON tagged with <json> and </json> — nothing else.",
        "STRICT JSON: {\"fields\":[{...}]}. Do not output prose.",
    ]
    prompt = (
        "Schema discovery.\n- " + "\n- ".join(guidance) +
        f"\n\nTEXT:\n{_shorten_for_infer(text)}\n\n"
        "<json>\n"   # NOTE: no seeded JSON here; forces generation
    )

    # Hints we keep locally, for compatibility—watsonx ignores schema hints for text.generation anyway.
    field_def = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "type": {"type": "string"},
            "description": {"type": "string"},
            "required": {"type": "boolean"},
        },
        "required": ["name", "type", "required"],
        "additionalProperties": False,
    }
    schema_hint = {"type": "object", "properties": {"fields": {"type": "array", "items": field_def}}, "required": ["fields"], "additionalProperties": False}

    raw = await client.complete(prompt, temperature=INFER_TEMPERATURE)

    if os.getenv("LOG_LLM_RAW", "").lower() == "true":
        _log("LLM raw (infer) head:", raw[:800].replace("\n", "⏎"))

    parsed = _repair_json(raw) or {"fields": []}

    out_fields: List[EntityField] = []
    allowed = {"string","number","boolean","date","array<string>","array<number>","array<boolean>","array<date>"}
    for f in parsed.get("fields", []):
        t = str(f.get("type", "string")).lower().strip()
        if t not in allowed: t = "string"
        name = re.sub(r"[^a-z0-9_]", "_", str(f.get("name", "")).lower()) or "field"
        desc = f.get("description") or ""
        req = bool(f.get("required", True))
        out_fields.append(EntityField(name=name, type=t, description=desc, required=req))

    # dedupe + cap
    seen = set(); uniq: List[EntityField] = []
    for f in out_fields:
        if f.name not in seen:
            uniq.append(f); seen.add(f.name)
    uniq = uniq[:max_fields]

    # If LLM returned nothing, use heuristics (existing behavior)
    if not uniq:
        _log("LLM inference produced 0 fields; using heuristic schema")
        uniq = _heuristic_schema_from_text(text, max_fields).fields

    # Enforce a minimum by padding with canonical shipping fields (from your v1 min_fields work)
    if len(uniq) < min_fields:
        canonical = [
            EntityField(name="customer_name", type="string", description="Customer full name"),
            EntityField(name="order_id", type="string", description="Customer order identifier"),
            EntityField(name="tracking_number", type="string", description="Carrier tracking number"),
            EntityField(name="contact_email", type="string", description="Customer contact email"),
            EntityField(name="contact_phone", type="string", description="Customer phone"),
            EntityField(name="delivery_address", type="string", description="Destination address"),
            EntityField(name="pickup_address", type="string", description="Origin address"),
            EntityField(name="promised_delivery_date", type="date", description="Promised delivery date"),
            EntityField(name="last_scan_date", type="date", description="Last carrier scan date"),
            EntityField(name="package_status", type="string", description="Shipment status"),
            EntityField(name="issue_type", type="string", description="Issue category"),
            EntityField(name="item_description", type="string", description="Item description"),
            EntityField(name="declared_value", type="number", description="Declared value"),
            EntityField(name="refund_requested", type="boolean", description="Refund requested"),
        ]
        have = {f.name for f in uniq}
        for f in canonical:
            if len(uniq) >= min_fields or len(uniq) >= max_fields: break
            if f.name not in have:
                uniq.append(f); have.add(f.name)

    return ExtractionSchema(fields=uniq)


# --------------------- FastAPI ---------------------
app = FastAPI(title="LLM Entity Extractor (watsonx-only)", version="1.3.0")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest) -> ExtractResponse:
    t0 = time.time()
    request_id = str(uuid.uuid4())
    warnings: List[str] = []

    try:
        raw_entities = await _extract_with_llm(req)
    except Exception as e:
        warnings.append(f"LLM extraction failed: {e}.")
        raw_entities = {f.name: None for f in req.extraction_schema.fields}

    try:
        entities = _coerce_types(raw_entities, req.extraction_schema)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Type coercion failed: {e}")

    # Gates
    entities, w1 = _enforce_substrings(req.text, entities, req.extraction_schema)
    entities, w2 = _apply_validators(req.text, entities)
    warnings.extend(w1 + w2)

    # Optional repair if many fields failed
    bad = [k for k, v in entities.items() if v in (None, [], "")]
    if len(bad) >= max(2, len(req.extraction_schema.fields) // 3):
        try:
            repaired = await _repair_with_llm(
                req.text, req.extraction_schema, bad, entities
            )
            if repaired:
                entities = _coerce_types(repaired, req.extraction_schema)
        except Exception:
            warnings.append("Repair step failed; returning best-effort values.")

    required = [f.name for f in req.extraction_schema.fields if f.required]
    present = sum(1 for n in required if entities.get(n) not in (None, [], ""))
    confidence = (
        round(present / max(1, len(required)), 2) if req.return_confidence else None
    )

    return ExtractResponse(
        request_id=request_id,
        model=os.getenv("MODEL", "openai/gpt-oss-120b"),
        provider="watsonx",
        latency_ms=int((time.time() - t0) * 1000),
        entities=entities,
        confidence=confidence,
        warnings=warnings,
        raw=raw_entities
        if os.getenv("INCLUDE_RAW", "false").lower() == "true"
        else None,
    )


@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest) -> InferResponse:
    schema = await _infer_schema_with_llm(
        req.text, req.max_fields, req.min_fields, req.prefer_common_fields
    )
    _log(f"/infer fields={len(schema.fields)}")
    return InferResponse(fields=schema.fields)


@app.post("/auto-extract", response_model=AutoExtractResponse)
async def auto_extract(req: AutoExtractRequest) -> AutoExtractResponse:
    schema = await _infer_schema_with_llm(
        req.text, req.max_fields, req.min_fields, True
    )
    _log(f"/auto-extract inferred fields={len(schema.fields)}")
    # Never let schema be empty
    if not schema.fields:
        schema = _heuristic_schema_from_text(req.text, req.max_fields)
        _log(f"/auto-extract heuristic fields={len(schema.fields)}")
    ex_req = ExtractRequest(
        text=req.text, extraction_schema=schema, temperature=req.temperature or 0.0
    )
    extraction = await extract(ex_req)  # reuse handler
    _log(
        f"/auto-extract entities_present={sum(1 for v in extraction.entities.values() if v not in (None, [], ''))}"
    )
    return AutoExtractResponse(schema=schema, extraction=extraction)


@app.post("/extract_pick", response_model=PickResponse)
async def extract_pick(req: PickRequest) -> PickResponse:
    t0 = time.time()
    rid = str(uuid.uuid4())
    warnings: List[str] = []

    sel_props = {
        f.name: {"type": ["integer", "null"]} for f in req.extraction_schema.fields
    }
    sel_hint = {
        "type": "object",
        "properties": {
            "selection": {
                "type": "object",
                "properties": sel_props,
                "additionalProperties": False,
            }
        },
        "required": ["selection"],
        "additionalProperties": False,
    }

    client = get_client()
    prompt = _build_pick_prompt(req.text, req.extraction_schema, req.candidates)
    raw = await client.complete(prompt, sel_hint, temperature=req.temperature or 0.0)
    parsed = _repair_json(raw) or {"selection": {}}
    selection: Dict[str, Optional[int]] = parsed.get("selection", {})

    chosen: Dict[str, Any] = {}
    for f in req.extraction_schema.fields:
        idx = selection.get(f.name)
        arr = req.candidates.get(f.name, [])
        if isinstance(idx, int) and 0 <= idx < len(arr):
            chosen[f.name] = arr[idx]
        else:
            chosen[f.name] = None

    entities = _coerce_types(chosen, req.extraction_schema)
    entities, w1 = _enforce_substrings(req.text, entities, req.extraction_schema)
    entities, w2 = _apply_validators(req.text, entities)
    warnings.extend(w1 + w2)

    return PickResponse(
        request_id=rid,
        model=os.getenv("MODEL", "meta-llama/llama-3-405b-instruct"),
        provider="watsonx",
        latency_ms=int((time.time() - t0) * 1000),
        entities=entities,
        selection=selection,
        warnings=warnings,
    )
