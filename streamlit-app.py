# """
# Streamlit UI for LLM Entity Extractor
# -------------------------------------

# Run locally:
# 1) Install deps (in a clean venv):
#    pip install streamlit==1.37.0 httpx==0.27.0 pydantic==2.8.2 orjson==3.10.7

# 2) Start your FastAPI backend (from the other canvas):
#    uvicorn app:app --reload --port 8000

# 3) Launch the UI:
#    export EXTRACTOR_API=http://localhost:8000  # optional; you can set in the UI too
#    streamlit run streamlit_app.py

# Notes:
# - Supports two modes:
#   (A) Extract (you provide a schema)
#   (B) Auto‑extract (server infers schema via /auto-extract; will gracefully fall back to /infer→/extract if /auto-extract is missing or returns an empty schema)
# - Generates an equivalent curl for any request you make from the UI.
# """
# import json
# import os
# import time
# from typing import Any, Dict, Optional

# import httpx
# import streamlit as st

# DEFAULT_API = os.getenv("EXTRACTOR_API", "http://localhost:8000")
# DEFAULT_SCHEMA = {
#     "fields": [
#         {"name": "order_id", "type": "string", "required": True, "description": "Customer order identifier"},
#         {"name": "tracking_number", "type": "string", "required": True},
#         {"name": "customer_name", "type": "string", "required": True},
#         {"name": "contact_email", "type": "string"},
#         {"name": "contact_phone", "type": "string"},
#         {"name": "issue_type", "type": "string", "description": "e.g., late delivery, lost, damaged"},
#         {"name": "pickup_address", "type": "string"},
#         {"name": "delivery_address", "type": "string"},
#         {"name": "promised_delivery_date", "type": "date"},
#         {"name": "last_scan_date", "type": "date"},
#         {"name": "last_scan_location", "type": "string"},
#         {"name": "package_status", "type": "string"},
#         {"name": "item_description", "type": "string"},
#         {"name": "declared_value", "type": "number"},
#         {"name": "refund_requested", "type": "boolean"},
#         {"name": "preferred_contact_method", "type": "string"},
#     ]
# }

# st.set_page_config(page_title="LLM Entity Extractor UI", layout="wide")
# st.title("🔎 LLM Entity Extractor – Streamlit UI")

# with st.sidebar:
#     st.header("API Settings")
#     api_url = st.text_input("Extractor API base URL", value=DEFAULT_API, help="FastAPI service base URL")
#     mode = st.radio(
#         "Mode",
#         options=["Extract (provide schema)", "Auto-extract (infer schema server-side)"]
#     )
#     temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
#     max_fields = st.number_input("Max fields when inferring", min_value=3, max_value=40, value=15, step=1)
#     show_raw = st.checkbox("Show raw response", value=False)

# # Text input
# col1, col2 = st.columns([3, 2])
# with col1:
#     st.subheader("Input Text")
#     uploaded = st.file_uploader("Optional: upload a .txt file", type=["txt"], accept_multiple_files=False)
#     text_value = ""
#     if uploaded is not None:
#         try:
#             text_value = uploaded.read().decode("utf-8", errors="ignore")
#         except Exception:
#             st.warning("Could not decode file; falling back to empty text.")
#     text_value = st.text_area("Paste conversation / document", value=text_value, height=300)

# with col2:
#     st.subheader("Schema / Options")
#     instructions = st.text_area("Optional extraction instructions", help="e.g., 'Prefer the most recent dates' or domain hints")
#     schema_text: Optional[str] = None
#     if mode.startswith("Extract"):
#         schema_text = st.text_area("Schema (JSON)", value=json.dumps(DEFAULT_SCHEMA, indent=2), height=300)

# # Action button
# run = st.button("🚀 Run Extraction")

# # Helpers

# def _pretty_json(data: Dict[str, Any]) -> str:
#     try:
#         import orjson  # type: ignore
#         return orjson.dumps(data, option=orjson.OPT_INDENT_2).decode()
#     except Exception:
#         return json.dumps(data, indent=2)


# def _curl_for(endpoint: str, payload: Dict[str, Any]) -> str:
#     body = json.dumps(payload).replace("\\n", "\\n")
#     return (
#         "curl -X POST "
#         + f"{api_url}{endpoint} "
#         + "-H 'Content-Type: application/json' "
#         + f"-d '{body}'"
#     )


# # Main logic
# if run:
#     if not text_value.strip():
#         st.error("Please provide some input text.")
#     else:
#         with st.spinner("Calling extractor..."):
#             t0 = time.time()
#             client = httpx.Client(timeout=120)
#             try:
#                 if mode.startswith("Auto-extract"):
#                     # First attempt: /auto-extract
#                     payload = {"text": text_value, "max_fields": int(max_fields), "temperature": float(temperature)}
#                     try:
#                         r = client.post(f"{api_url}/auto-extract", json=payload)
#                         if r.status_code == 404:
#                             raise httpx.HTTPStatusError("/auto-extract missing", request=r.request, response=r)
#                         r.raise_for_status()
#                         data = r.json()
#                         elapsed = (time.time() - t0) * 1000

#                         # Check for empty schema and gracefully fall back if needed
#                         schema_obj = data.get("schema", {}) or {}
#                         extraction_obj = data.get("extraction", {}) or {}
#                         entities_obj = extraction_obj.get("entities", {}) or {}
#                         empty_schema = (not schema_obj) or (not schema_obj.get("fields"))

#                         if empty_schema:
#                             st.info("Server returned empty schema; falling back to /infer → /extract …")
#                             infer_payload = {"text": text_value, "max_fields": int(max_fields), "prefer_common_fields": True}
#                             r1 = client.post(f"{api_url}/infer", json=infer_payload)
#                             r1.raise_for_status()
#                             inferred = r1.json()
#                             schema = {"fields": inferred.get("fields", [])}

#                             ex_payload = {
#                                 "text": text_value,
#                                 "schema": schema,
#                                 "temperature": float(temperature),
#                                 "instructions": instructions or None,
#                             }
#                             r2 = client.post(f"{api_url}/extract", json=ex_payload)
#                             r2.raise_for_status()
#                             ex_data = r2.json()

#                             st.success("Inference + Extraction succeeded (client fallback)")
#                             st.caption(f"Latency: {elapsed:.0f} ms (combined)")
#                             st.subheader("Inferred Schema")
#                             st.json(schema)
#                             st.subheader("Entities")
#                             st.json(ex_data.get("entities", {}))

#                             if show_raw:
#                                 st.subheader("Raw Inference Response")
#                                 st.code(_pretty_json(inferred))
#                                 st.subheader("Raw Extraction Response")
#                                 st.code(_pretty_json(ex_data))

#                             st.subheader("Equivalent curl (infer)")
#                             st.code(_curl_for("/infer", infer_payload))
#                             st.subheader("Equivalent curl (extract)")
#                             st.code(_curl_for("/extract", ex_payload))
#                         else:
#                             st.success("Auto-extraction succeeded")
#                             st.caption(f"Latency: {elapsed:.0f} ms")
#                             st.subheader("Inferred Schema")
#                             st.json(schema_obj)
#                             st.subheader("Entities")
#                             st.json(entities_obj)

#                             if show_raw:
#                                 st.subheader("Raw Response")
#                                 st.code(_pretty_json(data))

#                             st.subheader("Equivalent curl")
#                             st.code(_curl_for("/auto-extract", payload))
#                     except httpx.HTTPStatusError:
#                         # Fallback: /infer -> /extract chain
#                         st.info("/auto-extract not available; falling back to /infer → /extract …")
#                         infer_payload = {"text": text_value, "max_fields": int(max_fields), "prefer_common_fields": True}
#                         r1 = client.post(f"{api_url}/infer", json=infer_payload)
#                         r1.raise_for_status()
#                         inferred = r1.json()
#                         schema = {"fields": inferred.get("fields", [])}
#                         ex_payload = {
#                             "text": text_value,
#                             "schema": schema,
#                             "temperature": float(temperature),
#                             "instructions": instructions or None,
#                         }
#                         r2 = client.post(f"{api_url}/extract", json=ex_payload)
#                         r2.raise_for_status()
#                         ex_data = r2.json()
#                         elapsed = (time.time() - t0) * 1000

#                         st.success("Inference + Extraction succeeded")
#                         st.caption(f"Latency: {elapsed:.0f} ms (combined)")
#                         st.subheader("Inferred Schema")
#                         st.json(schema)
#                         st.subheader("Entities")
#                         st.json(ex_data.get("entities", {}))

#                         if show_raw:
#                             st.subheader("Raw Inference Response")
#                             st.code(_pretty_json(inferred))
#                             st.subheader("Raw Extraction Response")
#                             st.code(_pretty_json(ex_data))

#                         st.subheader("Equivalent curl (infer)")
#                         st.code(_curl_for("/infer", infer_payload))
#                         st.subheader("Equivalent curl (extract)")
#                         st.code(_curl_for("/extract", ex_payload))

#                 else:
#                     # Mode: Extract (user provides schema)
#                     try:
#                         schema = json.loads(schema_text or "{}")
#                     except Exception as e:
#                         st.error(f"Schema JSON is invalid: {e}")
#                         st.stop()

#                     ex_payload = {
#                         "text": text_value,
#                         "schema": schema,
#                         "temperature": float(temperature),
#                         "instructions": instructions or None,
#                     }
#                     r = client.post(f"{api_url}/extract", json=ex_payload)
#                     r.raise_for_status()
#                     data = r.json()
#                     elapsed = (time.time() - t0) * 1000

#                     st.success("Extraction succeeded")
#                     st.caption(f"Latency: {elapsed:.0f} ms")

#                     # Quick metadata row
#                     meta_cols = st.columns(4)
#                     meta_cols[0].metric("Confidence", str(data.get("confidence", "")))
#                     meta_cols[1].metric("Provider", data.get("provider", ""))
#                     meta_cols[2].metric("Model", data.get("model", ""))
#                     meta_cols[3].metric("Latency (ms)", str(data.get("latency_ms", "")))

#                     st.subheader("Entities")
#                     st.json(data.get("entities", {}))

#                     if data.get("warnings"):
#                         st.warning("\n".join(data["warnings"]))

#                     if show_raw:
#                         st.subheader("Raw Response")
#                         st.code(_pretty_json(data))

#                     st.subheader("Equivalent curl")
#                     st.code(_curl_for("/extract", ex_payload))

#             except httpx.HTTPStatusError as e:
#                 try:
#                     err_json = e.response.json()
#                 except Exception:
#                     err_json = {"detail": e.response.text}
#                 st.error(f"HTTP {e.response.status_code}: {err_json}")
#             except httpx.HTTPError as e:
#                 st.error(f"HTTP error: {e}")
#             except Exception as e:
#                 st.error(f"Unexpected error: {e}")
#             finally:
#                 client.close()

"""
Streamlit UI for LLM Entity Extractor
-------------------------------------

Run locally:
1) Install deps (in a clean venv):
   pip install streamlit==1.37.0 httpx==0.27.0 pydantic==2.8.2 orjson==3.10.7

2) Start your FastAPI backend (from the other canvas):
   uvicorn app:app --reload --port 8000

3) Launch the UI:
   export EXTRACTOR_API=http://localhost:8000  # optional; you can set in the UI too
   streamlit run streamlit_app.py

Notes:
- Supports two modes:
  (A) Extract (you provide a schema)
  (B) Auto‑extract (server infers schema via /auto-extract; will gracefully fall back to /infer→/extract if /auto-extract is missing or returns an empty schema)
- Generates an equivalent curl for any request you make from the UI.
"""
import json
import os
import time
from typing import Any, Dict, Optional

import httpx
import streamlit as st

DEFAULT_API = os.getenv("EXTRACTOR_API", "http://localhost:8000")
DEFAULT_SCHEMA = {
    "fields": [
        {
            "name": "order_id",
            "type": "string",
            "required": True,
            "description": "Customer order identifier",
        },
        {"name": "tracking_number", "type": "string", "required": True},
        {"name": "customer_name", "type": "string", "required": True},
        {"name": "contact_email", "type": "string"},
        {"name": "contact_phone", "type": "string"},
        {
            "name": "issue_type",
            "type": "string",
            "description": "e.g., late delivery, lost, damaged",
        },
        {"name": "pickup_address", "type": "string"},
        {"name": "delivery_address", "type": "string"},
        {"name": "promised_delivery_date", "type": "date"},
        {"name": "last_scan_date", "type": "date"},
        {"name": "last_scan_location", "type": "string"},
        {"name": "package_status", "type": "string"},
        {"name": "item_description", "type": "string"},
        {"name": "declared_value", "type": "number"},
        {"name": "refund_requested", "type": "boolean"},
        {"name": "preferred_contact_method", "type": "string"},
    ]
}

st.set_page_config(page_title="LLM Entity Extractor UI", layout="wide")
st.title("🔎 LLM Entity Extractor – Streamlit UI")

with st.sidebar:
    st.header("API Settings")
    api_url = st.text_input(
        "Extractor API base URL", value=DEFAULT_API, help="FastAPI service base URL"
    )
    mode = st.radio(
        "Mode",
        options=["Extract (provide schema)", "Auto-extract (infer schema server-side)"],
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    max_fields = st.number_input(
        "Max fields when inferring", min_value=3, max_value=40, value=15, step=1
    )
    min_fields = st.number_input(
        "Min fields when inferring", min_value=1, max_value=40, value=8, step=1
    )
    show_raw = st.checkbox("Show raw response", value=False)

# Text input
col1, col2 = st.columns([3, 2])
with col1:
    st.subheader("Input Text")
    uploaded = st.file_uploader(
        "Optional: upload a .txt file", type=["txt"], accept_multiple_files=False
    )
    text_value = ""
    if uploaded is not None:
        try:
            text_value = uploaded.read().decode("utf-8", errors="ignore")
        except Exception:
            st.warning("Could not decode file; falling back to empty text.")
    text_value = st.text_area(
        "Paste conversation / document", value=text_value, height=300
    )

with col2:
    st.subheader("Schema / Options")
    instructions = st.text_area(
        "Optional extraction instructions",
        help="e.g., 'Prefer the most recent dates' or domain hints",
    )
    schema_text: Optional[str] = None
    if mode.startswith("Extract"):
        schema_text = st.text_area(
            "Schema (JSON)", value=json.dumps(DEFAULT_SCHEMA, indent=2), height=300
        )

# Action button
run = st.button("🚀 Run Extraction")

# Helpers


def _pretty_json(data: Dict[str, Any]) -> str:
    try:
        import orjson  # type: ignore

        return orjson.dumps(data, option=orjson.OPT_INDENT_2).decode()
    except Exception:
        return json.dumps(data, indent=2)


def _curl_for(endpoint: str, payload: Dict[str, Any]) -> str:
    body = json.dumps(payload).replace("\\n", "\\n")
    return (
        "curl -X POST "
        + f"{api_url}{endpoint} "
        + "-H 'Content-Type: application/json' "
        + f"-d '{body}'"
    )


# Main logic
if run:
    if not text_value.strip():
        st.error("Please provide some input text.")
    else:
        with st.spinner("Calling extractor..."):
            t0 = time.time()
            client = httpx.Client(timeout=120)
            try:
                if mode.startswith("Auto-extract"):
                    # First attempt: /auto-extract
                    payload = {
                        "text": text_value,
                        "max_fields": int(max_fields),
                        "temperature": float(temperature),
                    }
                    try:
                        r = client.post(f"{api_url}/auto-extract", json=payload)
                        if r.status_code == 404:
                            raise httpx.HTTPStatusError(
                                "/auto-extract missing", request=r.request, response=r
                            )
                        r.raise_for_status()
                        data = r.json()
                        elapsed = (time.time() - t0) * 1000

                        # Check for empty schema and gracefully fall back if needed
                        schema_obj = data.get("schema", {}) or {}
                        extraction_obj = data.get("extraction", {}) or {}
                        entities_obj = extraction_obj.get("entities", {}) or {}
                        empty_schema = (not schema_obj) or (
                            not schema_obj.get("fields")
                        )

                        if empty_schema:
                            st.info(
                                "Server returned empty schema; falling back to /infer → /extract …"
                            )
                            infer_payload = {
                                "text": text_value,
                                "max_fields": int(max_fields),
                                "prefer_common_fields": True,
                            }
                            r1 = client.post(f"{api_url}/infer", json=infer_payload)
                            r1.raise_for_status()
                            inferred = r1.json()
                            schema = {"fields": inferred.get("fields", [])}

                            ex_payload = {
                                "text": text_value,
                                "schema": schema,
                                "temperature": float(temperature),
                                "instructions": instructions or None,
                                "min_fields": int(min_fields),
                            }
                            r2 = client.post(f"{api_url}/extract", json=ex_payload)
                            r2.raise_for_status()
                            ex_data = r2.json()

                            st.success(
                                "Inference + Extraction succeeded (client fallback)"
                            )
                            st.caption(f"Latency: {elapsed:.0f} ms (combined)")
                            st.subheader("Inferred Schema")
                            st.json(schema)
                            st.subheader("Entities")
                            st.json(ex_data.get("entities", {}))

                            if show_raw:
                                st.subheader("Raw Inference Response")
                                st.code(_pretty_json(inferred))
                                st.subheader("Raw Extraction Response")
                                st.code(_pretty_json(ex_data))

                            st.subheader("Equivalent curl (infer)")
                            st.code(_curl_for("/infer", infer_payload))
                            st.subheader("Equivalent curl (extract)")
                            st.code(_curl_for("/extract", ex_payload))
                        else:
                            st.success("Auto-extraction succeeded")
                            st.caption(f"Latency: {elapsed:.0f} ms")
                            st.subheader("Inferred Schema")
                            st.json(schema_obj)
                            st.subheader("Entities")
                            st.json(entities_obj)

                            if show_raw:
                                st.subheader("Raw Response")
                                st.code(_pretty_json(data))

                            st.subheader("Equivalent curl")
                            st.code(_curl_for("/auto-extract", payload))
                    except httpx.HTTPStatusError:
                        # Fallback: /infer -> /extract chain
                        st.info(
                            "/auto-extract not available; falling back to /infer → /extract …"
                        )
                        infer_payload = {
                            "text": text_value,
                            "max_fields": int(max_fields),
                            "prefer_common_fields": True,
                        }
                        r1 = client.post(f"{api_url}/infer", json=infer_payload)
                        r1.raise_for_status()
                        inferred = r1.json()
                        schema = {"fields": inferred.get("fields", [])}
                        ex_payload = {
                            "text": text_value,
                            "schema": schema,
                            "temperature": float(temperature),
                            "instructions": instructions or None,
                            "min_fields": int(min_fields),
                        }
                        r2 = client.post(f"{api_url}/extract", json=ex_payload)
                        r2.raise_for_status()
                        ex_data = r2.json()
                        elapsed = (time.time() - t0) * 1000

                        st.success("Inference + Extraction succeeded")
                        st.caption(f"Latency: {elapsed:.0f} ms (combined)")
                        st.subheader("Inferred Schema")
                        st.json(schema)
                        st.subheader("Entities")
                        st.json(ex_data.get("entities", {}))

                        if show_raw:
                            st.subheader("Raw Inference Response")
                            st.code(_pretty_json(inferred))
                            st.subheader("Raw Extraction Response")
                            st.code(_pretty_json(ex_data))

                        st.subheader("Equivalent curl (infer)")
                        st.code(_curl_for("/infer", infer_payload))
                        st.subheader("Equivalent curl (extract)")
                        st.code(_curl_for("/extract", ex_payload))

                else:
                    # Mode: Extract (user provides schema)
                    try:
                        schema = json.loads(schema_text or "{}")
                    except Exception as e:
                        st.error(f"Schema JSON is invalid: {e}")
                        st.stop()

                    ex_payload = {
                        "text": text_value,
                        "schema": schema,
                        "temperature": float(temperature),
                        "instructions": instructions or None,
                        "min_fields": int(min_fields),
                    }
                    r = client.post(f"{api_url}/extract", json=ex_payload)
                    r.raise_for_status()
                    data = r.json()
                    elapsed = (time.time() - t0) * 1000

                    st.success("Extraction succeeded")
                    st.caption(f"Latency: {elapsed:.0f} ms")

                    # Quick metadata row
                    meta_cols = st.columns(4)
                    meta_cols[0].metric("Confidence", str(data.get("confidence", "")))
                    meta_cols[1].metric("Provider", data.get("provider", ""))
                    meta_cols[2].metric("Model", data.get("model", ""))
                    meta_cols[3].metric("Latency (ms)", str(data.get("latency_ms", "")))

                    st.subheader("Entities")
                    st.json(data.get("entities", {}))

                    if data.get("warnings"):
                        st.warning("\n".join(data["warnings"]))

                    if show_raw:
                        st.subheader("Raw Response")
                        st.code(_pretty_json(data))

                    st.subheader("Equivalent curl")
                    st.code(_curl_for("/extract", ex_payload))

            except httpx.HTTPStatusError as e:
                try:
                    err_json = e.response.json()
                except Exception:
                    err_json = {"detail": e.response.text}
                st.error(f"HTTP {e.response.status_code}: {err_json}")
            except httpx.HTTPError as e:
                st.error(f"HTTP error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
            finally:
                client.close()
