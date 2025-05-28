import streamlit as st
import os
import csv
import pyperclip as py
import duckdb
import pandas as pd
import random
import re
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from io import StringIO
from faker import Faker

# Enable detailed error messages
st.set_option('client.showErrorDetails', True)

# Sidebar settings for model selection
st.sidebar.title("Settings")

# --- Dynamic Roadmap Sidebar Tracker ---
# if "roadmap_status" not in st.session_state:
#     st.session_state.roadmap_status = {}

# roadmap = {
#     "Phase 1: Foundation & Usability": {
#         "clipboard": "Copy-to-clipboard",
#         "sql_formatting": "SQL query formatting",
#         "csv_export": "CSV export of logs",
#         "edit_reask": "One-click 'Re-ask with edit'",
#         "tagging": "Tag/favorite previous questions",
#         "tabbed_preview": "Dynamic preview tab",
#         "inline_validation": "Inline schema validation",
#         "schema_inspector": "Live schema inspector",
#         "sample_data": "Sample data preview using DuckDB",
#     },
#     "Phase 2: Intelligence & Interactivity": {
#         "refine_query": "Refine this query (functional)",
#         "followups": "Follow-up instruction support",
#         "templates": "Prompt templates + suggested dropdown",
#         "refine_loop": "LLM-guided refinement loop",
#         "join_suggestions": "Join recommendations via prompt",
#         "nl_errors": "Natural language error handling",
#     },
#     "Phase 3: Power User & Deployment": {
#         "model_selector": "Model selector",
#         "token_usage": "Token usage & cost display",
#         "theme_toggle": "Theme toggle (light/dark)",
#         "save_state": "Save/load session state",
#         "secure_mode": "Secure multi-user mode",
#         "diff_view": "Version/diff view",
#     },
#     "Phase 4: Analyst-Centric Differentiators": {
#         "visualizations": "Visualizations (Altair/Plotly)",
#         "sql_explainer": "Query Explanation Engine",
#         "semantic_layer": "Semantic aliasing (YAML/UI)",
#         "multilingual": "Multilingual prompting",
#     },
# }

# with st.sidebar.expander("🗺️ Roadmap Progress", expanded=False):
#     for phase, features in roadmap.items():
#         st.markdown(f"### {phase}")
#         for key, label in features.items():
#             state_key = f"roadmap_{key}"
#             default = st.session_state.roadmap_status.get(state_key, False)
#             value = st.checkbox(label, value=default, key=state_key)
#             st.session_state.roadmap_status[state_key] = value

#Choose GPT Model
model_options = {
    "gpt-3.5-turbo": "💨 GPT-3.5 Turbo ($0.50 in / $1.50 out)",
    "o4-mini": "⚡ o4-mini ($1.10 in / $4.40 out)",
    "gpt-4.1": "🧠 GPT-4.1 ($2.00 in / $8.00 out)",
    "gpt-4-turbo": "🚀 GPT-4 Turbo ($10.00 in / $30.00 out)",
}

model_labels = list(model_options.values())
model_keys = list(model_options.keys())

default_model = model_keys.index("gpt-4.1")

selected_label = st.sidebar.selectbox("Choose a model:", model_labels, index=default_model)
st.session_state["model"] = model_keys[model_labels.index(selected_label)]

# model_choice = st.sidebar.selectbox(
#     "Choose a model:",
#     options=["gpt-3.5-turbo", "gpt-4-turbo"],
#     index=1
# )
#st.session_state["model"] = model_choice

with st.sidebar.expander("🗺️ Roadmap Progress", expanded=False):
    st.markdown("### Phase 1: Foundation & Usability")
    st.markdown("✅ Copy-to-clipboard")
    st.markdown("✅ SQL query formatting")
    st.markdown("✅ CSV export of logs")
    st.markdown("✅ One-click 'Re-ask with edit'")
    st.markdown("✅ Tag/favorite previous questions")
    st.markdown("✅ Dynamic preview tab tied to selected SQL")
    st.markdown("✅ Inline schema validation")
    st.markdown("✅ Live schema inspector panel")
    st.markdown("✅ Sample data preview using DuckDB")
    st.markdown("🟡 'Wider output box' — superseded by tabbed layout")

    st.markdown("---")
    st.markdown("### Phase 2: Intelligence & Interactivity")
    st.markdown("✅ Refine this query (functional)")
    st.markdown("✅ Follow-up instruction support")
    st.markdown("✅ Prompt templates + suggested dropdown")
    st.markdown("🟡 LLM-guided refinement loop — polishing")
    st.markdown("🟡 Join recommendations via prompt logic")
    st.markdown("⬜ Natural language error explanations")

    st.markdown("---")
    st.markdown("### Phase 3: Power User & Deployment")
    st.markdown("✅ Model selector")
    st.markdown("✅ Token usage & cost estimate")
    st.markdown("⬜ Theme toggle (light/dark)")
    st.markdown("⬜ Save/load session state")
    st.markdown("⬜ Secure multi-user mode")
    st.markdown("⬜ Version/diff view")

    st.markdown("---")
    st.markdown("### Phase 4: Analyst-Centric Differentiators")
    st.markdown("⬜ Visualizations (Altair/Plotly)")
    st.markdown("⬜ Query Explanation Engine")
    st.markdown("⬜ Semantic aliasing (YAML or UI)")
    st.markdown("⬜ Multilingual prompting")


# Initialize Faker globally
fake = Faker()

def infer_relationships(schema_text):
    tables = {}
    lines = schema_text.strip().split("\n")
    for line in lines:
        match = re.match(r"(\w+)\((.*?)\)", line.strip())
        if not match:
            continue
        table, cols = match.groups()
        tables[table] = [col.strip() for col in cols.split(",")]

    relationships = []
    for table, cols in tables.items():
        for col in cols:
            if col.lower().endswith("_id"):
                ref_table_singular = col.lower().replace("_id", "")
                for candidate in tables:
                    if (
                        candidate.lower() == ref_table_singular
                        or candidate.lower().rstrip("s") == ref_table_singular
                        or candidate.lower() + "s" == ref_table_singular
                    ):
                        relationships.append(f"{table}.{col} → {candidate}.id")
                        break
    return relationships


def generate_mock_dataframe_from_schema(schema_text):
    def infer_column_type(col_name):
        col = col_name.lower()
        if any(k in col for k in ['id', 'number', 'count', 'qty', 'quantity']):
            return 'int'
        elif any(k in col for k in ['amount', 'total', 'price', 'cost', 'revenue']):
            return 'float'
        elif any(k in col for k in ['date', 'time']):
            return 'date'
        elif 'email' in col:
            return 'email'
        elif 'name' in col:
            return 'name'
        elif any(k in col for k in ['desc', 'description', 'text', 'note']):
            return 'text'
        elif 'country' in col:
            return 'country'
        elif 'region' in col:
            return 'region'
        else:
            return 'string'

    tables = {}
    reference_data = {}
    lines = schema_text.strip().split("\n")

    for line in lines:
        match = re.match(r"(\w+)\((.*?)\)", line.strip())
        if not match:
            continue
        table_name, columns_raw = match.groups()
        columns = [col.strip() for col in columns_raw.split(",")]
        data = {}

        for col in columns:
            col_type = infer_column_type(col)
            if col_type == 'int':
                data[col] = [i + 1 for i in range(10)]
            elif col_type == 'float':
                data[col] = [round(random.uniform(100, 1000), 2) for _ in range(10)]
            elif col_type == 'date':
                data[col] = [fake.date_between(start_date='-2y', end_date='today') for _ in range(10)]
            elif col_type == 'email':
                data[col] = [fake.email() for _ in range(10)]
            elif col_type == 'name':
                data[col] = [fake.name() for _ in range(10)]
            elif col_type == 'text':
                data[col] = [fake.sentence() for _ in range(10)]
            elif col_type == 'country':
                data[col] = [random.choice(['US', 'CA', 'UK']) for _ in range(10)]
            elif col_type == 'region':
                data[col] = [random.choice(['East', 'West', 'South', 'North']) for _ in range(10)]
            else:
                data[col] = [fake.word() for _ in range(10)]

        df = pd.DataFrame(data)
        for col in df.columns:
            if 'country' in col.lower() or 'region' in col.lower():
                df[col] = df[col].astype(str)

        tables[table_name] = df
        if 'id' in df.columns:
            reference_data[table_name] = df['id'].tolist()

    for table_name, df in tables.items():
        for col in df.columns:
            if col.lower().endswith('_id'):
                ref_table = col.lower().replace('_id', '')
                if ref_table in reference_data:
                    df[col] = [random.choice(reference_data[ref_table]) for _ in range(len(df))]
                else:
                    df[col] = [random.randint(1, 10) for _ in range(len(df))]

    return tables

def parse_schema(schema_text):
    tables = {}
    lines = schema_text.strip().split("\n")
    for line in lines:
        match = re.match(r"(\w+)\((.*?)\)", line.strip())
        if match:
            table, cols = match.groups()
            columns = [col.strip() for col in cols.split(",")]
            tables[table] = columns
    return tables

# Initialize OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize OpenAI client
# client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# Initialize session state
if "sql_output" not in st.session_state:
    st.session_state.sql_output = None
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "last_schema" not in st.session_state:
    st.session_state.last_schema = ""
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
# Add 'tags' key to older entries (if missing)
for entry in st.session_state.conversation_history:
    if "tags" not in entry:
        entry["tags"] = []
if "follow_up_input" not in st.session_state:
    st.session_state.follow_up_input = ""


# Backfill missing 'favorite' field for older entries
for entry in st.session_state.conversation_history:
    if "favorite" not in entry:
        entry["favorite"] = False

st.title("🧠 Natural Language to SQL Assistant")

if "inject_ref" in st.session_state:
    prev = st.session_state.get("last_question", "")
    st.session_state["last_question"] = f"{prev} {st.session_state.pop('inject_ref')}".strip()

# Check if we're in edit mode and set values BEFORE widgets are instantiated
if st.session_state.get("trigger_edit_mode"):
    st.session_state["last_question"] = st.session_state.get("edit_mode_question", "")
    st.session_state["follow_up_input"] = st.session_state.get("edit_mode_followup", "")
    st.session_state["last_schema"] = st.session_state.get("edit_mode_schema", "")
    st.session_state["trigger_edit_mode"] = False
    st.toast("✏️ Loaded previous query for editing.")

st.sidebar.markdown(f"🔍 last_question: `{st.session_state.get('last_question', '')}`")

if "inject_template" in st.session_state:
    st.session_state["last_question"] = st.session_state.pop("inject_template")


user_input = st.text_area(
    "Ask your SQL question:",
    value=st.session_state.get("last_question", ""),
    placeholder="e.g., Show me the total revenue by category for last month."
)

# --- Prompt Templates Section ---
if "collapse_prompt_templates" not in st.session_state:
    st.session_state.collapse_prompt_templates = True

with st.expander("💡 Prompt Templates", expanded=not st.session_state.get("collapse_prompt_templates", True)):
    st.markdown("Click an example to insert it into your question:")

    prompt_templates = [
        "Top 10 customers by revenue",
        "Total sales by region for the last year",
        "List all orders over $500",
        "Average order value per customer",
        "Monthly revenue trend",
    ]

    for i, example in enumerate(prompt_templates):
        if st.button(example, key=f"template_{i}"):
            st.session_state["inject_template"] = example
            st.session_state.collapse_prompt_templates = True
            st.rerun()


# with st.expander("💡 Prompt Templates", expanded=False):
#     st.markdown("Click an example to insert it into your question:")

#     prompt_templates = [
#         "Top 10 customers by revenue",
#         "Total sales by region for the last year",
#         "List all orders over $500",
#         "Average order value per customer",
#         "Monthly revenue trend",
#     ]

#     for i, example in enumerate(prompt_templates):
#         if st.button(example, key=f"template_{i}"):
#             st.session_state["inject_template"] = example
#             st.rerun()


st.markdown("### 🧩 Schema Help")
with st.expander("How should I format my schema?"):
    st.markdown("""
**Schema format tips:**
- Use one table per line
- Each table should follow: `table_name(column1, column2, ...)`
- Separate multiple columns with commas

**Example:**
orders(id, customer_id, order_date, total_amount)
customers(id, name, region, email)
""")

st.sidebar.markdown(f"🔍 last_question: `{st.session_state.get('last_question', '')}`")

# 📂 Schema Import (multiple files)
uploaded_files = st.file_uploader(
    "Upload schema files (CSV, JSON schema, or DDL)",
    type=["csv", "json", "sql"],
    accept_multiple_files=True
)

if uploaded_files:
    lines = []
    for uploaded_file in uploaded_files:
        ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
        table_name = uploaded_file.name.rsplit(".", 1)[0]
        
        if ext == "csv":
            import pandas as _pd
            df = _pd.read_csv(uploaded_file)
            cols = df.columns.tolist()
            lines.append(f"{table_name}({', '.join(cols)})")
        
        elif ext == "json":
            import json as _json
            schema = _json.load(uploaded_file)
            if "properties" in schema:
                cols = list(schema["properties"].keys())
                lines.append(f"{table_name}({', '.join(cols)})")
            else:
                st.error(f"JSON schema `{uploaded_file.name}` is missing a top-level 'properties' key.")
        
        elif ext in ("sql", "ddl"):
            import re as _re
            text = uploaded_file.read().decode("utf-8")
            for tbl, cols in _re.findall(r"CREATE\s+TABLE\s+(\w+)\s*\(([^;]+?)\);", text, _re.IGNORECASE | _re.DOTALL):
                col_names = [c.strip().split()[0] for c in cols.split(",")]
                lines.append(f"{tbl}({', '.join(col_names)})")
        
        else:
            st.warning(f"Skipping unsupported file type: {uploaded_file.name}")
    
    if lines:
        # Join all table definitions with newlines
        schema_input = "\n".join(lines)
        st.success("Schemas imported from all uploaded files!")
        st.session_state.last_schema = schema_input

# Fallback to manual paste if no upload
schema_input = schema_input if uploaded_files else st.text_area(
    "Paste your table schema (optional, improves accuracy):",
    height=150,
    placeholder=(
        "e.g.,\n"
        "orders(id, customer_id, order_date, total_amount)\n"
        "products(id, name, category, price)"
    ),
    value=st.session_state.last_schema,
    key="manual_schema_input"
)

st.session_state.last_schema = schema_input

# Save/load schema UI
os.makedirs("schemas", exist_ok=True)
saved_files = [f for f in os.listdir("schemas") if f.endswith(".txt")]

col1, col2 = st.columns([3, 1])
with col1:
    selected_schema_file = st.selectbox("📂 Load saved schema", ["-- Select --"] + saved_files)
    if selected_schema_file != "-- Select --":
        with open(os.path.join("schemas", selected_schema_file), "r") as f:
            schema_input = f.read()
            st.session_state.last_schema = schema_input

with col2:
    if st.button("💾 Save Schema"):
        filename = f"schema_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(os.path.join("schemas", filename), "w") as f:
            f.write(schema_input)
        st.success(f"Schema saved as {filename}")
        st.rerun()


if schema_input.strip():
    st.markdown("### 🧮 Schema Preview")
    parsed_schema = parse_schema(schema_input)
    mock_tables = generate_mock_dataframe_from_schema(schema_input)

        # 🔍 Live Schema Inspector
    with st.expander("🔎 Live Schema Inspector"):
        if mock_tables:
            for table_name, df in mock_tables.items():
                st.markdown(f"### 📦 {table_name}")
                st.markdown(f"**Columns:** {', '.join(df.columns)}")
                st.dataframe(df.head(), use_container_width=True)

        else:
            st.info("No schema detected.")


    relationships = infer_relationships(schema_input)
    st.sidebar.write("DEBUG: inferred joins", relationships)
    if relationships:
        st.markdown("### 🔗 Inferred Join Suggestions")
        for rel in relationships:
            st.markdown(f"- `{rel}`")

    if parsed_schema:
        for table, columns in parsed_schema.items():
            st.markdown(f"**📦 {table}**")
            for col in columns:
                st.markdown(f"- `{col}`")
    else:
        st.warning("⚠️ Could not parse schema. Please use the format: `table_name(column1, column2, ...)`")


    all_columns = []
    for table, columns in parsed_schema.items():
        for col in columns:
            all_columns.append(f"{table}.{col}")

    # if all_columns:
    #     with st.expander("🧠 Schema Hints"):
    #         st.markdown("Try referencing:")
    #         st.code(", ".join(all_columns), language="sql")


    #     st.markdown("Click to insert:")

    #     for ref in all_columns:
    #         if st.button(ref, key=f"insert_{ref}"):
    #             st.session_state["inject_ref"] = ref
    #             st.rerun()


missing_columns = []

if st.button("🌟Generate SQL") and user_input:
    with st.spinner("Thinking..."):
        try:
            if schema_input.strip() and not re.search(r"\w+\s*\(.*?\)", schema_input):
                st.warning("⚠️ Schema format looks invalid. Please use format like:\norders(id, date, total_amount)")
                raise st.stop()

            schema_prompt = f"Schema:\n{schema_input}\n\n" if schema_input.strip() else "Assume reasonable table and column names.\n\n"
            relationships = infer_relationships(schema_input)
            if relationships:
                schema_prompt += "Inferred Relationships:\n" + "\n".join(relationships) + "\n\n"

            if st.session_state.follow_up_input and st.session_state.last_question:
                user_prompt = (
                    f"Original request: {st.session_state.last_question}\n"
                    f"Follow-up instruction: {st.session_state.follow_up_input}\n\n"
                    f"Note: If breaking down by month, use a date column like 'order_date' from the schema. "
                    f"Only use fields that exist in the schema provided."
                )

            else:
                user_prompt = f"{schema_prompt}\n\nTranslate the following into SQL:\n{user_input}"

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates SQL queries. "
                               "If a schema is provided, use only the tables and columns it contains. "
                               "If not, make educated assumptions based on the user's question."
                },
                {"role": "user", "content": user_prompt}
            ]

            model = st.session_state["model"]
            chat_args = {
                "model": model,
                "messages": messages,
            }

            # Only set temperature if supported
            if model != "o4-mini":
                chat_args["temperature"] = 0.0  # Lower temperature for deterministic output

            response = client.chat.completions.create(**chat_args)


            if response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
            else:
                prompt_tokens = completion_tokens = total_tokens = 0

            # Prices per 1M tokens
            model_prices = {
                "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
                "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
                "gpt-4.1": {"prompt": 0.0020, "completion": 0.0080},
                "o4-mini": {"prompt": 0.0011, "completion": 0.0044},
            }

            model = st.session_state.get("model", "gpt-3.5-turbo")
            pricing = model_prices.get(model, {"prompt": 0.0, "completion": 0.0})

            cost_estimate = (
                prompt_tokens * pricing["prompt"] / 1000 +
                completion_tokens * pricing["completion"] / 1000
            )

            st.caption(
                f"**Tokens used:** {total_tokens:,} "
                f"(Prompt: {prompt_tokens:,}, Completion: {completion_tokens:,}) "
                f"| **Estimated cost:** ${cost_estimate:.5f}"
            )


            raw_response = response.choices[0].message.content.strip()
            sql_match = re.search(r"```sql\s*(.*?)```", raw_response, re.DOTALL | re.IGNORECASE)
            st.session_state.sql_output = sql_match.group(1).strip() if sql_match else raw_response

            st.session_state.last_question = user_input
            st.session_state.last_schema = schema_input

            mock_tables = generate_mock_dataframe_from_schema(schema_input)

            all_columns = {tbl.lower(): set(map(str.lower, df.columns)) for tbl, df in mock_tables.items()}
            column_refs = re.findall(r'\b([a-z_][a-z0-9_]*)\.(\w+)', st.session_state.sql_output, re.IGNORECASE)

            missing_columns = [
                f"{table}.{col}" for table, col in column_refs
                if table.lower() in all_columns and col.lower() not in all_columns[table.lower()]
            ]

            if missing_columns:
                st.warning("⚠️ Some parts of the generated SQL reference unknown fields:\n" + "\n".join(missing_columns))

            with open("query_log.csv", mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if os.stat("query_log.csv").st_size == 0:
                    writer.writerow(["timestamp", "question", "schema", "sql_output"])
                writer.writerow([datetime.now().isoformat(), user_input, schema_input, st.session_state.sql_output])

            st.session_state.conversation_history.append({
                "question": user_input,
                "follow_up": st.session_state.follow_up_input,
                "sql": st.session_state.sql_output,
                "favorite": False
            })


            st.session_state.follow_up_input = ""

        except Exception as e:
            st.error(f"Error: {e}")

if st.session_state.sql_output:
    tabs = st.tabs(["🧾 SQL", "🔍 Preview"])
    parsed_schema = parse_schema(st.session_state.last_schema or "")

    with tabs[0]:
        sql = st.session_state.sql_output
        parsed_schema = parse_schema(st.session_state.last_schema or "")

        if parsed_schema:
            # ✅ Add your table/column validation + highlighting here
            valid_tables = {tbl.lower() for tbl in parsed_schema}

            used_tables = set(
                re.findall(r'\bFROM\s+(\w+)', sql, re.IGNORECASE) +
                re.findall(r'\bJOIN\s+(\w+)', sql, re.IGNORECASE)
            )
            missing_tables = [t for t in used_tables if t.lower() not in valid_tables]
            if missing_tables:
                st.warning("⚠️ Unknown tables referenced: " + ", ".join(missing_tables))

            all_columns = {
                tbl.lower(): set(col.lower() for col in cols)
                for tbl, cols in parsed_schema.items()
            }
            column_refs = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\.(\w+)', sql)
            missing_columns = [
                f"{tbl}.{col}"
                for tbl, col in column_refs
                if tbl.lower() in all_columns and col.lower() not in all_columns[tbl.lower()]
            ]
            if missing_columns:
                st.warning("⚠️ Unknown fields referenced: " + ", ".join(missing_columns))

            highlighted = sql
            for tbl in missing_tables:
                highlighted = re.sub(
                    rf"\b{re.escape(tbl)}\b",
                    f"<span style='background-color:rgba(255,0,0,0.2);'>{tbl}</span>",
                    highlighted,
                    flags=re.IGNORECASE
                )
            for ref in missing_columns:
                highlighted = re.sub(
                    rf"\b{re.escape(ref)}\b",
                    f"<span style='background-color:rgba(255,0,0,0.2);'>{ref}</span>",
                    highlighted
                )

            # Also highlight bare column names used with aliases
            for col in re.findall(r'\b\w+\.(\w+)\b', sql):
                if any(col.lower() in mc.lower() for mc in missing_columns):
                    highlighted = re.sub(
                        rf"\b{re.escape(col)}\b",
                        f"<span style='background-color:rgba(255,0,0,0.2);'>{col}</span>",
                        highlighted
                    )
            st.markdown(f"<pre><code>{highlighted}</code></pre>", unsafe_allow_html=True)

        else:
            # only when there’s no schema, show the raw SQL
            st.code(sql, language="sql")
            st.info("No schema provided — showing generated SQL without validation.")


    with tabs[1]:
        try:
            mock_tables = generate_mock_dataframe_from_schema(schema_input)
            con = duckdb.connect()
            for table_name, df in mock_tables.items():
                con.register(table_name, df)

            sql_for_duckdb = re.sub(r"(?i)GETDATE\(\)|CURDATE\(\)", "current_date", sql)
            sql_for_duckdb = re.sub(
                r"(?i)DATEADD\s*\(\s*(\w+)\s*,\s*(-?\d+)\s*,\s*([\w\.]+)\s*\)",
                lambda m: f"{m.group(3)} + INTERVAL '{m.group(2)}' {m.group(1)}",
                sql_for_duckdb
            )
            # Handle DATE_SUB(col, INTERVAL N unit)
            sql_for_duckdb = re.sub(
                r"(?i)DATE_SUB\s*\(\s*([\w\.]+)\s*,\s*INTERVAL\s+(\d+)\s+(\w+)\s*\)",
                lambda m: f"{m.group(1)} - INTERVAL '{m.group(2)}' {m.group(3).upper()}",
                sql_for_duckdb
            )

            preview_df = con.execute(sql_for_duckdb).fetchdf()
            st.dataframe(preview_df, use_container_width=True)

            csv_buffer = StringIO()
            preview_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "📥 Download Preview as CSV",
                csv_buffer.getvalue(),
                file_name="preview_results.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"⚠️ Could not preview results: {e}")


# Reset follow-up input if the flag is set
if "reset_follow_up_input" not in st.session_state:
    st.session_state.reset_follow_up_input = False

if st.session_state.reset_follow_up_input:
    st.session_state.follow_up_input = ""
    st.session_state.reset_follow_up_input = False


# Inject selected follow-up before widget renders
if "inject_followup" in st.session_state:
    st.session_state["follow_up_input"] = st.session_state.pop("inject_followup")


follow_up_input = st.text_input(
    "🔁 Follow-up instruction (optional):",
    value=st.session_state.get("follow_up_input", ""),
    key="manual_followup_input",
    placeholder="e.g., Now break that down by month"
)



followup_suggestions = [
    "",  # default empty
    "Now break it down by month",
    "Add region to the breakdown",
    "Filter for last 12 months only",
    "Sort by descending revenue",
    "Exclude customers with zero orders",
]

selected_suggestion = st.selectbox(
    "💡 Suggested follow-ups:",
    followup_suggestions,
    index=0,
    key="followup_suggestion"
)

if selected_suggestion and selected_suggestion != st.session_state.get("follow_up_input", ""):
    st.session_state["inject_followup"] = selected_suggestion
    st.rerun()

st.caption(f"Refining:\n➡️ {st.session_state.last_question}\n↪️ {st.session_state.follow_up_input}")

if st.button("🔁 Apply Follow-Up") and st.session_state.last_question and st.session_state.follow_up_input:
    with st.spinner("Refining SQL..."):
        try:
            schema_prompt = (
                f"Schema:\n{st.session_state.last_schema}\n\n"
                if st.session_state.last_schema.strip()
                else "Assume reasonable table and column names.\n\n"
            )
            relationships = infer_relationships(st.session_state.last_schema)
            if relationships:
                schema_prompt += "Inferred Relationships:\n" + "\n".join(relationships) + "\n\n"

            user_prompt = (
                "You previously generated the following SQL query:\n\n"
                f"{st.session_state.sql_output}\n\n"
                "The original request was:\n"
                f"{st.session_state.last_question}\n\n"
                "The user now provided this follow-up instruction:\n"
                f"{st.session_state.follow_up_input}\n\n"
                "Please revise the original SQL to reflect the follow-up. Return only the updated SQL in a ```sql block.\n"
                "Use only columns that exist in the provided schema. If grouping by month, use EXTRACT(MONTH FROM order_date)."
            )



            messages = [
                {"role": "system", "content": "You are a helpful assistant that generates SQL queries. "
                                              "If a schema is provided, use only the tables and columns it contains. "
                                              "If not, make educated assumptions based on the user's question."},
                {"role": "user", "content": schema_prompt + user_prompt}
            ]

            model = st.session_state["model"]
            chat_args = {
                "model": model,
                "messages": messages,
            }

            # Only set temperature if supported
            if model != "o4-mini":
                chat_args["temperature"] = 0.0  # Lower temperature for deterministic output

            response = client.chat.completions.create(**chat_args)


            if response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
            else:
                prompt_tokens = completion_tokens = total_tokens = 0

            # Prices per 1M tokens
            model_prices = {
                "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
                "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
                "gpt-4.1": {"prompt": 0.0020, "completion": 0.0080},
                "o4-mini": {"prompt": 0.0011, "completion": 0.0044},
            }

            model = st.session_state.get("model", "gpt-3.5-turbo")
            pricing = model_prices.get(model, {"prompt": 0.0, "completion": 0.0})

            cost_estimate = (
                prompt_tokens * pricing["prompt"] / 1000 +
                completion_tokens * pricing["completion"] / 1000
            )

            st.caption(
                f"**Tokens used:** {total_tokens:,} "
                f"(Prompt: {prompt_tokens:,}, Completion: {completion_tokens:,}) "
                f"| **Estimated cost:** ${cost_estimate:.5f}"
            )


            raw_response = response.choices[0].message.content.strip()
            sql_match = re.search(r"```sql\s*(.*?)```", raw_response, re.DOTALL | re.IGNORECASE)
            refined_sql = sql_match.group(1).strip() if sql_match else raw_response

            # ✅ This is critical: update the main output
            st.session_state.sql_output = refined_sql

            # ✅ Log the refined conversation
            st.session_state.conversation_history.append({
                "question": st.session_state.last_question,
                "follow_up": st.session_state.follow_up_input,
                "sql": refined_sql
            })

            st.session_state.follow_up_input = ""

        except Exception as e:
            st.error(f"Refinement Error: {e}")

if st.button("❌ Clear Follow-Up"):
    if "follow_up_input" in st.session_state:
        del st.session_state["follow_up_input"]
    st.rerun()


if os.path.exists("query_log.csv"):
    with st.expander("📜 View Query History"):
        with open("query_log.csv", "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
            if len(rows) > 1:
                header, *data = rows
                st.dataframe(
                    data[-st.slider("Show last N queries", 1, min(len(data), 50), 5):],
                    use_container_width=True
                )

        with open("query_log.csv", "r", encoding="utf-8") as f:
            st.download_button("📤 Download Query Log", f.read(), file_name="query_log.csv", mime="text/csv")

        if st.button("🗑️ Clear Query Log"):
            os.remove("query_log.csv")
            st.success("Query log cleared.")
            st.rerun()

# Build tag filter options
all_tags = sorted({
    tag
    for convo in st.session_state.conversation_history
    for tag in convo.get("tags", [])
})
filter_tag = st.selectbox("🔍 Filter by tag", ["-- All --"] + all_tags, key="filter_tag")

# Apply tag filter
filtered_history = (
    st.session_state.conversation_history
    if filter_tag == "-- All --"
    else [c for c in st.session_state.conversation_history if filter_tag in c.get("tags", [])]
)


with st.expander("🧠 Conversation History"):
    show_only_favorites = st.checkbox("🔎 Show only favorites", value=False)
    history_to_show = [
        (i, convo)
        for i, convo in enumerate(filtered_history[::-1])
        if not show_only_favorites or convo.get("favorite", False)
    ]

    for i, convo in history_to_show:
        step_num = len(st.session_state.conversation_history) - i
        st.markdown(f"---\n**Step {step_num}**")
        st.markdown(f"**Q:** {convo['question']}")
        if convo.get("follow_up"):
            st.markdown(f"**↪️ Follow-up:** {convo['follow_up']}")
        st.code(convo["sql"], language="sql", line_numbers=True)

        # ⭐ Favorite toggle
        fav_key = f"fav_{i}"
        is_fav = convo.get("favorite", False)
        if st.checkbox("⭐ Favorite", value=is_fav, key=fav_key):
            st.session_state.conversation_history[-(i + 1)]["favorite"] = True
        else:
            st.session_state.conversation_history[-(i + 1)]["favorite"] = False

        # 🏷️ Tag input
        tag_key = f"tags_{i}"
        tag_input = st.text_input("🏷️ Tags (comma-separated)", value=", ".join(convo.get("tags", [])), key=tag_key)
        tags = [t.strip() for t in tag_input.split(",") if t.strip()]
        st.session_state.conversation_history[-(i + 1)]["tags"] = tags

        # ✏️ Edit & Re-ask
        if st.button("✏️ Edit & Re-ask", key=f"reask_{i}"):
            st.session_state["edit_mode_question"] = convo["question"]
            st.session_state["edit_mode_followup"] = convo.get("follow_up", "")
            st.session_state["edit_mode_schema"] = st.session_state.get("last_schema", "")
            st.session_state["trigger_edit_mode"] = True
            st.rerun()




if st.button("🔄 Clear All"):
    for key in ["sql_output", "last_question", "last_schema", "follow_up_input", "conversation_history"]:
        st.session_state.pop(key, None)
    st.rerun()
