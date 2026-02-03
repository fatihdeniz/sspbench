import os
import re
import time
import json
import requests
from functools import wraps

# Dictionary maps the airflow task to the corresponding error message
# that should be shown to the user
TASK_MESSAGES = {
    "create-update-report": "Failed to generate the examination report.",
    "evaluate": "Failed to evaluate the model's responses.",
    "load": "Failed to load the model.",
    "query": "Failed to query the model.",
    "store": "Failed to store the examination result.",
    "summarize-model": "Failed to generate model summary."
}

def retry_on_exception(max_retries=3, delay=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_retries:
                try:
                    return func(*args, **kwargs)
                except requests.RequestException as e:
                    attempts += 1
                    if attempts >= max_retries:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator

def build_log_path(context):
    """Best-effort construction of the log file path for the failed task.

    Airflow default pattern:
    <AIRFLOW_HOME>/logs/dag_id=<dag_id>/run_id=<run_id>/task_id=<task_id>/attempt=<n>.log
    """
    try:
        airflow_home = os.getenv("AIRFLOW_HOME", "/opt/airflow").rstrip("/")
        dag_id = context.get("dag").dag_id if context.get("dag") else context.get("dag_run").dag_id
        run_id = context.get("dag_run").run_id if context.get("dag_run") else context.get("run_id")
        task_id = context.get("ti").task_id
        base_dir = f"{airflow_home}/logs/dag_id={dag_id}/run_id={run_id}/task_id={task_id}"
        if not os.path.isdir(base_dir):
            return None

        # Find highest attempt file
        attempts = []
        for name in os.listdir(base_dir):
            m = re.match(r"attempt=(\d+)\.log$", name)
            if m:
                attempts.append(int(m.group(1)))
        if not attempts:
            return None
        latest = max(attempts)
        return os.path.join(base_dir, f"attempt={latest}.log")
    except Exception:
        return None

def fetch_log(context, max_lines=200):
    """Return (log_path, log_tail) using filesystem logs."""
    log_path = build_log_path(context)
    if not log_path or not os.path.isfile(log_path):
        return None

    try:
        with open(log_path, "rb") as f:
            f.seek(0)
            content = f.read().decode(errors="replace")
            content = "\n".join(content.splitlines()[-max_lines:])
    except Exception:
        return None

    return content.lower()

def classify_failure(task_id, log):
    """Return (category, reason, user_message) based on heuristics.

    category: 'user' | 'server'
    reason: short machine-friendly code
    user_message: concise, human-facing text
    """
    print(f"Classifying failure for task {task_id} with log:\n{log}\n--- End of log ---\n")

    # Known tasks that indicate internal errors
    if any(sub in task_id for sub in [
        "check-model", 
        "download-model", 
        "use-cached-model",
        "resolve-model-path",
        "summarize-model",
        "store-",
        "create-update-report",
        "evaluate"
    ]):
        return (
            "server",
            "infrastructure-error",
            "We encountered an internal error while running your test. Our team has been notified."
        )
    else:
        # Empty response from model
        if "models.llm_base.emptymodelresponseerror" in log:
            percentage = None
            pattern = r"models\.llm_base\.emptymodelresponseerror: model returned \d+ empty responses out of \d+ prompts \(([^)]*)\)\."
            m = re.search(pattern, log)
            if m:
                raw_pct = m.group(1)
                numeric = re.sub(r"%$", "", raw_pct)
                try:
                    percentage = f"{round(float(numeric))}%"
                except ValueError:
                    percentage = raw_pct
            return (
                "user",
                "empty-response",
                (
                    "Model returned too many empty responses. Please check the model configuration."
                    if percentage is None
                    else f"{percentage} of model responses are empty. Please check the model configuration."
                ),
            )

        if "openai.ratelimiterror: error code: 429" in log:
            return (
                "user",
                "rate-limit-exceeded",
                "OpenAI API rate limit has been exceeded. Please try again later."
            )

        if "-local" in task_id:
            # Untrusted code required
            TRUST_REQUIRED_PAT = re.compile(
                r"""(?ixs)
                (?:                                          
                    \bplease\s+pass\s+the\s+argument\s+[`'"]?trust_remote_code\s*=\s*true[`'"]?\b
                    |
                    \b(?:the\s+)?repository\b.*?\bcontains\s+custom\s+code\b.*?\bmust\s+be\s+executed\b.*?\b(correctly\s+)?load\s+the\s+model\b
                )
                """,
            )
            if TRUST_REQUIRED_PAT.search(log):
                return (
                    "user",
                    "untrusted-code-required",
                    "Querying the model requires executing untrusted code, which is not supported."
                )

            # CUDA out of memory
            if "torch.outofmemoryerror: cuda out of memory. tried to allocate" in log:
                return (
                    "server",
                    "insufficient-gpu-memory",
                    "The model requires more GPU memory than available. Please use a smaller model."
                )
        elif "-api" in task_id:
            # Generic API errors
            return (
                "user",
                "api-configuration",
                "There was an issue with the model API configuration. Please verify the API settings."
            )

    # Generic fallback
    return (
        "server",
        "infrastructure-error",
        "We encountered an internal error while running your test. Our team has been notified."
    )

@retry_on_exception(max_retries=3, delay=2)
def update_summary(aix_url, auth_key, model_id, summary):
    headers = {
        "Authorization": f"Bearer {auth_key}",
        "Content-Type": "application/json"
    }

    url = f"{aix_url}/models/{model_id}"
    response = requests.patch(url, headers=headers, json=summary)
    
    if response.status_code in [200, 201]:
        data = json.loads(response.content)
        return data
    else:
        response.raise_for_status()

@retry_on_exception(max_retries=3, delay=2)
def store_results(aix_url, auth_key, examination_id, results, duration):
    payload = {
        "examination": examination_id,
        "results": results,
        "duration": duration
    }
    headers = {
        "Authorization": f"Bearer {auth_key}",
        "Content-Type": "application/json"
    }

    url = f"{aix_url}/results"
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code in [200, 201]:
        return response
    else:
        response.raise_for_status()

@retry_on_exception(max_retries=3, delay=2)
def get_report(aix_url, auth_key, report_id):
    headers = {
        "Authorization": f"Bearer {auth_key}",
        "Content-Type": "application/json"
    }

    url = f"{aix_url}/reports/admin/{report_id}"
    response = requests.get(url, headers=headers)
    status_code = response.status_code
    
    if status_code in [200, 201]:
        data = json.loads(response.content)
        return data
    elif status_code == 404:
        return None
    else:
        response.raise_for_status()

@retry_on_exception(max_retries=3, delay=2)
def update_report(aix_url, auth_key, report_id, payload):
    headers = {
        "Authorization": f"Bearer {auth_key}",
        "Content-Type": "application/json"
    }

    url = f"{aix_url}/reports/{report_id}"
    response = requests.patch(url, headers=headers, json=payload)

    if response.status_code in [200, 201]:
        data = json.loads(response.content)
        return data
    else:
        response.raise_for_status()

@retry_on_exception(max_retries=3, delay=2)
def update_status(aix_url, auth_key, examination_id, status, log_message=None):
    payload = {
        "status": status
    }
    if log_message:
        payload["logMessage"] = log_message

    headers = {
        "Authorization": f"Bearer {auth_key}",
        "Content-Type": "application/json"
    }

    url = f"{aix_url}/examinations/{examination_id}"
    response = requests.patch(url, headers=headers, json=payload)

    if response.status_code in [200, 201]:
        return response
    else:
        response.raise_for_status()

@retry_on_exception(max_retries=3, delay=2)
def update_progress(context, log_message=None):
    aix_url = context["params"]["aix_url"]
    auth_key = context["params"]["aix_token"]
    examination_id = context["params"]["examination_id"]

    total_tasks = len(context["dag"].tasks)
    task_list = context["dag_run"].get_task_instances()
    completed_ids = {
        task.task_id
        for task in task_list
        if task.state in {"success", "skipped"}
    }
    current_task = context["ti"]
    if current_task.state in {"success", "skipped"}:
        completed_ids.add(current_task.task_id)

    completed_tasks  = len(completed_ids)
    progress = int((completed_tasks / total_tasks) * 100)

    payload = {
        "progress": progress
    }
    if log_message:
        payload["logMessage"] = log_message

    headers = {
        "Authorization": f"Bearer {auth_key}",
        "Content-Type": "application/json"
    }

    url = f"{aix_url}/examinations/{examination_id}"
    response = requests.patch(url, headers=headers, json=payload)

    if response.status_code in [200, 201]:
        return response
    else:
        response.raise_for_status()
        
@retry_on_exception(max_retries=3, delay=2)
def generate_report_files(aix_url, auth_key, report_id):
    headers = {
        "Authorization": f"Bearer {auth_key}",
        "Content-Type": "application/json"
    }
    base = aix_url.rstrip("/")

    pdf_url = f"{base}/reports/{report_id}/generate/pdf"
    pdf_response = requests.post(pdf_url, headers=headers)
    if pdf_response.status_code not in [200, 201]:
        pdf_response.raise_for_status()

    return True

def update_failure(context):
    aix_token = context["params"]["aix_token"]
    aix_url = context["params"]["aix_url"]
    examination_id = context["params"]["examination_id"]
    task_id = context["ti"].task_id

    log_tail = fetch_log(context, max_lines=400)
    category, reason, user_message = classify_failure(task_id, log_tail)

    message = {
        "level": "error",
        "value": user_message,
        "rawData": (
            f"category={category}; reason={reason}; "
            f"task={task_id}; log_tail=\n{log_tail if log_tail else 'N/A'}"
        ),
    }
    response = update_status(aix_url, aix_token, examination_id, status="failed", log_message=message)
    
    return response
