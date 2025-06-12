#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/07/18
@Author  : Your Name / AI Agent
@File    : metagpt_streamlit_ui.py
@Desc    : Streamlit frontend prototype for MetaGPT.
"""

import streamlit as st
import requests
import time

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"  # FastAPI backend URL

# --- Helper Functions ---

def create_project_on_backend(idea: str, project_name: Optional[str] = None) -> Optional[dict]:
    """Sends a request to the backend to create a new project."""
    payload = {"idea": idea}
    if project_name:
        payload["project_name"] = project_name

    try:
        response = requests.post(f"{API_BASE_URL}/projects/", json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error creating project: {e}")
        return None

def get_project_status_from_backend(project_id: str) -> Optional[dict]:
    """Fetches the status of a project from the backend."""
    try:
        response = requests.get(f"{API_BASE_URL}/projects/{project_id}/")
        response.raise_for_status() # Raise an HTTPError for bad responses (4XX or 5XX)
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            st.warning(f"Project '{project_id}' not found on backend during status check.")
        else:
            st.error(f"HTTP error fetching project status: {e}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Request error fetching project status: {e}")
        return None

def get_project_messages_from_backend(project_id: str) -> Optional[List[Dict]]:
    """Fetches messages for a project from the backend."""
    try:
        response = requests.get(f"{API_BASE_URL}/projects/{project_id}/messages/")
        response.raise_for_status()
        return response.json().get("messages", [])
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404: # Should not happen if project exists, but good practice
            st.warning(f"Messages not found for project '{project_id}'.")
        else:
            st.error(f"HTTP error fetching project messages: {e}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Request error fetching project messages: {e}")
        return None

def get_project_artifacts_from_backend(project_id: str) -> Optional[List[Dict]]:
    """Fetches artifacts for a project from the backend."""
    try:
        response = requests.get(f"{API_BASE_URL}/projects/{project_id}/artifacts/")
        response.raise_for_status()
        return response.json().get("artifacts", [])
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404: # Project itself or its artifacts path might be missing
            st.warning(f"Artifacts not found for project '{project_id}'.")
        else:
            st.error(f"HTTP error fetching project artifacts: {e}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Request error fetching project artifacts: {e}")
        return None

def get_project_logs_from_backend(project_id: str) -> Optional[List[str]]:
    """Fetches logs for a project from the backend."""
    try:
        response = requests.get(f"{API_BASE_URL}/projects/{project_id}/logs/")
        response.raise_for_status()
        return response.json().get("logs", [])
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            st.warning(f"Logs not found for project '{project_id}'.")
        else:
            st.error(f"HTTP error fetching project logs: {e}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Request error fetching project logs: {e}")
        return None

# --- Streamlit UI ---

st.set_page_config(layout="wide")
st.title("MetaGPT Frontend Prototype")

# Initialize session state variables if they don't exist
if "project_id" not in st.session_state:
    st.session_state.project_id = None
if "project_name" not in st.session_state: # To store project name from status
    st.session_state.project_name = "N/A"
if "project_status_data" not in st.session_state:
    st.session_state.project_status_data = None
if "project_messages_data" not in st.session_state:
    st.session_state.project_messages_data = []
if "project_artifacts_data" not in st.session_state:
    st.session_state.project_artifacts_data = []
if "project_logs_data" not in st.session_state:
    st.session_state.project_logs_data = []
if "last_status_check" not in st.session_state:
    st.session_state.last_status_check = 0.0 # Ensure it's a float

# --- Project Creation Section ---
st.header("1. Create New Project")

# Optional project name input
# project_name_input = st.text_input("Optional: Enter a name for your project (e.g., 'my_snake_game')")
idea_input = st.text_area("Enter your project idea:", height=150, placeholder="e.g., A CLI snake game in Python")

if st.button("Create Project", type="primary"):
    if idea_input:
        with st.spinner("Initiating project creation..."):
            # project_name = project_name_input if project_name_input else None
            # For now, let backend decide project name if not provided or use a fixed one for simplicity.
            # The API currently expects 'project_name' to be optional.
            creation_response = create_project_on_backend(idea_input)
            if creation_response and "project_id" in creation_response:
                st.session_state.project_id = creation_response["project_id"]
                st.session_state.project_status_data = creation_response
                st.session_state.project_name = creation_response.get("project_name", st.session_state.project_id) # Use project_id if name not in response
                # Clear data from previous projects
                st.session_state.project_messages_data = []
                st.session_state.project_artifacts_data = []
                st.session_state.project_logs_data = []
                st.success(f"Project creation initiated! Project ID: {st.session_state.project_id}")
                st.info(f"Initial status: {creation_response.get('status', 'pending')}")
                st.session_state.last_status_check = time.time()
                st.experimental_rerun()
            else:
                st.error("Failed to get project ID from backend or invalid response.")
    else:
        st.warning("Please enter a project idea.")

# --- Project Status and Details Section ---
if st.session_state.project_id:
    st.markdown("---")
    st.header(f"2. Project Status: {st.session_state.project_id}")

    # Auto-refresh mechanism (simple version)
    # More sophisticated auto-refresh can be done with st.empty() and a loop,
    # but st.experimental_rerun is simpler for now if periodic updates are desired.
    # However, for true polling, a frontend mechanism or websockets would be better.
    # Here, we'll offer a manual refresh button and refresh if enough time has passed,
    # or if the project status is still active.

    REFRESH_INTERVAL = 5  # seconds
    force_refresh_data = False # Flag to force refresh of messages/logs/artifacts

    col_status_1, col_status_2 = st.columns([3,1])
    with col_status_2:
        if st.button("🔄 Refresh All Data"):
            force_refresh_data = True
            with st.spinner("Refreshing all project data..."):
                status_data = get_project_status_from_backend(st.session_state.project_id)
                if status_data:
                    st.session_state.project_status_data = status_data
                    st.session_state.project_name = status_data.get("project_name", st.session_state.project_id)

                # Force refresh messages, logs, and artifacts as well
                st.session_state.project_messages_data = get_project_messages_from_backend(st.session_state.project_id) or []
                st.session_state.project_logs_data = get_project_logs_from_backend(st.session_state.project_id) or []
                if st.session_state.project_status_data and st.session_state.project_status_data.get("status") == "completed":
                    st.session_state.project_artifacts_data = get_project_artifacts_from_backend(st.session_state.project_id) or []

                st.session_state.last_status_check = time.time()
                st.experimental_rerun() # Rerun to reflect all new data

    # Auto-refresh status if project is active
    current_time = time.time()
    is_active_project = st.session_state.project_status_data and \
                        st.session_state.project_status_data.get("status") not in ["completed", "failed"]

    if is_active_project and (current_time - st.session_state.last_status_check > REFRESH_INTERVAL):
        with st.spinner(f"Auto-refreshing status for {st.session_state.project_id}..."):
            status_data = get_project_status_from_backend(st.session_state.project_id)
            if status_data:
                st.session_state.project_status_data = status_data
                st.session_state.project_name = status_data.get("project_name", st.session_state.project_id)
            st.session_state.last_status_check = current_time
            force_refresh_data = True # Trigger data refresh for active project
            st.experimental_rerun()


    # Display current status details
    if st.session_state.project_status_data:
        status_info = st.session_state.project_status_data
        with col_status_1: # Display status in the first column
            st.subheader(f"Status for: {st.session_state.project_name}")
            cols = st.columns(3)
            cols[0].metric("Status", status_info.get("status", "Unknown"))
            created_at_str = status_info.get("created_at")
            updated_at_str = status_info.get("updated_at")
            cols[1].metric("Created At", datetime.fromisoformat(created_at_str).strftime('%Y-%m-%d %H:%M:%S') if created_at_str else "N/A")
            cols[2].metric("Last Updated", datetime.fromisoformat(updated_at_str).strftime('%Y-%m-%d %H:%M:%S') if updated_at_str else "N/A")

            error_message = status_info.get("error_message")
            if error_message:
                st.error(f"Error: {error_message}")
    else: # Initial load or if status fetch failed
        st.info("Click 'Refresh All Data' to fetch project details.")


    # --- Display Messages, Artifacts, Logs ---
    st.markdown("---")

    tab_messages, tab_artifacts, tab_logs = st.tabs(["Messages", "Artifacts", "Logs"])

    with tab_messages:
        st.subheader("Agent Messages")
        if force_refresh_data and is_active_project and not st.session_state.project_messages_data: # Initial load for active project
             with st.spinner("Loading messages..."):
                st.session_state.project_messages_data = get_project_messages_from_backend(st.session_state.project_id) or []

        if st.session_state.project_messages_data:
            for msg in st.session_state.project_messages_data:
                with st.chat_message(name=msg.get("role", "unknown")):
                    st.write(f"**{msg.get('role', 'Unknown')}** ({msg.get('timestamp')})")
                    st.markdown(msg.get("content", ""))
        elif not is_active_project and not st.session_state.project_messages_data : # If not active and no data, try fetching once
            with st.spinner("Loading messages..."):
                st.session_state.project_messages_data = get_project_messages_from_backend(st.session_state.project_id) or []
                if not st.session_state.project_messages_data:
                    st.caption("No messages found or project is not active.")
                else:
                    st.experimental_rerun() # show loaded messages
        else:
            st.caption("No messages to display yet, or project is not active.")


    with tab_artifacts:
        st.subheader("Project Artifacts")
        # Artifacts are typically shown when project is completed, or upon manual refresh
        if force_refresh_data or (st.session_state.project_status_data and st.session_state.project_status_data.get("status") == "completed" and not st.session_state.project_artifacts_data):
            with st.spinner("Loading artifacts..."):
                st.session_state.project_artifacts_data = get_project_artifacts_from_backend(st.session_state.project_id) or []

        if st.session_state.project_artifacts_data:
            for artifact in st.session_state.project_artifacts_data:
                col1, col2, col3 = st.columns([3,2,2])
                with col1:
                    st.text(artifact.get("artifact_name"))
                with col2:
                    st.text(f"Type: {artifact.get('artifact_type')}")
                with col3:
                    # Construct full URL for download
                    artifact_download_url = f"{API_BASE_URL}/projects/{st.session_state.project_id}/artifacts/{artifact.get('artifact_name')}"
                    st.link_button("Download", artifact_download_url) # Direct link
        elif st.session_state.project_status_data and st.session_state.project_status_data.get("status") == "completed":
             st.caption("No artifacts found for this project.")
        else:
            st.caption("Artifacts will be available once the project is completed, or press 'Refresh All Data'.")

    with tab_logs:
        st.subheader("Project Logs")
        if force_refresh_data and not st.session_state.project_logs_data: # Initial load on refresh
             with st.spinner("Loading logs..."):
                st.session_state.project_logs_data = get_project_logs_from_backend(st.session_state.project_id) or []

        if st.session_state.project_logs_data:
            st.text_area("Logs", "\n".join(st.session_state.project_logs_data), height=400, disabled=True)
        elif not is_active_project and not st.session_state.project_logs_data: # If not active and no data, try fetching once
            with st.spinner("Loading logs..."):
                st.session_state.project_logs_data = get_project_logs_from_backend(st.session_state.project_id) or []
                if not st.session_state.project_logs_data:
                    st.caption("No logs available.")
                else:
                    st.experimental_rerun() # show loaded logs
        else:
            st.caption("No logs to display yet, or project is not active.")


else:
    st.info("Create a new project to see details.")


# To run this Streamlit app:
# 1. Ensure the FastAPI backend (`frontend_api_svc.py`) is running.
# 2. Save this file as `metagpt_streamlit_ui.py`.
# 3. Install Streamlit and requests: `pip install streamlit requests`
# 4. Run from terminal: `streamlit run metagpt_streamlit_ui.py`
#
# Note on st.experimental_rerun for polling:
# Using st.experimental_rerun in a loop with time.sleep can lead to
# high resource usage or unexpected behavior. A more robust solution for
# live updates involves websockets or a callback mechanism if Streamlit
# adds more direct support. For this prototype, a manual refresh button
# and a simple time-based refresh on interaction is a starting point.
# The current implementation reruns if a refresh happened, to show spinner and updated data.
# If project is still active, it will trigger another rerun after interval.
# This can be aggressive. Consider removing the auto-rerun if it causes issues.
# For now, the auto-refresh is tied to the REFRESH_INTERVAL and only if the project is active.
#
# The `project_name_input` was commented out because the API's `ProjectCreateRequest`
# currently only has `idea` and `project_name`. If the backend handles project name
# generation or if it's always based on idea, this simplifies the UI.
# If you want the user to set the project name, uncomment it and pass it to `create_project_on_backend`.
#
# Removed progress and current_task from display as they were commented out in API's ProjectStatusResponse.
# Can be re-added if API provides them.
from typing import Optional # Added for type hint
