"""
faculty.py – Faculty-only dashboard and related utilities.
"""

import os
import uuid
import re
import streamlit as st
from db import get_all_resources, add_resource, delete_resource

ALLOWED_EXTENSIONS = {"pdf", "docx", "doc", "txt", "pptx", "ppt", "xlsx"}

def allowed_file(filename: str) -> bool:
    """Return True if the file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def safe_filename(filename: str) -> str:
    """Sanitize filename by removing unsafe characters."""
    return re.sub(r"[^\w.\- ]", "_", filename)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def _logout():
    """Clear session and return to home."""
    for key in ["logged_in", "username", "full_name", "role", "page"]:
        st.session_state.pop(key, None)
    st.session_state.page = "home"
    st.rerun()

def show_faculty_dashboard():
    """Render the Faculty Dashboard."""

    if not st.session_state.get("logged_in") or st.session_state.get("role") != "faculty":
        st.error("🚫 Access denied. Please log in as a faculty member.")
        st.stop()

    with st.sidebar:
        st.markdown(
            f"""
            <div class='sidebar-profile'>
                <div class='avatar'>🎓</div>
                <h3>{st.session_state.get('full_name', 'Faculty')}</h3>
                <span class='role-badge faculty'>Faculty</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("### Navigation")
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            _logout()

    st.markdown(
        "<h2 class='page-title'>🏫 Faculty Dashboard</h2>",
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["⬆️ Upload Resource", "📋 Manage Resources"])

    with tab1:
        st.markdown("### Upload a Study Resource")
        st.caption("Allowed file types: PDF, DOCX, DOC, PPTX, PPT, TXT, XLSX")

        with st.form("upload_form", clear_on_submit=True):
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=["pdf", "docx", "doc", "pptx", "ppt", "txt", "xlsx"],
            )
            description = st.text_area("Description (optional)", max_chars=300, height=80)
            submit_btn = st.form_submit_button("📤 Upload File", use_container_width=True)

        if submit_btn:
            if not uploaded_file:
                st.error("Please select a file to upload.")
            elif not allowed_file(uploaded_file.name):
                st.error(f"File type not allowed: {uploaded_file.name}")
            else:

                original_name  = uploaded_file.name
                safe_name      = safe_filename(original_name)

                stored_name    = f"{uuid.uuid4().hex}_{safe_name}"
                file_path      = os.path.join(UPLOAD_DIR, stored_name)

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                add_resource(
                    filename      = stored_name,
                    original_name = original_name,
                    description   = description.strip(),
                    uploader      = st.session_state.get("username", "faculty"),
                )
                st.success(f"✅ '{original_name}' uploaded successfully!")
                st.rerun()

    with tab2:
        st.markdown("### All Uploaded Resources")
        resources = get_all_resources()

        if not resources:
            st.info("No resources uploaded yet. Use the Upload tab to add files.")
        else:
            for res in resources:
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(
                        f"""
                        <div class='resource-card'>
                            <b>📄 {res['original_name']}</b><br>
                            <small>Uploaded by <i>{res['uploader']}</i> on {res['upload_date']}</small>
                            {f"<br><small>{res['description']}</small>" if res['description'] else ""}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col2:
                    if st.button("🗑️ Delete", key=f"del_{res['id']}"):
                        filename = delete_resource(res["id"])
                        if filename:
                            file_path = os.path.join(UPLOAD_DIR, filename)
                            if os.path.exists(file_path):
                                os.remove(file_path)
                            st.success("File deleted.")
                            st.rerun()
                        else:
                            st.error("Could not delete file.")
                st.markdown("---")
