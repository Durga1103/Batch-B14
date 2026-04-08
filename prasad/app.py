"""
app.py – Main Streamlit entry point for the Smart Education: Prediction of Learning Effectiveness using ML.

Run with:
    python -m streamlit run app.py
"""

import os
import hashlib
import re
import streamlit as st

st.set_page_config(
    page_title="Smart Education: Prediction of Learning Effectiveness using ML",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="auto",
)

CSS_PATH = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from db import init_db, create_user, get_user, username_exists
init_db()                                                            

_defaults = {
    "page":       "home",
    "logged_in":  False,
    "username":   "",
    "full_name":  "",
    "role":       "",                                   
}
for key, value in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

def hash_password(plain: str) -> str:
    """Return SHA-256 hex digest of a plain-text password."""
    return hashlib.sha256(plain.encode()).hexdigest()

def verify_password(plain: str, hashed: str) -> bool:
    """Check if plain password matches its stored hash."""
    return hash_password(plain) == hashed

def is_valid_email(email: str) -> bool:
    """Validate email – accepts standard email formats (not just gmail)."""
    pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None

def show_home():
    """Render the Home page."""
    st.markdown(
        """
        <div class="hero-banner">
            <h1>📚 Smart Education: Prediction of Learning Effectiveness using ML</h1>
            <p>Empowering students through intelligent skill assessment and personalised guidance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <h3>🔬 Smart Prediction</h3>
                <p>ML models (XGBoost, SVM, AdaBoost) classify your skill level instantly.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <h3>💡 Personalised Tips</h3>
                <p>Get targeted suggestions to improve your weak areas and grow faster.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <h3>📂 Faculty Resources</h3>
                <p>Access notes, PDFs, and study materials shared by your instructors.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("<h3 style='text-align:center;'>Get Started</h3>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        col_l, col_r = st.columns(2)
        with col_l:
            if st.button("🔐 Login", use_container_width=True):
                st.session_state.page = "login"
                st.rerun()
        with col_r:
            if st.button("✏️ Sign Up", use_container_width=True):
                st.session_state.page = "signup"
                st.rerun()

def show_signup():
    """Render the Signup form."""
    st.markdown("<h2 class='page-title'>✏️ Create New Account</h2>", unsafe_allow_html=True)

    with st.form("signup_form", clear_on_submit=False):
        full_name = st.text_input("Full Name")
        username  = st.text_input("Username")
        password  = st.text_input("Password", type="password")
        confirm   = st.text_input("Confirm Password", type="password")
        email     = st.text_input("Email")
        gender    = st.selectbox("Gender", ["Male", "Female", "Other"])
        age       = st.number_input("Age", min_value=10, max_value=100, value=18)
        role      = st.selectbox("Register As", ["student", "faculty"], format_func=lambda r: r.capitalize())

        submitted = st.form_submit_button("Create Account", use_container_width=True)

    if submitted:
        errors = []
        if not all([full_name, username, password, confirm, email]):
            errors.append("All fields are required.")
        if password != confirm:
            errors.append("Passwords do not match.")
        if not is_valid_email(email):
            errors.append("Invalid email address.")
        if len(password) < 6:
            errors.append("Password must be at least 6 characters.")

        if errors:
            for e in errors:
                st.error(e)
            return

        if username_exists(username):
            st.warning("⚠️ Username already exists. Please choose another.")
            return

        ok = create_user(full_name, username, hash_password(password), email, gender, age, role)
        if ok:
            st.success("✅ Account created successfully! Please log in.")
            st.session_state.page = "login"
            st.rerun()
        else:
            st.error("Something went wrong. Please try again.")

    st.markdown("---")
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

def show_login():
    """Render the dual-card Login forms (Faculty & Student)."""
    st.markdown("<h2 class='page-title' style='text-align: center;'>🔐 Login to Your Account</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, spacer, col2 = st.columns([1, 0.1, 1])

    fac_submitted = False
    stu_submitted = False
    username_val = ""
    password_val = ""
    attempted_role = ""

    with col1:
        st.markdown(
            """
            <div class='dual-login-card faculty-login'>
                <div class='login-header'>Faculty Login</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("faculty_login"):
            st.markdown("<div style='padding: 1rem;'>", unsafe_allow_html=True)
            fac_user = st.text_input("Username", key="fac_user")
            fac_pass = st.text_input("Password", type="password", key="fac_pass")
            fac_sub = st.form_submit_button("LOGIN", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            if fac_sub:
                fac_submitted = True
                username_val = fac_user
                password_val = fac_pass
                attempted_role = "faculty"

    with col2:
        st.markdown(
            """
            <div class='dual-login-card student-login'>
                <div class='login-header'>Student Login</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("student_login"):
            st.markdown("<div style='padding: 1rem;'>", unsafe_allow_html=True)
            stu_user = st.text_input("Username", key="stu_user")
            stu_pass = st.text_input("Password", type="password", key="stu_pass")
            stu_sub = st.form_submit_button("LOGIN", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            if stu_sub:
                stu_submitted = True
                username_val = stu_user
                password_val = stu_pass
                attempted_role = "student"

    if fac_submitted or stu_submitted:
        if not username_val.strip() or not password_val.strip():
            st.error("Username and Password cannot be empty.")
            return

        user = get_user(username_val.strip())
        if user and verify_password(password_val, user["password"]):
            if user["role"] != attempted_role:
                st.error(f"❌ Account found, but it is registered as a {user['role'].title()}. Please use the correct login form.")
                return

            st.session_state.logged_in  = True
            st.session_state.username   = user["username"]
            st.session_state.full_name  = user["full_name"]
            st.session_state.role       = user["role"]

            if user["role"] == "faculty":
                st.session_state.page = "faculty_dashboard"
            else:
                st.session_state.page = "student_dashboard"

            st.success(f"Welcome back, {user['full_name']}! 👋")
            st.rerun()
        else:
            st.error("❌ Invalid username or password. Please try again.")

    st.markdown("<br><hr>", unsafe_allow_html=True)
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        if st.button("← Back to Home", key="btn_back_home"):
            st.session_state.page = "home"
            st.rerun()
    with bcol2:
        if st.button("Don't have an account? Sign Up", key="btn_signup"):
            st.session_state.page = "signup"
            st.rerun()

page = st.session_state.page

if page == "home":
    show_home()
elif page == "signup":
    show_signup()
elif page == "login":
    show_login()
elif page == "student_dashboard":
    from student import show_student_dashboard
    show_student_dashboard()
elif page == "faculty_dashboard":
    from faculty import show_faculty_dashboard
    show_faculty_dashboard()
else:
    st.session_state.page = "home"
    st.rerun()
