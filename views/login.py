import streamlit as st

from src import auth
from src.ui import inject_css

inject_css()
st.markdown("<style>[data-testid='stSidebar'],[data-testid='stSidebarCollapsedControl']{display:none}</style>", unsafe_allow_html=True)


def _sign_in(user: dict) -> None:
    st.session_state["user"] = user
    st.session_state["_welcome"] = user["full_name"].split()[0]
    st.rerun()


_, mid, _ = st.columns([1, 1.5, 1])
with mid:
    st.markdown("<div style='text-align:center;margin-top:2rem'><h1 style='margin-bottom:0'>RetailPulse</h1>"
                "<p style='opacity:.7'>Sales, customers, forecasts and inventory in one place.</p></div>", unsafe_allow_html=True)
    with st.container(border=True):
        t_login, t_signup = st.tabs(["Log in", "Sign up"])
        with t_login:
            with st.form("login_form"):
                ident = st.text_input("Username or email", key="login_id")
                pw = st.text_input("Password", type="password", key="login_pw")
                go = st.form_submit_button("Log in", type="primary", width="stretch", key="login_submit")
            if go:
                user, err = auth.authenticate(ident, pw)
                if user:
                    _sign_in(user)
                else:
                    st.error(err)
            st.caption("New here? Open the **Sign up** tab to create an account.")
        with t_signup:
            with st.form("signup_form"):
                name = st.text_input("Full name", key="su_name")
                c1, c2 = st.columns(2)
                uname = c1.text_input("Username", key="su_user")
                email = c2.text_input("Email", key="su_email")
                p1 = c1.text_input("Password", type="password", key="su_pw")
                p2 = c2.text_input("Confirm password", type="password", key="su_pw2")
                st.caption("Password: at least 8 characters, with a letter and a number.")
                go = st.form_submit_button("Create account", type="primary", width="stretch", key="su_submit")
            if go:
                user, errs = auth.create_user(name, uname, email, p1, p2)
                if user:
                    _sign_in(user)
                else:
                    for e in errs:
                        st.error(e)
