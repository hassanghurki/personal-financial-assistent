"""
auth.py — Login / Sign-up screen for the Personal Finance Assistant.

Renders a stunning sky-blue glassmorphism card with seamless white pill inputs and a centered layout.
Manages st.session_state keys: logged_in, username, user_id, auth_mode.
"""

from __future__ import annotations

import streamlit as st
from db import create_user, verify_user

def render_auth_page(conn) -> None:
    if "auth_mode" not in st.session_state:
        st.session_state["auth_mode"] = "login"

    if st.session_state["auth_mode"] == "login":
        with st.form("sky_login_card", clear_on_submit=False):
            # SVG User Icon at top of form
            st.markdown("""
                <div style="display: flex; justify-content: center; margin-bottom: 0.5rem;">
                    <svg width="68" height="68" viewBox="0 0 24 24" fill="#00426b" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 2C9.24 2 7 4.24 7 7C7 9.76 9.24 12 12 12C14.76 12 17 9.76 17 7C17 4.24 14.76 2 12 2ZM12 14C8.67 14 2 15.67 2 19V22H22V19C22 15.67 15.33 14 12 14Z"/>
                    </svg>
                </div>
                <div class="sky-title">User Login</div>
            """, unsafe_allow_html=True)
            
            username = st.text_input("Username", placeholder="Username", label_visibility="collapsed")
            password = st.text_input("Password", type="password", placeholder="Password", label_visibility="collapsed")
            
            st.markdown('<div style="text-align: center; color: #2c4a5c; font-size: 0.95rem; margin: 0.8rem 0 1rem; cursor: pointer;">Forgot Password?</div>', unsafe_allow_html=True)
            
            submitted = st.form_submit_button("Login")
            if submitted:
                if not username or not password:
                    st.error("Please enter both username and password.")
                else:
                    res = verify_user(conn, username, password)
                    if res["ok"]:
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = res["username"]
                        st.session_state["user_id"] = res["user_id"]
                        st.rerun()
                    else:
                        st.error(res["error"])
        
        st.markdown('<div class="sky-toggle-container">', unsafe_allow_html=True)
        st.button("Don't have an account? Sign Up", key="goto_signup", on_click=lambda: st.session_state.update(auth_mode="signup"))
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        with st.form("sky_signup_card", clear_on_submit=True):
            st.markdown("""
                <div style="display: flex; justify-content: center; margin-bottom: 0.5rem;">
                    <svg width="68" height="68" viewBox="0 0 24 24" fill="#00426b" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 2C9.24 2 7 4.24 7 7C7 9.76 9.24 12 12 12C14.76 12 17 9.76 17 7C17 4.24 14.76 2 12 2ZM12 14C8.67 14 2 15.67 2 19V22H22V19C22 15.67 15.33 14 12 14Z"/>
                    </svg>
                </div>
                <div class="sky-title">Create Account</div>
            """, unsafe_allow_html=True)
            
            new_username = st.text_input("Username", placeholder="Choose Username", label_visibility="collapsed")
            new_password = st.text_input("Password", type="password", placeholder="Choose Password", label_visibility="collapsed")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm Password", label_visibility="collapsed")
            
            st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
            
            submitted = st.form_submit_button("Sign Up")
            if submitted:
                if not new_username or not new_password or not confirm_password:
                    st.error("Please fill all fields.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    res = create_user(conn, new_username, new_password)
                    if res["ok"]:
                        st.success("Account created! Switch to Login.")
                    else:
                        st.error(res.get("error", "Error creating user"))
        
        st.markdown('<div class="sky-toggle-container">', unsafe_allow_html=True)
        st.button("Back to Login", key="goto_login", on_click=lambda: st.session_state.update(auth_mode="login"))
        st.markdown('</div>', unsafe_allow_html=True)


def is_logged_in() -> bool:
    return st.session_state.get("logged_in", False)


def get_current_user_id() -> str:
    return st.session_state.get("user_id", "default")


def get_current_username() -> str:
    return st.session_state.get("username", "user")


def logout() -> None:
    for key in ["logged_in", "username", "user_id", "chat_history", "last_report", "auth_mode"]:
        st.session_state.pop(key, None)
    st.rerun()