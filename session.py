import hashlib
import uuid

def get_session_id(st, token):
    if "session_id" not in st.session_state:
        user_agent= st.request.headers.get("User-Agent", "")
        remote_ip= st.request.remote_addr or "0.0.0.0"
        raw= f"{remote_ip}-{user_agent}-{token}"
        session_hash= hashlib.sha256(raw.encode()).hexdigest()
        st.session_state.session_id= session_hash
    return st.session_state.session_id

def get_temp_key():
    return f"temp_{uuid.uuid4().hex}"