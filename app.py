
import hashlib, json, os, subprocess, sys, time, urllib.parse
import numpy as np, pandas as pd, joblib, streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="VR BioAuth | Behavioral Biometrics Login", page_icon="🥽",
                   layout="wide", initial_sidebar_state="collapsed")
FEATURES = ["typing_speed_wpm", "dwell_ms", "flight_ms", "grip_pressure",
            "trigger_force", "hand_velocity", "orientation_deg"]
NICE = ["Typing speed (WPM)", "Dwell time (ms)", "Flight time (ms)", "Grip pressure",
        "Trigger force", "Hand velocity (m/s)", "Orientation (deg)"]
GEN_MEAN = np.array([48, 95, 140, 0.62, 0.55, 1.10, 32])
GEN_STD  = np.array([5, 10, 18, 0.05, 0.06, 0.12, 4])
IMP_MEAN = np.array([25, 145, 230, 0.30, 0.28, 1.80, 58])
T = 10
STEPS = ["Start Login", "Data Capture", "Extraction", "AI/ML Model", "Decision"]
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
USERS = {"2405051530061": ("Kunal Tiwari", sha("kunal@2008"))}   # CRN / Roll -> (name, password hash)

for k, v in dict(step=1, user=None, uid=None, attempts=0, capture=None, history=[]).items():
    st.session_state.setdefault(k, v)
S = st.session_state

def go(n): S.step = n; st.rerun()
def logout():
    for k in ("user", "uid", "capture"): S[k] = None
    S.attempts = 0; go(1)
st.markdown("""<style>
[data-testid="stSidebar"], [data-testid="collapsedControl"] {display:none;}
.stApp {background: radial-gradient(circle at 15% 10%, #0b2540 0%, #0a0f1c 45%, #0d1024 100%); color:#e6edf7;}
h1,h2,h3 {color:#ffffff;} .accent {color:#38bdf8;}
.card {background:rgba(15,23,42,.8); border:1px solid #1e3a5f; border-radius:16px; padding:22px; margin-bottom:14px;}
.pill {display:inline-block; padding:4px 14px; border:1px solid #1e5f8f; border-radius:999px; color:#38bdf8;
       font-size:12px; letter-spacing:2px; background:#0b2236;}
.steps {display:flex; gap:8px; margin:6px 0 22px 0;}
.s {flex:1; padding:10px; text-align:center; border-radius:10px; background:#111a2e; border:1px solid #1e293b; color:#64748b; font-size:13px;}
.s.done {color:#38bdf8; border-color:#1e5f8f;} .s.now {background:#0c4a6e; color:#fff; border-color:#38bdf8; font-weight:700;}
.ok {background:#052e1b; border:1px solid #16a34a; color:#86efac; border-radius:14px; padding:20px; font-size:20px;}
.bad {background:#350a0a; border:1px solid #dc2626; color:#fca5a5; border-radius:14px; padding:20px; font-size:20px;}
div.stButton > button, div[data-testid="stFormSubmitButton"] > button
  {background:#0284c7; color:white; border:0; border-radius:10px; padding:.55rem 1.4rem; font-weight:600;}
</style>""", unsafe_allow_html=True)
OWNER = "Kunal Tiwari"
_svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="460" height="280"><text x="30" y="160" '
        'transform="rotate(-18 230 140)" font-family="Arial" font-size="34" font-weight="700" '
        f'fill="rgba(56,189,248,0.07)">{OWNER.upper()}</text></svg>')
st.markdown(f"""<style>.stApp{{background-image:url("data:image/svg+xml;utf8,{urllib.parse.quote(_svg)}"),
radial-gradient(circle at 15% 10%, #0b2540 0%, #0a0f1c 45%, #0d1024 100%);}}
.footer{{text-align:center; color:#64748b; font-size:13px; margin-top:30px;}}</style>""", unsafe_allow_html=True)
HERO = """
<div id="c" style="width:100%;height:__H__px"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const el=document.getElementById('c');
const W=()=>el.clientWidth, H=()=>el.clientHeight;
const r=new THREE.WebGLRenderer({antialias:true,alpha:true});
r.setPixelRatio(window.devicePixelRatio); r.setSize(W(),H()); el.appendChild(r.domElement);
const s=new THREE.Scene(), cam=new THREE.PerspectiveCamera(45,W()/H(),.1,100); cam.position.set(0,0,10);
s.add(new THREE.AmbientLight(0x8899ff,.7));
const L=new THREE.PointLight(0x38bdf8,1.6); L.position.set(5,5,7); s.add(L);
// VR headset
const hs=new THREE.Group();
const dark=new THREE.MeshStandardMaterial({color:0x1e293b,metalness:.6,roughness:.35});
const glow=new THREE.MeshStandardMaterial({color:0x38bdf8,emissive:0x0ea5e9,emissiveIntensity:1.2});
hs.add(new THREE.Mesh(new THREE.BoxGeometry(3.2,1.6,1.5),dark));
[-.8,.8].forEach(x=>{const l=new THREE.Mesh(new THREE.CylinderGeometry(.5,.5,.2,32),glow);
  l.rotation.x=Math.PI/2; l.position.set(x,0,.8); hs.add(l);});
const strap=new THREE.Mesh(new THREE.TorusGeometry(1.7,.13,12,48),dark); strap.position.z=-.9; strap.scale.set(1,.7,1); hs.add(strap);
hs.position.y=.5; hs.scale.setScalar(.85); s.add(hs);
// security shield globe + particles
const globe=new THREE.Mesh(new THREE.IcosahedronGeometry(3.1,1),
  new THREE.MeshBasicMaterial({color:0x38bdf8,wireframe:true,transparent:true,opacity:.22})); globe.position.y=.5; s.add(globe);
const N=500, pos=new Float32Array(N*3); for(let i=0;i<N*3;i++) pos[i]=(Math.random()-.5)*22;
const pg=new THREE.BufferGeometry(); pg.setAttribute('position',new THREE.BufferAttribute(pos,3));
const pts=new THREE.Points(pg,new THREE.PointsMaterial({color:0x7dd3fc,size:.05})); s.add(pts);
// name plate
const cv=document.createElement('canvas'); cv.width=1024; cv.height=200; const g=cv.getContext('2d');
const gr=g.createLinearGradient(0,0,1024,0); gr.addColorStop(0,'#ffffff'); gr.addColorStop(1,'#38bdf8');
g.fillStyle=gr; g.font='bold 120px Arial'; g.textAlign='center'; g.textBaseline='middle'; g.fillText('KUNAL TIWARI',512,100);
const name=new THREE.Mesh(new THREE.PlaneGeometry(6.4,1.25),new THREE.MeshBasicMaterial({map:new THREE.CanvasTexture(cv),transparent:true}));
name.position.y=-3.1; s.add(name);
let mx=0,my=0; window.addEventListener('mousemove',e=>{mx=e.clientX/innerWidth-.5; my=e.clientY/innerHeight-.5;});
window.addEventListener('resize',()=>{r.setSize(W(),H()); cam.aspect=W()/H(); cam.updateProjectionMatrix();});
(function loop(t){ requestAnimationFrame(loop); t*=.001;
  hs.rotation.y=Math.sin(t*.8)*.7+mx; hs.rotation.x=my*.4; hs.position.y=.5+Math.sin(t*1.5)*.12;
  globe.rotation.y=t*.3; globe.rotation.x=t*.15; pts.rotation.y=t*.03; name.position.y=-3.1+Math.sin(t)*.05;
  r.render(s,cam); })(0);
</script>
"""
def hero(h):
    html = HERO.replace("__H__", str(h))
    if hasattr(st, "iframe"):          
        st.iframe(html, height=h)
    else:                             
        components.html(html, height=h)

def ensure_capture():
    if S.capture is None:
        rng = np.random.default_rng()
        S.capture = pd.DataFrame(GEN_MEAN + rng.normal(0, GEN_STD * .25, (T, 7)), columns=FEATURES)
        S.capture.index = [f"t{i+1}" for i in range(T)]

def ensure_scores():
    ensure_capture()
    scaler, models = load_models()
    if S.get("model") not in models: S.model = list(models)[0]
    S.setdefault("threshold", 0.5)
    S.scores = {n: score_of(n, m, scaler, S.capture.values) for n, m in models.items()}

def header():
    c1, c2 = st.columns([5, 2])
    c1.markdown("### 🥽 VR <span class='accent'>BioAuth</span> &nbsp;<span class='pill'>NEXT-GEN CYBERSECURITY</span>",
                unsafe_allow_html=True)
    if S.user:
        c2.markdown(f"👤 **{S.user}** ({S.uid})")
        if c2.button("Logout"): logout()
    hero(330 if S.step == 1 and not S.user else 230)
    html = "".join(f"<div class='s {'now' if i+1==S.step else 'done' if i+1<S.step else ''}'>{i+1}. {n}</div>"
                   for i, n in enumerate(STEPS))
    st.markdown(f"<div class='steps'>{html}</div>", unsafe_allow_html=True)
    if S.user:   # after login every platform page is open
        for i, (c, n) in enumerate(zip(st.columns(4), STEPS[1:]), 2):
            if c.button(f"{i}. {n}", key=f"nav{i}", use_container_width=True): go(i)

@st.cache_resource
def load_models():
    if not os.path.exists("models/scaler.pkl"):
        subprocess.run([sys.executable, "train_models.py"], check=True)
    scaler, m = joblib.load("models/scaler.pkl"), {}
    for name, f in [("SVM", "svm.pkl"), ("Random Forest", "rf.pkl"), ("XGBoost", "xgb.pkl")]:
        if os.path.exists(f"models/{f}"): m[name] = joblib.load(f"models/{f}")
    if os.path.exists("models/lstm.keras"):
        from tensorflow import keras
        m["LSTM"] = keras.models.load_model("models/lstm.keras")
    return scaler, m

def score_of(name, model, scaler, seq):
    if name == "LSTM":
        return float(model.predict(scaler.transform(seq)[None], verbose=0)[0][0])
    return float(model.predict_proba(scaler.transform(seq.mean(axis=0, keepdims=True)))[0][1])

header()

if S.step == 1:
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.markdown("<div class='card'><h2>Secure Login</h2><p>Enter your <b>CRN / Roll Number</b> and "
                    "<b>Password</b> inside the VR portal. Your behaviour will be verified next.</p></div>",
                    unsafe_allow_html=True)
        if S.attempts >= 3:
            st.error("🔒 Account locked after 3 wrong attempts. Contact the administrator.")
            if st.button("Reset (demo)"): S.attempts = 0; go(1)
        else:
            with st.form("login"):
                uid = st.text_input("CRN / Roll Number", placeholder="Enter your CRN / Roll Number")
                pw = st.text_input("Password", type="password", placeholder="Enter your password")
                ok = st.form_submit_button("Login  →", use_container_width=True)
            if ok:
                u = USERS.get(uid.strip())
                if u and u[1] == sha(pw):
                    S.user, S.uid, S.attempts = u[0], uid.strip(), 0
                    st.success("Password correct ✔"); time.sleep(.6); go(2)
                else:
                    S.attempts += 1
                    st.error(f"❌ Incorrect CRN/Roll number or password. {3 - S.attempts} attempt(s) left.")

elif S.step == 2:
    st.markdown(f"## Step 2: Data Capture — Keystroke + Controller Dynamics")
    st.write(f"Welcome **{S.user}**. Password accepted. Now the system records how you type and use the VR controller.")
    mode = st.radio("Capture mode", ["Enrolled user (genuine behaviour)", "Impostor (someone else)", "Manual sliders"],
                    horizontal=True)
    if mode == "Manual sliders":
        c1, c2 = st.columns(2)
        with c1:
            wpm = st.slider("Typing speed (WPM)", 10, 90, 48); dwell = st.slider("Dwell time (ms)", 40, 200, 95)
            flight = st.slider("Flight time (ms)", 60, 300, 140)
        with c2:
            grip = st.slider("Grip pressure", 0.0, 1.0, 0.62, 0.01); trig = st.slider("Trigger force", 0.0, 1.0, 0.55, 0.01)
            vel = st.slider("Hand velocity (m/s)", 0.0, 2.5, 1.10, 0.05); ang = st.slider("Orientation (deg)", 0, 90, 32)
        base, std = np.array([wpm, dwell, flight, grip, trig, vel, ang], float), GEN_STD * .25
    else:
        base = GEN_MEAN if mode.startswith("Enrolled") else IMP_MEAN
        std = GEN_STD * (1 if mode.startswith("Enrolled") else .5)
    if st.button("🎙️ Start capture"):
        bar = st.progress(0, text="Recording keystrokes and controller motion...")
        for i in range(T): time.sleep(.08); bar.progress((i + 1) / T)
        rng = np.random.default_rng()
        centre = base + rng.normal(0, std) if mode != "Manual sliders" else base
        S.capture = pd.DataFrame(centre + rng.normal(0, GEN_STD * .25, (T, 7)), columns=FEATURES)
        S.capture.index = [f"t{i+1}" for i in range(T)]
    if S.capture is not None:
        st.success(f"Captured {T} time-steps × 7 behavioural signals")
        a, b = st.columns(2)
        a.dataframe(S.capture.style.format("{:.2f}"), height=300)
        b.line_chart(S.capture[["dwell_ms", "flight_ms"]])
        if st.button("Next: Feature Extraction →"): go(3)
elif S.step == 3:
    st.markdown("## Step 3: Extraction — Feature Vector")
    ensure_capture()
    scaler, models = load_models()
    raw = S.capture.mean().values
    scaled = scaler.transform([raw])[0]
    df = pd.DataFrame({"Feature": NICE, "Raw mean value": raw, "Normalised (z-score)": scaled})
    a, b = st.columns(2)
    a.dataframe(df.style.format({"Raw mean value": "{:.2f}", "Normalised (z-score)": "{:.2f}"}), hide_index=True)
    b.bar_chart(pd.DataFrame({"z-score": scaled}, index=NICE))
    st.caption("Raw telemetry is averaged and standardised into one feature vector that the ML models understand.")
    if st.button("Next: AI/ML Model →"): go(4)
elif S.step == 4:
    st.markdown("## Step 4: AI/ML Model — Match Score Calculation")
    ensure_capture()
    with st.spinner("Loading models..."): scaler, models = load_models()
    S.model = st.selectbox("Decision model", list(models))
    scores = {n: score_of(n, m, scaler, S.capture.values) for n, m in models.items()}
    S.scores = scores
    cols = st.columns(len(scores))
    for c, (n, v) in zip(cols, scores.items()): c.metric(n, f"{v:.1%}")
    st.progress(min(max(scores[S.model], 0.0), 1.0), text=f"Match score ({S.model})")
    S.threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)
    if os.path.exists("models/metrics.json"):
        with st.expander("Model accuracy (test set)"):
            st.dataframe(pd.DataFrame(json.load(open("models/metrics.json"))).T.style.format("{:.2%}"))
    if st.button("Next: Decision →"): go(5)

else:
    st.markdown("## Step 5: Decision — Access Granted / Failed")
    ensure_scores()
    score, granted = S.scores[S.model], S.scores[S.model] >= S.threshold
    S.history.append({"time": time.strftime("%H:%M:%S"), "user": S.uid, "model": S.model,
                      "score": f"{score:.1%}", "result": "GRANTED" if granted else "FAILED"}) \
        if not S.history or S.history[-1].get("key") != id(S.capture) else None
    if S.history: S.history[-1]["key"] = id(S.capture)
    if granted:
        st.markdown(f"<div class='ok'>✅ <b>ACCESS GRANTED</b> — behaviour matches {S.user} "
                    f"(score {score:.1%} ≥ {S.threshold:.0%})</div>", unsafe_allow_html=True)
        st.balloons()
        st.markdown("### 🔐 Secure VR Workspace")
        t1, t2, t3 = st.tabs(["Session", "Behaviour profile", "Continuous monitoring"])
        with t1:
            c = st.columns(4)
            c[0].metric("User", S.user); c[1].metric("CRN / Roll", S.uid)
            c[2].metric("Trust score", f"{score:.1%}"); c[3].metric("Model", S.model)
            st.dataframe(pd.DataFrame(S.history).drop(columns="key", errors="ignore"), hide_index=True)
        with t2:
            st.dataframe(pd.DataFrame({"Feature": NICE, "Your value": S.capture.mean().values,
                                       "Enrolled mean": GEN_MEAN}).style.format({"Your value": "{:.2f}", "Enrolled mean": "{:.2f}"}),
                         hide_index=True)
        with t3:
            rng = np.random.default_rng(1)
            st.line_chart(pd.DataFrame({"Trust score": np.clip(score + rng.normal(0, .015, 30), 0, 1)}))
            st.caption("Background session monitoring keeps re-checking behaviour while you use VR (simulated).")
    else:
        st.markdown(f"<div class='bad'>⛔ <b>ACCESS FAILED</b> — behaviour does not match the enrolled user "
                    f"(score {score:.1%} &lt; {S.threshold:.0%}). Even with the correct password, access is denied.</div>",
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(S.history).drop(columns="key", errors="ignore"), hide_index=True)
    a, b = st.columns(2)
    if a.button("🔁 Try capture again"): S.capture = None; go(2)
    if b.button("🚪 Logout"): logout()

st.markdown(f"<div class='footer'>Designed &amp; developed by <b>{OWNER}</b> · VR BioAuth Behavioral Biometrics</div>", unsafe_allow_html=True)