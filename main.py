import os, re, json, math, sqlite3, zlib, time, uuid, hmac, hashlib, base64, secrets
from contextlib import contextmanager
from typing import Optional
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()
KEY = os.getenv("NEXUS_API_KEY", "")
BASE = os.getenv("NEXUS_BASE_URL", "").rstrip("/")
MODEL = os.getenv("NEXUS_MODEL", "")
EMBED_MODEL = os.getenv("NEXUS_EMBED_MODEL", "")
SECRET = os.getenv("SECRET_KEY") or secrets.token_hex(32)   # set SECRET_KEY in production
if not os.getenv("SECRET_KEY"):
    print("WARNING: SECRET_KEY not set. Logins will reset on every restart.")
TOKEN_HOURS = 12
DB = "reloop.db"
CATEGORIES = ["Education", "Electronics", "Furniture", "Clothing", "Household", "Sports",
              "Toys and Games", "Appliances", "Office Supplies", "Other Reusable Items"]

app = FastAPI(title="ReLoop AI")

# ---------------- DB ----------------
@contextmanager
def db():
    con = sqlite3.connect(DB, isolation_level=None)
    con.row_factory = sqlite3.Row
    try:
        yield con
    finally:
        con.close()

def rows(cur): return [dict(r) for r in cur.fetchall()]

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(user_id TEXT PRIMARY KEY, full_name TEXT, email TEXT UNIQUE,
 password_hash TEXT, role TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS resources(resource_id INTEGER PRIMARY KEY AUTOINCREMENT, owner_id TEXT, title TEXT,
 description TEXT, category TEXT, quantity INTEGER, unit TEXT, condition TEXT, location TEXT,
 sharing_type TEXT, status TEXT DEFAULT 'Published', created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS needs(need_id INTEGER PRIMARY KEY AUTOINCREMENT, seeker_id TEXT, item_name TEXT,
 category TEXT, quantity INTEGER, location TEXT, preferred_condition TEXT, raw_text TEXT,
 created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS requests(request_id INTEGER PRIMARY KEY AUTOINCREMENT, need_id INTEGER, resource_id INTEGER,
 seeker_id TEXT, owner_id TEXT, requested_quantity INTEGER, status TEXT DEFAULT 'Pending',
 created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS notifications(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, message TEXT,
 created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS knowledge_chunks(chunk_id INTEGER PRIMARY KEY AUTOINCREMENT, doc_title TEXT,
 heading TEXT, chunk_text TEXT, approved INTEGER DEFAULT 1, embedding TEXT);
CREATE TABLE IF NOT EXISTS workflows(workflow_id TEXT PRIMARY KEY, user_id TEXT, query TEXT, final_output TEXT,
 created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS steps(step_id INTEGER PRIMARY KEY AUTOINCREMENT, workflow_id TEXT, agent TEXT,
 tool TEXT, input_summary TEXT, output_summary TEXT, status TEXT, ms INTEGER,
 created_at TEXT DEFAULT CURRENT_TIMESTAMP);
"""

KB = [
 ("Request and Reservation Rules", "Partial quantity",
  "A seeker may request any quantity up to the currently available quantity of a listing. If an owner lists five books, a seeker may request only two. The remaining three stay available for other seekers."),
 ("Request and Reservation Rules", "Accepting requests",
  "A request starts as Pending. Only the resource owner can accept or reject it. Accepted means the owner agrees and the requested quantity is reserved. Quantity is deducted only after acceptance."),
 ("Quantity Management Rules", "Quantity updates",
  "Quantities change only through confirmed reservations. The system checks available stock inside a transaction so the same items cannot be reserved twice."),
 ("Resource Listing Guidelines", "Listing a resource",
  "To list a reusable resource, provide a title, description, category, quantity, unit, condition, location and sharing type (Free, Donation or Paid). Review any AI suggestions before publishing."),
 ("Resource Condition Guidelines", "Conditions",
  "Allowed conditions are New, Very Good, Good and Fair. Describe defects honestly. Misleading condition descriptions can be reported."),
 ("Sharing Rules", "Sharing types",
  "Free items cost nothing. Donation items are given to support a cause. Paid listings require a price. Free sharing is always available on the platform."),
 ("Organization Guidelines", "Bulk sharing",
  "Organizations may list many items at once and receive bulk requests. Organization profiles require administrator approval."),
 ("Community Guidelines", "Reporting",
  "Users can report misleading or unsafe listings. Administrators review reports and may remove listings or suspend accounts."),
 ("Privacy Policy", "Personal data",
  "Owner contact details are never shown on public pages. Only information needed to complete an exchange is shared between the two parties."),
 ("Exchange Guidelines", "Collection",
  "After acceptance, owner and seeker agree a collection time through request messages. The exchange is complete when the owner marks the items as collected."),
]

SEED_RESOURCES = [
 ("U001","Engineering Mathematics Books","Textbooks for engineering maths","Education",5,"Books","Good","Coimbatore","Free"),
 ("U004","Study Table","Wooden study table","Furniture",2,"Pieces","Good","Coimbatore","Free"),
 ("U001","Computer Keyboard","USB keyboards","Electronics",3,"Pieces","Good","Chennai","Donation"),
 ("U004","Physics Reference Books","Physics reference","Education",4,"Books","Very Good","Coimbatore","Free"),
 ("U003","Notebooks","Ruled notebooks","Office Supplies",20,"Pieces","New","Coimbatore","Free"),
 ("U001","Football","Match football","Sports",3,"Pieces","Good","Coimbatore","Free"),
]

# Synthetic demo accounts (development only). Password for all: demo123
DEMO_PASSWORD = "demo123"
SEED_USERS = [
 ("U001", "Demo Owner One", "u001@reloop.demo", "owner"),
 ("U002", "Demo Seeker", "u002@reloop.demo", "seeker"),
 ("U003", "Demo Organization", "u003@reloop.demo", "organization"),
 ("U004", "Demo Owner Two", "u004@reloop.demo", "owner"),
 ("U005", "Demo Administrator", "admin@reloop.demo", "admin"),
]

# ---------------- Auth (standard library only) ----------------
def hash_pw(pw: str, salt: Optional[str] = None) -> str:
    salt = salt or secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), 200_000).hex()
    return f"{salt}${h}"

def check_pw(pw: str, stored: str) -> bool:
    try:
        salt, h = stored.split("$")
        return hmac.compare_digest(hash_pw(pw, salt).split("$")[1], h)
    except Exception:
        return False

def _b64(b: bytes) -> str: return base64.urlsafe_b64encode(b).decode().rstrip("=")
def _unb64(s: str) -> bytes: return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))

def make_token(uid: str) -> str:
    payload = json.dumps({"u": uid, "exp": int(time.time()) + TOKEN_HOURS * 3600}).encode()
    sig = hmac.new(SECRET.encode(), payload, hashlib.sha256).digest()
    return _b64(payload) + "." + _b64(sig)

def read_token(tok: str) -> Optional[str]:
    try:
        p, s = tok.split(".")
        payload = _unb64(p)
        good = hmac.new(SECRET.encode(), payload, hashlib.sha256).digest()
        if not hmac.compare_digest(_unb64(s), good): return None
        d = json.loads(payload)
        return d["u"] if d["exp"] > time.time() else None
    except Exception:
        return None

def current_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Login required")
    uid = read_token(authorization[7:])
    if not uid: raise HTTPException(401, "Session expired. Please log in again.")
    with db() as con:
        u = con.execute("SELECT user_id,full_name,email,role FROM users WHERE user_id=?", (uid,)).fetchone()
    if not u: raise HTTPException(401, "Unknown user")
    return dict(u)

def admin_user(u=Depends(current_user)):
    if u["role"] != "admin": raise HTTPException(403, "Administrator access only")
    return u

FAILS = {}
def too_many(email):
    now = time.time()
    FAILS[email] = [t for t in FAILS.get(email, []) if now - t < 300]
    return len(FAILS[email]) >= 5

# ---------------- Nexus LLM + Embeddings ----------------
def llm(system: str, user: str, json_mode=True, timeout=40):
    if not (KEY and BASE and MODEL):
        raise RuntimeError("Nexus not configured")
    last = None
    for attempt in range(2):  # retry limit
        try:
            r = httpx.post(f"{BASE}/chat/completions",
                headers={"Authorization": f"Bearer {KEY}"},
                json={"model": MODEL, "temperature": 0.1,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": user}]},
                timeout=timeout)
            if r.status_code == 429:
                time.sleep(2); continue
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            if not json_mode: return text
            m = re.search(r"\{.*\}", text, re.S)
            return json.loads(m.group(0))
        except Exception as e:
            last = e
    raise RuntimeError(f"Nexus call failed: {last}")

STOP = set("a an the is are was were be do does did can could i my me we you your it its of to for in on at by with and or if as from that this what how when where which who whom there their they only has have had not no than then so".split())

def hash_embed(text, dim=256):
    v = [0.0] * dim
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if t in STOP: continue
        v[zlib.crc32(t.encode()) % dim] += 1.0
    n = math.sqrt(sum(x*x for x in v)) or 1.0
    return [x/n for x in v]

def embed(text):
    if EMBED_MODEL and KEY and BASE:
        try:
            r = httpx.post(f"{BASE}/embeddings", headers={"Authorization": f"Bearer {KEY}"},
                           json={"model": EMBED_MODEL, "input": [text]}, timeout=30)
            r.raise_for_status()
            return r.json()["data"][0]["embedding"]
        except Exception:
            pass
    return hash_embed(text)

def cos(a, b):
    if len(a) != len(b): return 0.0
    d = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)) or 1; nb = math.sqrt(sum(x*x for x in b)) or 1
    return d/(na*nb)

# ---------------- RAG ----------------
def ingest_kb():
    with db() as con:
        if con.execute("SELECT COUNT(*) c FROM knowledge_chunks").fetchone()["c"]: return
        for title, heading, text in KB:
            con.execute("INSERT INTO knowledge_chunks(doc_title,heading,chunk_text,embedding) VALUES(?,?,?,?)",
                        (title, heading, text, json.dumps(embed(f"{title}. {heading}. {text}"))))

def retrieve(query, k=3):
    qv = embed(query)
    with db() as con:
        ch = rows(con.execute("SELECT * FROM knowledge_chunks WHERE approved=1"))
    for c in ch: c["score"] = round(cos(qv, json.loads(c["embedding"])), 4)
    ch.sort(key=lambda c: c["score"], reverse=True)
    return [{k2: v for k2, v in c.items() if k2 != "embedding"} for c in ch[:k]]

MIN_SCORE = 0.15
def rag_answer(query):
    hits = retrieve(query)
    hits = [h for h in hits if h["score"] >= MIN_SCORE]
    if not hits:
        return {"answer": "I cannot find sufficient approved information to answer that.", "sources": [], "status": "no_match"}
    ctx = "\n\n".join(f"[{h['doc_title']} > {h['heading']}]\n{h['chunk_text']}" for h in hits)
    system = ("You are the ReLoop AI Knowledge Assistant. Answer ONLY from the provided context. "
              "If the context lacks the answer, say you cannot find sufficient approved information. "
              "Never invent rules. Return JSON: {\"answer\": string}")
    try:
        out = llm(system, f"Context:\n{ctx}\n\nQuestion: {query}")
        return {"answer": out.get("answer", ""), "sources": hits, "status": "grounded"}
    except Exception:
        return {"answer": "AI assistance unavailable. Closest approved guideline: " + hits[0]["chunk_text"],
                "sources": hits, "status": "fallback"}

# ---------------- Tools (backend functions agents call) ----------------
def tool_search_resources(item, category):
    with db() as con:
        return rows(con.execute(
            "SELECT * FROM resources WHERE status='Published' AND quantity>0 AND (category=? OR title LIKE ?)",
            (category, f"%{item.split()[0] if item else ''}%")))

def tool_calculate_match_score(need, res):
    cat = 1.0 if need["category"] == res["category"] else 0.0
    nt = set(re.findall(r"[a-z]+", need["item_name"].lower()))
    rt = set(re.findall(r"[a-z]+", (res["title"] + " " + res["description"]).lower()))
    item = len(nt & rt) / len(nt) if nt else 0.0
    loc = 1.0 if need.get("location") and need["location"].lower() == (res["location"] or "").lower() else (0.5 if not need.get("location") else 0.0)
    qty = 1.0 if res["quantity"] >= need["quantity"] else res["quantity"] / max(need["quantity"], 1)
    order = ["Fair", "Good", "Very Good", "New"]
    pref = need.get("preferred_condition")
    cond = 1.0 if not pref or (res["condition"] in order and pref in order and order.index(res["condition"]) >= order.index(pref)) else 0.4
    avail = 1.0 if res["quantity"] > 0 else 0.0
    total = 30*cat + 25*item + 15*loc + 15*qty + 10*cond + 5*avail
    return {"category": cat, "item": round(item, 2), "location": loc, "quantity": round(qty, 2),
            "condition": cond, "availability": avail, "total": round(total, 1)}

# ---------------- Agents ----------------
def log_step(con, wf, agent, tool, inp, out, status, ms):
    con.execute("INSERT INTO steps(workflow_id,agent,tool,input_summary,output_summary,status,ms) VALUES(?,?,?,?,?,?,?)",
                (wf, agent, tool, str(inp)[:300], str(out)[:500], status, ms))

def need_agent(query):
    system = ("You are the Need Understanding Agent for ReLoop AI. Extract a need from the user text. "
              f"Categories: {CATEGORIES}. Return JSON: {{\"item_name\":str,\"category\":one of categories,"
              "\"quantity\":int,\"location\":str or \"\",\"preferred_condition\":str or \"\","
              "\"missing_fields\":[str]}}. Do not invent details not in the text.")
    try:
        n = llm(system, query)
        n["quantity"] = int(n.get("quantity") or 1)
        if n.get("category") not in CATEGORIES: n["category"] = "Other Reusable Items"
        return n, "nexus"
    except Exception:
        q = query.lower()
        m = re.search(r"\b(\d+|one|two|three|four|five)\b", q)
        words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        qty = int(m.group(1)) if m and m.group(1).isdigit() else words.get(m.group(1), 1) if m else 1
        item = re.sub(r"\b(i need|i want|find|free|please|\d+|one|two|three|four|five)\b", "", q).strip()
        cat = "Education" if "book" in q else "Furniture" if "table" in q or "chair" in q else \
              "Electronics" if "keyboard" in q else "Office Supplies" if "notebook" in q else "Other Reusable Items"
        return {"item_name": item or query, "category": cat, "quantity": qty, "location": "",
                "preferred_condition": "", "missing_fields": []}, "fallback"

def policy_agent(need, top):
    q = f"Can I reserve only {need['quantity']} {need['item_name']} if the owner has {top['quantity']}? Reservation and quantity rules"
    return rag_answer(q)

def explain(need, res, score):
    try:
        out = llm("You explain match results for ReLoop AI. Use ONLY the given numbers; never invent scores. "
                  "Return JSON {\"explanation\": string} in 2 sentences.",
                  json.dumps({"need": need, "resource": {k: res[k] for k in ("title", "quantity", "condition", "location")}, "score": score}))
        return out.get("explanation", "")
    except Exception:
        return f"Score {score['total']}/100 based on category, item, location, quantity and condition."

def orchestrator(user_id, query):
    wf = str(uuid.uuid4())[:8]
    result = {"workflow_id": wf, "query": query}
    with db() as con:
        con.execute("INSERT INTO workflows(workflow_id,user_id,query) VALUES(?,?,?)", (wf, user_id, query))
        log_step(con, wf, "Orchestrator", "classify_task", query, "task=need_matching", "ok", 0)

        t = time.time(); need, src = need_agent(query)
        log_step(con, wf, "Need Understanding Agent", f"nexus_llm({src})", query, need, "ok", int((time.time()-t)*1000))
        result["need"] = need
        con.execute("INSERT INTO needs(seeker_id,item_name,category,quantity,location,preferred_condition,raw_text) VALUES(?,?,?,?,?,?,?)",
                    (user_id, need["item_name"], need["category"], need["quantity"], need.get("location", ""),
                     need.get("preferred_condition", ""), query))
        result["need_id"] = con.execute("SELECT last_insert_rowid()").fetchone()[0]

        t = time.time(); found = tool_search_resources(need["item_name"], need["category"])
        log_step(con, wf, "Resource Discovery Agent", "search_resources (SQL)", need["item_name"],
                 f"{len(found)} live listings", "ok", int((time.time()-t)*1000))

        t = time.time(); matches = []
        for r in found:
            s = tool_calculate_match_score(need, r)
            matches.append({"resource": r, "score": s})
        matches.sort(key=lambda m: m["score"]["total"], reverse=True)
        for m in matches[:3]:
            m["explanation"] = explain(need, m["resource"], m["score"])
        log_step(con, wf, "Matching and Recommendation Agent", "calculate_match_score",
                 f"{len(found)} candidates", [m["score"]["total"] for m in matches], "ok", int((time.time()-t)*1000))
        result["matches"] = matches[:5]

        if matches:
            t = time.time(); pol = policy_agent(need, matches[0]["resource"])
            log_step(con, wf, "Policy and Trust Agent", "retrieve_knowledge (RAG)", "reservation rules",
                     f"{pol['status']}; sources={[s['doc_title'] for s in pol['sources']]}", "ok", int((time.time()-t)*1000))
            result["policy"] = pol
        else:
            result["policy"] = None
        result["human_approval_required"] = True  # owner must accept before any reservation
        con.execute("UPDATE workflows SET final_output=? WHERE workflow_id=?",
                    (json.dumps({"matches": len(matches)}), wf))
        log_step(con, wf, "Orchestrator", "combine_results", "all agents", f"{len(matches)} recommendations", "ok", 0)
    return result

# ---------------- API models ----------------
class Ask(BaseModel): query: str
class AgentReq(BaseModel): query: str
class NewRes(BaseModel):
    title: str; description: str = ""; category: str; quantity: int
    unit: str = "Pieces"; condition: str = "Good"; location: str = ""; sharing_type: str = "Free"
class ScanReq(BaseModel): text: str
class ReqIn(BaseModel): need_id: Optional[int] = None; resource_id: int; quantity: int
class Reg(BaseModel): full_name: str; email: str; password: str; role: str = "seeker"
class Login(BaseModel): email: str; password: str

@app.on_event("startup")
def startup():
    with db() as con:
        con.executescript(SCHEMA)
        if not con.execute("SELECT COUNT(*) c FROM users").fetchone()["c"]:
            for uid, name, email, role in SEED_USERS:
                con.execute("INSERT INTO users(user_id,full_name,email,password_hash,role) VALUES(?,?,?,?,?)",
                            (uid, name, email, hash_pw(DEMO_PASSWORD), role))
        if not con.execute("SELECT COUNT(*) c FROM resources").fetchone()["c"]:
            for r in SEED_RESOURCES:
                con.execute("INSERT INTO resources(owner_id,title,description,category,quantity,unit,condition,location,sharing_type) VALUES(?,?,?,?,?,?,?,?,?)", r)
    ingest_kb()

# ---------------- Auth routes ----------------
@app.post("/api/auth/register")
def register(r: Reg):
    email = r.email.strip().lower()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email): raise HTTPException(400, "Enter a valid email address")
    if len(r.password) < 8: raise HTTPException(400, "Password must be at least 8 characters")
    if not r.full_name.strip(): raise HTTPException(400, "Enter your full name")
    if r.role not in ("owner", "seeker", "organization"):   # admin accounts cannot be self-registered
        raise HTTPException(400, "Invalid role")
    with db() as con:
        if con.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
            raise HTTPException(409, "An account with this email already exists")
        n = con.execute("SELECT COALESCE(MAX(CAST(SUBSTR(user_id,2) AS INTEGER)),0)+1 AS n FROM users").fetchone()["n"]
        uid = f"U{n:03d}"
        con.execute("INSERT INTO users(user_id,full_name,email,password_hash,role) VALUES(?,?,?,?,?)",
                    (uid, r.full_name.strip()[:80], email, hash_pw(r.password), r.role))
    return {"token": make_token(uid), "user": {"user_id": uid, "full_name": r.full_name.strip(), "email": email, "role": r.role}}

@app.post("/api/auth/login")
def login(l: Login):
    email = l.email.strip().lower()
    if too_many(email): raise HTTPException(429, "Too many failed attempts. Try again in 5 minutes.")
    with db() as con:
        u = con.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if not u or not check_pw(l.password, u["password_hash"]):
        FAILS.setdefault(email, []).append(time.time())
        raise HTTPException(401, "Invalid email or password")
    FAILS.pop(email, None)
    return {"token": make_token(u["user_id"]),
            "user": {"user_id": u["user_id"], "full_name": u["full_name"], "email": u["email"], "role": u["role"]}}

@app.get("/api/auth/me")
def me(u=Depends(current_user)): return u

# ---------------- App routes ----------------
@app.get("/api/resources")
def list_resources():   # public: anyone can browse
    with db() as con: return rows(con.execute("SELECT * FROM resources ORDER BY resource_id DESC"))

@app.post("/api/resources")
def add_resource(r: NewRes, u=Depends(current_user)):
    if r.quantity < 1: raise HTTPException(400, "Quantity must be at least 1")
    if r.category not in CATEGORIES: raise HTTPException(400, "Invalid category")
    if r.sharing_type not in ("Free", "Donation", "Paid"): raise HTTPException(400, "Invalid sharing type")
    with db() as con:
        con.execute("INSERT INTO resources(owner_id,title,description,category,quantity,unit,condition,location,sharing_type) VALUES(?,?,?,?,?,?,?,?,?)",
                    (u["user_id"], r.title[:120], r.description[:1000], r.category, r.quantity, r.unit, r.condition, r.location, r.sharing_type))
    return {"ok": True}

@app.post("/api/scan")
def scan(s: ScanReq, u=Depends(current_user)):
    try:
        return llm(f"You are the Resource Analysis assistant. Categories: {CATEGORIES}. From the owner's text return JSON "
                   "{\"title\":str,\"category\":str,\"subcategory\":str,\"keywords\":[str],\"description\":str,\"quantity\":int,\"condition\":str}. "
                   "Only use details supported by the text.", s.text)
    except Exception as e:
        return {"error": "AI assistance unavailable. Please fill the form manually.", "detail": str(e)}

@app.post("/api/ask")
def ask(a: Ask, u=Depends(current_user)): return rag_answer(a.query)

@app.post("/api/agent/run")
def run_agent(a: AgentReq, u=Depends(current_user)): return orchestrator(u["user_id"], a.query)

@app.post("/api/requests")
def create_request(r: ReqIn, u=Depends(current_user)):
    with db() as con:
        res = con.execute("SELECT * FROM resources WHERE resource_id=?", (r.resource_id,)).fetchone()
        if not res: raise HTTPException(404, "Resource not found")
        if res["owner_id"] == u["user_id"]: raise HTTPException(400, "You cannot request your own listing")
        if r.quantity < 1 or r.quantity > res["quantity"]:
            raise HTTPException(400, f"Only {res['quantity']} available")
        con.execute("INSERT INTO requests(need_id,resource_id,seeker_id,owner_id,requested_quantity) VALUES(?,?,?,?,?)",
                    (r.need_id, r.resource_id, u["user_id"], res["owner_id"], r.quantity))
        con.execute("INSERT INTO notifications(user_id,message) VALUES(?,?)",
                    (res["owner_id"], f"New request for {r.quantity} x {res['title']} from {u['full_name']}"))
    return {"ok": True}

@app.get("/api/my/requests")
def my_requests(u=Depends(current_user)):
    with db() as con:
        return rows(con.execute("""SELECT q.*, r.title FROM requests q JOIN resources r ON r.resource_id=q.resource_id
                                   WHERE q.owner_id=? OR q.seeker_id=? ORDER BY q.request_id DESC""",
                                (u["user_id"], u["user_id"])))

@app.post("/api/requests/{rid}/accept")
def accept(rid: int, u=Depends(current_user)):
    with db() as con:
        con.execute("BEGIN IMMEDIATE")  # transaction-safe quantity update
        try:
            q = con.execute("SELECT * FROM requests WHERE request_id=?", (rid,)).fetchone()
            if not q or q["owner_id"] != u["user_id"]: raise HTTPException(403, "Only the owner can accept")
            if q["status"] != "Pending": raise HTTPException(400, "Request is not pending")
            res = con.execute("SELECT * FROM resources WHERE resource_id=?", (q["resource_id"],)).fetchone()
            if res["quantity"] < q["requested_quantity"]: raise HTTPException(400, "Insufficient quantity")
            con.execute("UPDATE resources SET quantity=quantity-? WHERE resource_id=?", (q["requested_quantity"], q["resource_id"]))
            con.execute("UPDATE requests SET status='Accepted' WHERE request_id=?", (rid,))
            con.execute("INSERT INTO notifications(user_id,message) VALUES(?,?)",
                        (q["seeker_id"], f"Your request for {q['requested_quantity']} x {res['title']} was accepted"))
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK"); raise
    return {"ok": True}

@app.post("/api/requests/{rid}/complete")
def complete(rid: int, u=Depends(current_user)):
    with db() as con:
        q = con.execute("SELECT * FROM requests WHERE request_id=?", (rid,)).fetchone()
        if not q or q["owner_id"] != u["user_id"] or q["status"] != "Accepted": raise HTTPException(400, "Cannot complete")
        con.execute("UPDATE requests SET status='Completed' WHERE request_id=?", (rid,))
        con.execute("INSERT INTO notifications(user_id,message) VALUES(?,?)", (q["seeker_id"], "Exchange completed"))
    return {"ok": True}

@app.get("/api/my/notifications")
def notes(u=Depends(current_user)):
    with db() as con: return rows(con.execute("SELECT * FROM notifications WHERE user_id=? ORDER BY id DESC", (u["user_id"],)))

@app.get("/api/monitor")
def monitor(u=Depends(admin_user)):   # admin only
    with db() as con:
        return rows(con.execute("SELECT * FROM steps ORDER BY step_id DESC LIMIT 60"))

@app.get("/api/impact")
def impact():
    with db() as con:
        r = con.execute("SELECT COUNT(*) n, COALESCE(SUM(requested_quantity),0) u FROM requests WHERE status='Completed'").fetchone()
    return {"completed_exchanges": r["n"], "units_redistributed": r["u"]}

app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/")
def home(): return FileResponse("static/index.html")