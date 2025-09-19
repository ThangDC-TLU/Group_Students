# -*- coding: utf-8 -*-
import os
import csv
import json
import math
import threading
import queue
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import itertools
import re

from fastapi import FastAPI, Header, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import requests

# ================== CẤU HÌNH ==================
BASE_URL = 'https://sinhvien1.tlu.edu.vn/education'
CLIENT_ID = 'education_client'
CLIENT_SECRET = 'password'

INSECURE_TLS = True
REQUESTS_VERIFY = (not INSECURE_TLS)

ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
TEMPLATES_DIR = os.path.join(ROOT_DIR, 'templates')
STATIC_DIR = os.path.join(ROOT_DIR, 'static')
INDEX_HTML = os.path.join(TEMPLATES_DIR, 'index.html')
ADMIN_HTML = os.path.join(TEMPLATES_DIR, 'admin.html')

# File theo dataset
STUDENTS_FILE = {
    "data":     os.path.join(DATA_DIR, "data.csv"),
    "data_h2":  os.path.join(DATA_DIR, "data_h2.csv"),
}
RESULT_FILE = {
    "data":     os.path.join(DATA_DIR, "chianhom.csv"),
    "data_h2":  os.path.join(DATA_DIR, "chianhom_h2.csv"),
}

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

ADMIN_TOKEN = os.getenv('ADMIN_TOKEN', 'change-me-very-strong')
ADMIN_USER = os.getenv('ADMIN_USER', 'admin')
ADMIN_PASS = os.getenv('ADMIN_PASS', 'admin123')

CSV_HEADERS = ['Timestamp','Username','Mã SV','Tên','Lớp','GPA','Ca học','Công việc IT','Sở thích']
RESULT_HEADERS = ['Mã SV', 'Tên', 'Lớp', 'Ca', 'GPA', 'Nhóm', 'GPA TB Nhóm']

# ====== HÀNG ĐỢI GHI CSV (non-blocking) ======
csv_lock = threading.Lock()
# queue item: {"path": "...csv", "row": {...}}
write_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=2000)

def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return rows
    with open(path, 'r', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def _write_csv_rows(path: str, rows: List[Dict[str, str]]):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader(); writer.writerows(rows)

def _upsert_csv(path: str, rec: Dict[str, Any]):
    """Ghi đè theo Username ở file path"""
    with csv_lock:
        rows = _read_csv_rows(path)
        idx = None
        for i, r in enumerate(rows):
            if r.get('Username','').strip() == str(rec['Username']).strip():
                idx = i; break
        row_str = {k: ('' if rec.get(k) is None else str(rec.get(k))) for k in CSV_HEADERS}
        if idx is None: rows.append(row_str)
        else: rows[idx] = row_str
        _write_csv_rows(path, rows)

def writer_worker():
    while True:
        item = write_q.get()
        try:
            _upsert_csv(item["path"], item["row"])
        except Exception:
            pass
        finally:
            write_q.task_done()

threading.Thread(target=writer_worker, daemon=True).start()

# ====== THROTTLING THEO USERNAME ======
THROTTLE_WINDOW_SEC = 10
_throttle_lock = threading.Lock()
_last_save: Dict[str, float] = {}

def throttled(username: str, window: int = THROTTLE_WINDOW_SEC) -> bool:
    now = time.time()
    with _throttle_lock:
        t = _last_save.get(username, 0)
        if now - t < window:
            return True
        _last_save[username] = now
        return False

# cache kết quả nhóm (theo dataset)
LAST_GROUP_RESULT: Dict[str, Dict[str, Any]] = {}

# ========== Nhãn công việc & sở thích ==========
JOB_LABELS = {
    'backend': 'Backend Developer','frontend': 'Frontend Developer','fullstack': 'Full-stack Developer',
    'mobile': 'Mobile Developer','data-engineer': 'Data Engineer','data-scientist': 'Data Scientist / Analyst',
    'devops': 'DevOps / Cloud Engineer','security': 'Security Engineer','qa': 'QA / Tester',
    'pm': 'Product / Project Manager','uiux': 'UI/UX Designer','other': 'Khác'
}
HOBBY_LABELS = {
    'bong-da':'Bóng đá','cau-long':'Cầu lông','bong-ro':'Bóng rổ','bong-chuyen':'Bóng chuyền','chay-bo':'Chạy bộ',
    'gym':'Gym','ca-hat':'Ca hát','nhac':'Nghe nhạc','nhiep-anh':'Nhiếp ảnh','doc-sach':'Đọc sách','du-lich':'Du lịch',
    'game':'Chơi game','lap-trinh':'Lập trình','sang-tao-noi-dung':'Sáng tạo nội dung','tu-thien':'Tình nguyện','other':'Khác'
}

# ========== Ca học (kỳ 13) ==========
SCHEDULE_PATH = '/api/StudentCourseSubject/studentLoginUser/13'

CODE_TO_CA_64HTTT1 = {
    '251071_CSE414_64HTTT1_1': 'Ca1',
    '251071_CSE414_64HTTT1_2': 'Ca2',
}
CODE_TO_CA_64HTTT2 = {
    '251071_CSE414_64HTTT2_1': 'Ca1',
    '251071_CSE414_64HTTT2_2': 'Ca2',
}

def now_vn_str() -> str:
    dt_utc = datetime.now(timezone.utc)
    dt_vn = dt_utc.astimezone()
    return dt_vn.strftime("%Y-%m-%d %H:%M:%S")

def safe_get(obj: Any, path: str, fallback=None):
    try:
        cur = obj
        for k in path.split('.'):
            if cur is None: return fallback
            cur = cur.get(k) if isinstance(cur, dict) else getattr(cur, k, None)
        return cur if cur is not None else fallback
    except Exception:
        return fallback

def extract_from_payload(payload: Dict[str, Any], fallback_username: Optional[str]):
    name = safe_get(payload, 'student.displayName', 'N/A')
    clazz = safe_get(payload, 'student.enrollmentClass.className', 'N/A')
    code = safe_get(payload, 'student.studentCode', fallback_username or 'N/A')
    gpa = payload.get('learningMark4')
    if not isinstance(gpa, (int, float)):
        for k in ['mark4', 'firstLearningMark4']:
            v = payload.get(k)
            if isinstance(v, (int, float)): gpa = v; break
    return {'name': name, 'clazz': clazz, 'code': code, 'gpa': gpa if isinstance(gpa, (int, float)) else None}

def normalize_job(job: Optional[str], job_other: Optional[str] = None) -> Optional[str]:
    if not job: return None
    if job == 'other':
        extra = (job_other or '').strip()
        return f"Khác: {extra}" if extra else 'Khác'
    return JOB_LABELS.get(job, job)

def normalize_hobbies(hobbies: Any, hobby_other: Optional[str] = None) -> Optional[str]:
    if isinstance(hobbies, list):
        values = hobbies
    elif isinstance(hobbies, str) and hobbies.strip():
        try:
            arr = json.loads(hobbies); values = arr if isinstance(arr, list) else [hobbies]
        except Exception:
            values = [hobbies]
    else:
        values = []
    labels = [HOBBY_LABELS.get(v, v) for v in values]
    if 'other' in values:
        extra = (hobby_other or '').strip()
        for i, lab in enumerate(labels):
            if lab == 'Khác':
                labels[i] = f"Khác: {extra}" if extra else 'Khác'
                break
    return ', '.join(labels) if labels else None

def _dataset_paths(dataset: str):
    """Trả về (students_csv, result_csv). dataset: 'data' hoặc 'data_h2'"""
    ds = 'data_h2' if dataset == 'data_h2' else 'data'
    return STUDENTS_FILE[ds], RESULT_FILE[ds]

def _detect_ca_and_dataset(access_token: str, clazz_hint: str) -> (Optional[str], str):
    """
    Trả về (label_ca, dataset_key). dataset_key in {'data','data_h2'}.
    Ưu tiên theo mã lớp trong thời khoá biểu; nếu không chắc, dùng clazz_hint.
    """
    try:
        url = f"{BASE_URL}{SCHEDULE_PATH}"
        resp = requests.get(url, headers={'Authorization': f'Bearer {access_token}'}, timeout=20, verify=REQUESTS_VERIFY)
        if resp.status_code >= 400:
            # fallback theo lớp
            if '64HTTT2' in (clazz_hint or ''): return None, 'data_h2'
            return None, 'data'
        data = resp.json()
        items = data if isinstance(data, list) else data.get('data', [])
        labels = set()
        got_h2 = False
        got_h1 = False
        for it in items:
            code = (safe_get(it, 'courseSubject.code','') or safe_get(it,'classCode','') or safe_get(it,'code','')).strip()
            if code in CODE_TO_CA_64HTTT2:
                labels.add(CODE_TO_CA_64HTTT2[code]); got_h2 = True
            if code in CODE_TO_CA_64HTTT1:
                labels.add(CODE_TO_CA_64HTTT1[code]); got_h1 = True
        # chọn dataset
        dataset = 'data_h2' if got_h2 else ('data' if got_h1 else ('data_h2' if '64HTTT2' in (clazz_hint or '') else 'data'))
        ca_label = ', '.join(sorted(labels)) if labels else None
        return ca_label, dataset
    except Exception:
        # fallback theo lớp
        if '64HTTT2' in (clazz_hint or ''): return None, 'data_h2'
        return None, 'data'

def _write_result_csv(dataset: str, rows: List[Dict[str, Any]]):
    _, result_path = _dataset_paths(dataset)
    with open(result_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=RESULT_HEADERS, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in rows:
            w.writerow({k: ('' if r.get(k) is None else r.get(k)) for k in RESULT_HEADERS})

def _read_result_csv(dataset: str) -> List[Dict[str, Any]]:
    _, result_path = _dataset_paths(dataset)
    if not os.path.exists(result_path) or os.path.getsize(result_path) == 0:
        return []
    out: List[Dict[str, Any]] = []
    with open(result_path, 'r', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rr: Dict[str, Any] = dict(r)
            rr['Nhóm'] = int(float(rr['Nhóm'])) if rr.get('Nhóm') not in (None,'','NaN') else ''
            rr['GPA'] = float(rr['GPA']) if rr.get('GPA') not in (None,'','NaN') else ''
            rr['GPA TB Nhóm'] = float(rr['GPA TB Nhóm']) if rr.get('GPA TB Nhóm') not in (None,'','NaN') else ''
            out.append(rr)
    return out

# ================== SCHEMAS ==================
class SaveGPAIn(BaseModel):
    username: str
    password: str
    job: Optional[str] = None
    jobOther: Optional[str] = None
    hobbies: Optional[List[str]] = None
    hobbyOther: Optional[str] = None

class AdminLoginIn(BaseModel):
    username: str
    password: str

# ================== PAGES ==================
app = FastAPI(title="Lưu GPA (FastAPI)",
              description="Lưu theo dataset (64HTTT1/64HTTT2), chia nhóm và phân trang.",
              version="1.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=FileResponse)
def home():
    return FileResponse(INDEX_HTML, media_type="text/html")

@app.get("/admin", response_class=FileResponse)
def admin_page():
    return FileResponse(ADMIN_HTML, media_type="text/html")

# ================== STUDENT FLOW ==================
@app.post("/api/gpa/save")
def save_gpa(payload: SaveGPAIn):
    if throttled(payload.username):
        return {'ok': True, 'message': 'Đã nhận yêu cầu gần đây. Vui lòng thử lại sau vài giây.'}

    # 1) Token
    try:
        form = {'grant_type':'password','client_id':CLIENT_ID,'client_secret':CLIENT_SECRET,
                'username':payload.username,'password':payload.password}
        resp = requests.post(f"{BASE_URL}/oauth/token",
                             data=form, headers={'Content-Type':'application/x-www-form-urlencoded'},
                             timeout=15, verify=REQUESTS_VERIFY)
        if resp.status_code >= 400:
            return JSONResponse(status_code=resp.status_code,
                                content={'message':'Tài khoản mật khẩu không chính xác','detail':resp.text})
        tok = resp.json().get('access_token')
        if not tok:
            return JSONResponse(status_code=500, content={'message':'Không nhận được access_token'})
    except requests.exceptions.Timeout:
        return JSONResponse(status_code=504, content={'message':'Token API timeout'})
    except Exception as e:
        return JSONResponse(status_code=500, content={'message':'Lỗi lấy token','error':str(e)})

    # 2) GPA
    try:
        resp = requests.get(f"{BASE_URL}/api/studentsummarymark/getbystudent",
                            headers={'Authorization': f'Bearer {tok}'},
                            timeout=15, verify=REQUESTS_VERIFY)
        if resp.status_code >= 400:
            return JSONResponse(status_code=resp.status_code,
                                content={'message':'Gọi API GPA thất bại','detail':resp.text})
        payload_gpa = resp.json()
    except requests.exceptions.Timeout:
        return JSONResponse(status_code=504, content={'message':'GPA API timeout'})
    except Exception as e:
        return JSONResponse(status_code=500, content={'message':'Lỗi gọi GPA','error':str(e)})

    # 3) Parse
    info = extract_from_payload(payload_gpa, payload.username)

    # 4) Ca + dataset
    ca_hoc, dataset = _detect_ca_and_dataset(access_token=tok, clazz_hint=info['clazz'])
    students_csv, _ = _dataset_paths(dataset)

    # 5) Chuẩn hoá job/hobby
    job_text = normalize_job(payload.job, payload.jobOther)
    hobbies_text = normalize_hobbies(payload.hobbies, payload.hobbyOther)

    # 6) Hàng đợi ghi
    rec = {
        'Timestamp': now_vn_str(),
        'Username': payload.username,
        'Mã SV': info['code'],
        'Tên': info['name'],
        'Lớp': info['clazz'],
        'GPA': info['gpa'] if info['gpa'] is not None else '',
        'Ca học': ca_hoc or '',
        'Công việc IT': job_text or '',
        'Sở thích': hobbies_text or '',
    }
    try:
        write_q.put_nowait({"path": students_csv, "row": rec})
    except queue.Full:
        try:
            _upsert_csv(students_csv, rec)
        except Exception as e:
            return JSONResponse(status_code=500, content={'message':'Lỗi ghi CSV','error':str(e)})

    return {'ok': True, 'message': f'Đã lưu vào {os.path.basename(students_csv)}.'}

# ================== ADMIN AUTH ==================
@app.post("/api/admin/login")
def admin_login(body: AdminLoginIn):
    if body.username == ADMIN_USER and body.password == ADMIN_PASS:
        return {'ok': True, 'adminToken': ADMIN_TOKEN}
    return JSONResponse(status_code=401, content={'ok': False, 'message': 'Đăng nhập admin thất bại.'})

def require_admin(token: Optional[str]) -> Optional[JSONResponse]:
    if token != ADMIN_TOKEN:
        return JSONResponse(status_code=401, content={'message':'Unauthorized'})
    return None

# ================== CLEAN & GROUPING UTILS ==================
CANON_COLS = ['Mã SV','Tên','Lớp','GPA','Ca học','Công việc IT','Sở thích']
RENAME_MAP = {
    'Mã SV': 'Mã SV','Ma SV':'Mã SV','MaSV':'Mã SV','MãSV':'Mã SV',
    'Tên':'Tên','Ten':'Tên','Họ tên':'Tên','Ho ten':'Tên','Ho_ten':'Tên',
    'Lớp':'Lớp','Lop':'Lớp','Lớp ':'Lớp',
    'GPA':'GPA','Điểm GPA':'GPA',
    'Ca học':'Ca học','Ca':'Ca học','Ca ':'Ca học','Ca học ':'Ca học',
    'Công việc IT':'Công việc IT','Cong viec IT':'Công việc IT','Cong viec':'Công việc IT',
    'Sở thích':'Sở thích','So thich':'Sở thích','So_thich':'Sở thích'
}

def _to_float_gpa(x):
    if pd.isna(x): return None
    if isinstance(x, (int, float)): return float(x)
    s = str(x).strip()
    if not s: return None
    s = s.replace(',', '.'); s = re.sub(r'[^0-9\.\-]+','', s)
    try: return float(s)
    except: return None

def _clean_text(x):
    if pd.isna(x): return None
    s = str(x).strip()
    return s if s else None

def _split_hobbies(x: str) -> List[str]:
    if not isinstance(x, str): return []
    tmp = x.replace('|', ',').replace(';', ',')
    parts = [p.strip() for p in tmp.split(',')]
    return [p for p in parts if p]

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c: RENAME_MAP.get(c, c) for c in df.columns}
    df = df.rename(columns=cols)
    for c in CANON_COLS:
        if c not in df.columns: df[c] = None
    df = df[CANON_COLS].copy()

    df['Mã SV'] = df['Mã SV'].apply(_clean_text)
    df['Tên'] = df['Tên'].apply(_clean_text)
    df['Lớp'] = df['Lớp'].apply(_clean_text)
    df['Ca học'] = df['Ca học'].apply(_clean_text)
    df['Công việc IT'] = df['Công việc IT'].apply(_clean_text)
    df['Sở thích'] = df['Sở thích'].apply(_clean_text)
    df['GPA'] = df['GPA'].apply(_to_float_gpa)

    df = df[~df['Ca học'].isna() & (df['Ca học'].str.strip() != '')]
    df = df[~df['Tên'].isna() & (df['Tên'].str.strip() != '')]
    df = df.reset_index(drop=True)
    return df

def tinh_diem_tuong_dong(sv1, sv2):
    g1 = sv1.get('GPA'); g2 = sv2.get('GPA')
    gpa_sv1 = (g1/4.0) if g1 is not None else 0.0
    gpa_sv2 = (g2/4.0) if g2 is not None else 0.0
    diem_gpa = 1.0 - abs(gpa_sv1 - gpa_sv2)

    cong_viec_chung = 0
    if sv1.get('Công việc IT') and sv2.get('Công việc IT') and sv1['Công việc IT'] == sv2['Công việc IT']:
        cong_viec_chung = 1

    so_thich_chung = 0
    if sv1.get('Sở thích') and sv2.get('Sở thích'):
        so_thich1 = set(map(str.lower, _split_hobbies(sv1['Sở thích'])))
        so_thich2 = set(map(str.lower, _split_hobbies(sv2['Sở thích'])))
        so_thich_chung = len(so_thich1 & so_thich2)

    w_gpa, w_cong_viec, w_so_thich = 0.4, 0.3, 0.3
    return w_gpa*diem_gpa + w_cong_viec*cong_viec_chung + w_so_thich*so_thich_chung

def gpa_tb_nhom(nhom):
    gpas = [sv.get('GPA') for sv in nhom if sv.get('GPA') is not None]
    return (sum(gpas)/len(gpas)) if gpas else 0.0

def tinh_diem_trung_binh_nhom(nhom):
    if not nhom or len(nhom) < 2: return 0.0
    tong = 0.0; so_cap = 0
    for sv1, sv2 in itertools.combinations(nhom, 2):
        tong += tinh_diem_tuong_dong(sv1, sv2); so_cap += 1
    return (tong/so_cap) if so_cap>0 else 0.0

def chia_nhom_tuong_dong(dssv, so_nhom_mong_muon_moi_ca=None, so_luong_moi_nhom=None):
    from collections import defaultdict
    sinh_vien_theo_ca = defaultdict(list)
    for sv in dssv:
        sinh_vien_theo_ca[sv['Ca học']].append(sv)

    ket_qua = {}
    for ca, lst in sinh_vien_theo_ca.items():
        if not lst: continue
        lst.sort(key=lambda x: x['GPA'] if x['GPA'] is not None else 0.0, reverse=True)

        if so_luong_moi_nhom and so_luong_moi_nhom > 0:
            num_groups = max(1, math.ceil(len(lst) / so_luong_moi_nhom))
        else:
            base = so_nhom_mong_muon_moi_ca if so_nhom_mong_muon_moi_ca and so_nhom_mong_muon_moi_ca>0 else 3
            num_groups = min(len(lst), base) if len(lst) >= base else max(1, len(lst)//2 or 1)

        groups = [[] for _ in range(num_groups)]
        # snake distribution
        for i, sv in enumerate(lst):
            idx = i % num_groups
            if (i // num_groups) % 2 == 1: idx = num_groups-1-idx
            groups[idx].append(sv)

        # tối ưu tương đồng
        for _ in range(60):
            changed = False
            for i1, i2 in itertools.combinations(range(num_groups), 2):
                n1, n2 = groups[i1], groups[i2]
                for a, sv1 in enumerate(n1):
                    for b, sv2 in enumerate(n2):
                        before = tinh_diem_trung_binh_nhom(n1) + tinh_diem_trung_binh_nhom(n2)
                        n1m = n1[:a]+n1[a+1:]+[sv2]; n2m = n2[:b]+n2[b+1:]+[sv1]
                        after = tinh_diem_trung_binh_nhom(n1m) + tinh_diem_trung_binh_nhom(n2m)
                        if after > before:
                            d_tr = gpa_tb_nhom(n1) - gpa_tb_nhom(n2)
                            d_sa = gpa_tb_nhom(n1m) - gpa_tb_nhom(n2m)
                            if abs(d_sa - d_tr) < 0.25:
                                groups[i1], groups[i2] = n1m, n2m
                                changed = True
                                break
                    if changed: break
                if changed: break
            if not changed: break

        # cân bằng GPA TB theo toàn ca
        gpas_ca = [sv.get('GPA') for sv in lst if sv.get('GPA') is not None]
        gpa_toan_ca = (sum(gpas_ca)/len(gpas_ca)) if gpas_ca else 0.0
        eps = 1e-9
        for _ in range(120):
            improved = False
            for i1, i2 in itertools.combinations(range(num_groups), 2):
                n1, n2 = groups[i1], groups[i2]
                best_delta = 0.0; best_pair = None
                var_before = ((gpa_tb_nhom(n1)-gpa_toan_ca)**2 + (gpa_tb_nhom(n2)-gpa_toan_ca)**2)
                for a, sv1 in enumerate(n1):
                    for b, sv2 in enumerate(n2):
                        n1m = n1[:a]+n1[a+1:]+[sv2]; n2m = n2[:b]+n2[b+1:]+[sv1]
                        var_after = ((gpa_tb_nhom(n1m)-gpa_toan_ca)**2 + (gpa_tb_nhom(n2m)-gpa_toan_ca)**2)
                        delta = var_before - var_after
                        if delta > best_delta + eps:
                            best_delta = delta; best_pair = (a, b)
                if best_pair is not None:
                    a, b = best_pair
                    n1[a], n2[b] = n2[b], n1[a]
                    improved = True
            if not improved: break

        ket_qua[ca] = groups
    return ket_qua

def _run_grouping_for_dataset(dataset: str,
                              group_size: Optional[int],
                              groups_per_ca: Optional[int]) -> Dict[str, Any]:
    students_csv, _ = _dataset_paths(dataset)
    if not os.path.exists(students_csv) or os.path.getsize(students_csv) == 0:
        return {'ok': False, 'message': f'Chưa có dữ liệu sinh viên ({os.path.basename(students_csv)}).'}

    with csv_lock:
        df_raw = pd.read_csv(students_csv, dtype=str, encoding='utf-8', engine='python')
    df = clean_dataframe(df_raw)
    if df.empty:
        return {'ok': False, 'message': 'Dữ liệu sau khi làm sạch rỗng.'}

    dssv = df.to_dict('records')
    result = chia_nhom_tuong_dong(
        dssv,
        so_nhom_mong_muon_moi_ca=groups_per_ca,
        so_luong_moi_nhom=group_size
    )

    rows: List[Dict[str, Any]] = []
    for ca, groups in result.items():
        for idx, nhom in enumerate(groups, start=1):
            gpa_tb = round(gpa_tb_nhom(nhom), 2)
            for sv in nhom:
                rows.append({
                    'Mã SV': sv.get('Mã SV') or '',
                    'Tên': sv.get('Tên') or '',
                    'Lớp': sv.get('Lớp') or '',
                    'Ca': ca,
                    'GPA': (round(float(sv.get('GPA')), 2) if sv.get('GPA') not in (None, '') else ''),
                    'Nhóm': idx,
                    'GPA TB Nhóm': gpa_tb
                })

    if not rows:
        return {'ok': False, 'message': 'Không tạo được kết quả chia nhóm.'}

    _write_result_csv(dataset, rows)
    LAST_GROUP_RESULT[dataset] = {'ok': True, 'rows': rows}
    return LAST_GROUP_RESULT[dataset]

# ================== ROUTES: ADMIN DATA (PHÂN TRANG + DATASET) ==================
@app.get("/api/admin/students")
def admin_students(
    x_admin_token: Optional[str] = Header(None),
    dataset: str = Query("data", pattern="^(data|data_h2)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    q: Optional[str] = Query(None, description="Từ khoá lọc (tuỳ chọn)")
):
    err = require_admin(x_admin_token)
    if err: return err

    students_csv, _ = _dataset_paths(dataset)
    with csv_lock:
        rows = _read_csv_rows(students_csv)

    if q:
        ql = q.lower()
        keys = ['Username','Mã SV','Tên','Lớp','Ca học','Công việc IT','Sở thích']
        rows = [r for r in rows if any(ql in (r.get(k,'').lower()) for k in keys)]

    from datetime import datetime as _dt
    def parse_ts(s):
        try:
            return _dt.strptime(s, "%Y-%m-%d %H:%M:%S")
        except:
            return _dt.min
    rows.sort(key=lambda r: parse_ts(r.get('Timestamp','')), reverse=True)

    total = len(rows)
    start = (page - 1) * page_size
    end = start + page_size
    page_rows = rows[start:end]
    total_pages = max(1, (total + page_size - 1) // page_size)

    return {
        'ok': True, 'rows': page_rows, 'page': page, 'page_size': page_size,
        'total': total, 'total_pages': total_pages,
        'file': os.path.basename(students_csv)
    }

@app.post("/api/admin/group/run")
def api_group_run(
    x_admin_token: Optional[str] = Header(None),
    dataset: str = Query("data", pattern="^(data|data_h2)$"),
    group_size: Optional[int] = Query(None, ge=2, description="Số người mỗi nhóm (ưu tiên)"),
    groups_per_ca: Optional[int] = Query(None, ge=1, description="Số nhóm mỗi ca (nếu không đặt group_size)")
):
    err = require_admin(x_admin_token)
    if err: return err
    res = _run_grouping_for_dataset(dataset, group_size, groups_per_ca)
    return JSONResponse(status_code=200 if res.get('ok') else 400, content=res)

@app.get("/api/admin/group/result")
def api_group_result(
    x_admin_token: Optional[str] = Header(None),
    dataset: str = Query("data", pattern="^(data|data_h2)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    ca: Optional[str] = Query(None, description="Lọc theo Ca"),
    nhom: Optional[int] = Query(None, description="Lọc theo số Nhóm")
):
    err = require_admin(x_admin_token)
    if err: return err

    rows = _read_result_csv(dataset)
    if not rows:
        if dataset in LAST_GROUP_RESULT and LAST_GROUP_RESULT[dataset].get('rows'):
            rows = LAST_GROUP_RESULT[dataset]['rows']
        else:
            return JSONResponse(status_code=404, content={'ok': False, 'message': 'Chưa có kết quả. Hãy chạy chia nhóm trước.'})

    if ca:
        rows = [r for r in rows if (str(r.get('Ca') or '').lower() == ca.lower())]
    if nhom is not None:
        rows = [r for r in rows if str(r.get('Nhóm')) == str(nhom)]

    total = len(rows)
    start = (page - 1) * page_size
    end = start + page_size
    page_rows = rows[start:end]
    total_pages = max(1, (total + page_size - 1) // page_size)

    return {
        'ok': True, 'rows': page_rows, 'page': page, 'page_size': page_size,
        'total': total, 'total_pages': total_pages,
        'file': os.path.basename(_dataset_paths(dataset)[1])
    }

# ================== MAIN ==================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=3535, reload=False, workers=1)
