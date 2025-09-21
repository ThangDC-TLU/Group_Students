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

# ====== THUẬT TOÁN & CLEANING (đã tách) ======
from grouping import clean_dataframe, chia_nhom_tuong_dong, gpa_tb_nhom

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

# ====== THƯ MỤC BACKUP (TRASH) CHO CLEAR DATA ======
TRASH_DIR = os.path.join(DATA_DIR, 'trash')
os.makedirs(TRASH_DIR, exist_ok=True)

def _trash_file(path: str) -> Optional[str]:
    """
    Di chuyển file sang thư mục TRASH_DIR kèm timestamp để backup an toàn.
    Trả về đường dẫn mới nếu di chuyển thành công, None nếu không tồn tại/empty.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_name = f"{ts}_{os.path.basename(path)}"
    dst = os.path.join(TRASH_DIR, new_name)
    os.replace(path, dst)  # atomic move
    return dst

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

# ====== Export Excel helper ======
def _export_result_xlsx(dataset: str, rows: List[Dict[str, Any]], out_path: str, sheet_name: str = "Groups"):
    """
    Ghi 'rows' (kết quả chia nhóm) ra file Excel .xlsx tại out_path.
    Tự căn độ rộng cột cơ bản.
    """
    if not rows:
        raise ValueError("No rows to export")

    df = pd.DataFrame(rows, columns=RESULT_HEADERS)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)

        # Auto width basic
        try:
            ws = writer.book[sheet_name]
            for col_idx, col_name in enumerate(df.columns, start=1):
                max_len = max(
                    len(str(col_name)),
                    *(len(str(v)) for v in df[col_name].astype(str).values)
                )
                best = max(10, min(50, max_len + 2))
                ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = best
        except Exception:
            pass

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
              version="1.5.0")

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

# ================== GROUPING WRAPPER ==================
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

# ====== DOWNLOAD EXCEL ======
@app.get("/api/admin/group/result.xlsx")
def api_group_result_xlsx(
    x_admin_token: Optional[str] = Header(None),
    dataset: str = Query("data", pattern="^(data|data_h2)$"),
    ca: Optional[str] = Query(None, description="Lọc theo Ca"),
    nhom: Optional[int] = Query(None, description="Lọc theo số Nhóm")
):
    # Auth
    err = require_admin(x_admin_token)
    if err:
        return err

    # Lấy dữ liệu kết quả
    rows = _read_result_csv(dataset)
    if not rows and dataset in LAST_GROUP_RESULT and LAST_GROUP_RESULT[dataset].get('rows'):
        rows = LAST_GROUP_RESULT[dataset]['rows']

    if not rows:
        return JSONResponse(status_code=404, content={'ok': False, 'message': 'Chưa có kết quả để xuất. Hãy chạy chia nhóm trước.'})

    # Lọc
    if ca:
        rows = [r for r in rows if (str(r.get('Ca') or '').lower() == ca.lower())]
    if nhom is not None:
        rows = [r for r in rows if str(r.get('Nhóm')) == str(nhom)]

    if not rows:
        return JSONResponse(status_code=404, content={'ok': False, 'message': 'Không có dòng nào sau khi lọc.'})

    # Tạo file .xlsx
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ket_qua_{dataset}_{ts}.xlsx"
    out_path = os.path.join(DATA_DIR, filename)

    try:
        _export_result_xlsx(dataset, rows, out_path, sheet_name="Groups")
    except Exception as e:
        return JSONResponse(status_code=500, content={'ok': False, 'message': 'Xuất Excel thất bại', 'error': str(e)})

    return FileResponse(
        out_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filename
    )

# ====== CLEAR DATA (backup vào data/trash) ======
@app.post("/api/admin/clear")
def api_admin_clear(
    x_admin_token: Optional[str] = Header(None),
    dataset: str = Query("data", pattern="^(data|data_h2)$"),
    target: str = Query("all", pattern="^(students|result|all)$", description="Xoá file dữ liệu nào"),
    backup: bool = Query(True, description="True: backup sang data/trash trước khi xoá")
):
    """
    Xoá dữ liệu theo dataset:
    - target=students: xoá file CSV dữ liệu sinh viên (data.csv|data_h2.csv)
    - target=result  : xoá file CSV kết quả chia nhóm (chianhom*.csv)
    - target=all     : xoá cả hai
    - backup=True    : di chuyển sang data/trash trước khi xoá
    """
    err = require_admin(x_admin_token)
    if err: return err

    students_csv, result_csv = _dataset_paths(dataset)
    acted = []

    def _clear_one(path: str):
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            acted.append({'file': os.path.basename(path), 'action': 'skip (not found or empty)'})
            return
        if backup:
            new_path = _trash_file(path)
            acted.append({
                'file': os.path.basename(path),
                'action': 'moved to trash' if new_path else 'skip (not found)',
                'trash_path': new_path
            })
        else:
            try:
                os.remove(path)
                acted.append({'file': os.path.basename(path), 'action': 'deleted'})
            except Exception as e:
                acted.append({'file': os.path.basename(path), 'action': f'error: {str(e)}'})

    if target in ('students', 'all'):
        _clear_one(students_csv)
    if target in ('result', 'all'):
        _clear_one(result_csv)
        # Xoá cache RAM nếu xoá result
        LAST_GROUP_RESULT.pop(dataset, None)

    return {'ok': True, 'dataset': dataset, 'target': target, 'backup': backup, 'actions': acted}

# ================== MAIN ==================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=3535, reload=False, workers=1)
