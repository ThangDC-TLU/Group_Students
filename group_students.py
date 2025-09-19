# -*- coding: utf-8 -*-
import argparse
import math
import pandas as pd
from collections import defaultdict
import itertools
import re
from typing import Optional


CANON_COLS = ['Mã SV', 'Tên', 'Lớp', 'GPA', 'Ca học', 'Công việc IT', 'Sở thích']

# các tên cột có thể gặp -> tên chuẩn
RENAME_MAP = {
    'Mã SV': 'Mã SV', 'Ma SV': 'Mã SV', 'MaSV': 'Mã SV', 'MãSV': 'Mã SV',
    'Tên': 'Tên', 'Ten': 'Tên', 'Họ tên': 'Tên', 'Ho ten': 'Tên', 'Ho_ten': 'Tên',
    'Lớp': 'Lớp', 'Lop': 'Lớp', 'Lớp ': 'Lớp',
    'GPA': 'GPA', 'Điểm GPA': 'GPA',
    'Ca học': 'Ca học', 'Ca': 'Ca học', 'Ca ': 'Ca học', 'Ca học ': 'Ca học',
    'Công việc IT': 'Công việc IT', 'Cong viec IT': 'Công việc IT', 'Cong viec': 'Công việc IT',
    'Sở thích': 'Sở thích', 'So thich': 'Sở thích', 'So_thich': 'Sở thích'
}

def _to_float_gpa(x):
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return None
    # đổi , -> . ; bỏ ký tự không phải số/chấm/trừ
    s = s.replace(',', '.')
    s = re.sub(r'[^0-9\.\-]+', '', s)
    try:
        v = float(s)
        # nếu nhập ngoài thang 0..4 vẫn giữ nguyên theo yêu cầu (chỉ chuyển số)
        return v
    except:
        return None

def _clean_text(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s if s else None

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Chuẩn hóa tên cột
    cols = {c: RENAME_MAP.get(c, c) for c in df.columns}
    df = df.rename(columns=cols)

    # 2) Chỉ giữ đúng 7 cột yêu cầu (có cột nào thiếu thì thêm cột rỗng)
    for c in CANON_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[CANON_COLS].copy()

    # 3) Làm sạch từng cột
    df['Mã SV']       = df['Mã SV'].apply(_clean_text)
    df['Tên']         = df['Tên'].apply(_clean_text)
    df['Lớp']         = df['Lớp'].apply(_clean_text)
    df['Ca học']      = df['Ca học'].apply(_clean_text)
    df['Công việc IT']= df['Công việc IT'].apply(_clean_text)
    df['Sở thích']    = df['Sở thích'].apply(_clean_text)
    df['GPA']         = df['GPA'].apply(_to_float_gpa)

    # 4) Loại bỏ hàng có Ca học rỗng/null (theo yêu cầu)
    df = df[~df['Ca học'].isna() & (df['Ca học'].str.strip() != '')]

    # loại bỏ hàng không có Tên
    df = df[~df['Tên'].isna() & (df['Tên'].str.strip() != '')]

    df = df.reset_index(drop=True)
    return df

def tinh_diem_tuong_dong(sv1, sv2):
    g1 = sv1.get('GPA'); g2 = sv2.get('GPA')
    gpa_sv1 = (g1 / 4.0) if g1 is not None else 0.0
    gpa_sv2 = (g2 / 4.0) if g2 is not None else 0.0
    diem_gpa = 1.0 - abs(gpa_sv1 - gpa_sv2)

    cong_viec_chung = 0
    if sv1.get('Công việc IT') and sv2.get('Công việc IT') and sv1['Công việc IT'] == sv2['Công việc IT']:
        cong_viec_chung = 1

    so_thich_chung = 0
    if sv1.get('Sở thích') and sv2.get('Sở thích'):
        def norm(x):
            if isinstance(x, str):
                raw = []
                # chuẩn hóa các dấu phân tách , ; | . xuống thành dấu phẩy
                tmp = x.replace('|', ',').replace(';', ',').replace('.', ',')
                raw = [p.strip() for p in tmp.split(',')]
                return [t for t in raw if t]
            return list(x)
        so_thich1_list = norm(sv1['Sở thích'])
        so_thich2_list = norm(sv2['Sở thích'])
        so_thich_chung = len(set(map(str.lower, so_thich1_list)) & set(map(str.lower, so_thich2_list)))

    w_gpa, w_cong_viec, w_so_thich = 0.4, 0.3, 0.3
    return w_gpa * diem_gpa + w_cong_viec * cong_viec_chung + w_so_thich * so_thich_chung

def gpa_tb_nhom(nhom):
    gpas = [sv.get('GPA') for sv in nhom if sv.get('GPA') is not None]
    return (sum(gpas)/len(gpas)) if gpas else 0.0

def tinh_diem_trung_binh_nhom(nhom):
    if not nhom or len(nhom) < 2: return 0.0
    tong = 0.0; so_cap = 0
    for sv1, sv2 in itertools.combinations(nhom, 2):
        tong += tinh_diem_tuong_dong(sv1, sv2); so_cap += 1
    return (tong / so_cap) if so_cap > 0 else 0.0

def phuong_sai_gpa(cac_nhom, gpa_toan_ca):
    return sum((gpa_tb_nhom(n) - gpa_toan_ca) ** 2 for n in cac_nhom)

def chia_nhom_tuong_dong(danh_sach_sinh_vien, so_nhom_mong_muon_moi_ca=None, so_luong_moi_nhom=None):
    sinh_vien_theo_ca = defaultdict(list)
    for sv in danh_sach_sinh_vien:
        sinh_vien_theo_ca[sv['Ca học']].append(sv)

    ket_qua_chia_nhom = {}

    for ca, dssv in sinh_vien_theo_ca.items():
        if not dssv: continue

        dssv.sort(key=lambda x: x['GPA'] if x['GPA'] is not None else 0.0, reverse=True)

        if so_luong_moi_nhom and so_luong_moi_nhom > 0:
            num_groups = max(1, math.ceil(len(dssv) / so_luong_moi_nhom))
        else:
            base = so_nhom_mong_muon_moi_ca if so_nhom_mong_muon_moi_ca and so_nhom_mong_muon_moi_ca > 0 else 3
            num_groups = min(len(dssv), base) if len(dssv) >= base else max(1, len(dssv)//2 or 1)

        cac_nhom = [[] for _ in range(num_groups)]

        # phân bổ snake
        for i, sv in enumerate(dssv):
            idx = i % num_groups
            if (i // num_groups) % 2 == 1:
                idx = num_groups - 1 - idx
            cac_nhom[idx].append(sv)

        # tối ưu tương đồng nội nhóm
        max_iterations = 60
        for _ in range(max_iterations):
            changed = False
            for i1, i2 in itertools.combinations(range(num_groups), 2):
                nhom1, nhom2 = cac_nhom[i1], cac_nhom[i2]
                for a, sv1 in enumerate(nhom1):
                    for b, sv2 in enumerate(nhom2):
                        diem_truoc = tinh_diem_trung_binh_nhom(nhom1) + tinh_diem_trung_binh_nhom(nhom2)
                        n1m = nhom1[:a] + nhom1[a+1:] + [sv2]
                        n2m = nhom2[:b] + nhom2[b+1:] + [sv1]
                        diem_sau = tinh_diem_trung_binh_nhom(n1m) + tinh_diem_trung_binh_nhom(n2m)

                        if diem_sau > diem_truoc:
                            d_truoc = gpa_tb_nhom(nhom1) - gpa_tb_nhom(nhom2)
                            d_sau = gpa_tb_nhom(n1m) - gpa_tb_nhom(n2m)
                            if abs(d_sau - d_truoc) < 0.25:
                                cac_nhom[i1], cac_nhom[i2] = n1m, n2m
                                changed = True
                                break
                    if changed: break
                if changed: break
            if not changed: break

        # tối ưu cân bằng GPA TB giữa các nhóm
        gpas_ca = [sv.get('GPA') for sv in dssv if sv.get('GPA') is not None]
        gpa_toan_ca = (sum(gpas_ca) / len(gpas_ca)) if gpas_ca else 0.0

        max_iterations_var = 120
        eps = 1e-9
        for _ in range(max_iterations_var):
            improved = False
            for i1, i2 in itertools.combinations(range(num_groups), 2):
                nhom1, nhom2 = cac_nhom[i1], cac_nhom[i2]
                best_delta = 0.0; best_pair = None
                var_before = ((gpa_tb_nhom(nhom1) - gpa_toan_ca) ** 2
                              + (gpa_tb_nhom(nhom2) - gpa_toan_ca) ** 2)
                for a, sv1 in enumerate(nhom1):
                    for b, sv2 in enumerate(nhom2):
                        n1m = nhom1[:a] + nhom1[a+1:] + [sv2]
                        n2m = nhom2[:b] + nhom2[b+1:] + [sv1]
                        var_after = ((gpa_tb_nhom(n1m) - gpa_toan_ca) ** 2
                                     + (gpa_tb_nhom(n2m) - gpa_toan_ca) ** 2)
                        delta = var_before - var_after
                        if delta > best_delta + eps:
                            best_delta = delta; best_pair = (a, b)
                if best_pair is not None:
                    a, b = best_pair
                    nhom1[a], nhom2[b] = nhom2[b], nhom1[a]
                    improved = True
            if not improved: break

        ket_qua_chia_nhom[ca] = cac_nhom

    return ket_qua_chia_nhom

def main():
    parser = argparse.ArgumentParser(description="Đọc gpa.xlsx và xuất chia_nhom.xlsx (kèm GPA TB Nhóm).")
    parser.add_argument("-i", "--input", type=str, default="gpa.xlsx", help="File Excel đầu vào (mặc định: gpa.xlsx)")
    parser.add_argument("-o", "--output", type=str, default="chia_nhom.xlsx", help="File Excel đầu ra (mặc định: chia_nhom.xlsx)")
    parser.add_argument("-g", "--groups-per-ca", type=int, default=None, help="Số nhóm mong muốn mỗi ca (tùy chọn)")
    parser.add_argument("-s", "--group-size", type=int, default=3, help="Số lượng thành viên mỗi nhóm (mặc định: 3)")
    args = parser.parse_args()

    # 1) Đọc dữ liệu
    try:
        df_raw = pd.read_excel(args.input, engine="openpyxl")
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file '{args.input}'.")
        return

    # 2) Làm sạch & chỉ giữ 7 cột, loại bỏ hàng Ca học rỗng/null
    df = clean_dataframe(df_raw)

    if df.empty:
        print("❌ Không còn dữ liệu hợp lệ sau khi làm sạch (mọi hàng Ca học rỗng/null đã bị loại).")
        return

    # 3) Chuyển records
    dssv = df.to_dict('records')

    # 4) Chia nhóm (ưu tiên --group-size; nếu không đặt thì dùng --groups-per-ca)
    ket_qua = chia_nhom_tuong_dong(
        dssv,
        so_nhom_mong_muon_moi_ca=args.groups_per_ca,
        so_luong_moi_nhom=args.group_size
    )

    # 5) Build rows + tính GPA TB Nhóm
    rows, thong_ke = [], []
    for ca, cac_nhom in ket_qua.items():
        for idx, nhom in enumerate(cac_nhom, start=1):
            gpa_tb = round(gpa_tb_nhom(nhom), 2)
            thong_ke.append({'Ca': ca, 'Nhóm': idx, 'Số lượng': len(nhom), 'GPA TB Nhóm': gpa_tb})
            for sv in nhom:
                rows.append({
                    'Mã SV': sv.get('Mã SV'),
                    'Tên': sv.get('Tên'),
                    'Lớp': sv.get('Lớp'),
                    'Ca': ca,
                    'GPA': sv.get('GPA'),
                    'Nhóm': idx,
                    'GPA TB Nhóm': gpa_tb
                })

    if not rows:
        print("❌ Không tạo được kết quả chia nhóm.")
        return

    df_out = pd.DataFrame(rows, columns=['Mã SV', 'Tên', 'Lớp', 'Ca', 'GPA', 'Nhóm', 'GPA TB Nhóm'])
    df_summary = pd.DataFrame(thong_ke, columns=['Ca', 'Nhóm', 'Số lượng', 'GPA TB Nhóm'])

    # 6) Xuất Excel
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="ChiaNhom", index=False)
        df_summary.to_excel(writer, sheet_name="ThongKe", index=False)

    print(f"✅ Đã xuất '{args.output}' với sheet 'ChiaNhom' và 'ThongKe'. Hàng thiếu 'Ca học' đã bị loại.")

# ==== THÊM PHẦN NÀY Ở CUỐI FILE ====
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io

app = FastAPI(title="Chia nhóm sinh viên", version="1.0.0")

# Bật CORS nếu bạn sẽ gọi từ front-end khác cổng
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # chỉnh domain nếu cần
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def home():
    return r"""
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Chia nhóm sinh viên</title>
  <style>
    :root { color-scheme: light dark; }
    html,body { height:100%; }
    body {
      margin:0; font-family: system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      background:#0b1020; color:#e9edf5;
      display:flex; align-items:center; justify-content:center; padding:24px;
    }
    .card {
      width:100%; max-width:720px; background:#0f172a;
      border:1px solid #1f2a44; border-radius:20px; padding:24px;
      box-shadow:0 16px 40px rgba(0,0,0,.45);
    }
    .header { display:flex; align-items:center; gap:12px; margin-bottom:16px; }
    .logo {
      width:40px; height:40px; border-radius:12px;
      background:linear-gradient(135deg,#60a5fa,#22d3ee);
      display:grid; place-items:center; font-weight:900; color:#0b1020;
    }
    h1 { margin:0; font-size:1.4rem; }
    p.muted { margin:.25rem 0 0 0; color:#98a2b3; font-size:.95rem; }

    .pane { background:#0b1224; border:1px dashed #23304f; border-radius:16px; padding:18px; }

    .drop {
      border:2px dashed #334155; border-radius:16px; padding:18px; text-align:center;
      background:#0c1326; transition:.2s border-color,.2s background;
    }
    .drop.dragover { border-color:#60a5fa; background:#0a142e; }
    .drop input { display:none; }
    .hint { color:#9aa7bd; font-size:.9rem; margin-top:8px; }

    /* --- Cụm 2 trường tham số --- */
    .fields {
      display:grid;
      grid-template-columns: 1fr;         /* mobile: 1 cột */
      row-gap: 14px;
      column-gap: 24px;                    /* tăng khoảng cách 2 cột */
      align-items:end;                     /* đáy label & input thẳng hàng */
      margin-top:14px;
    }
    @media (min-width: 560px) {
      .fields { grid-template-columns: auto auto; }  /* desktop: 2 cột gọn theo nội dung */
    }

    label { display:block; font-size:.92rem; color:#c9d2e2; margin-bottom:6px; }

    .input {
      width:100%; padding:10px 12px; border-radius:12px; border:1px solid #23304f;
      background:#0b1224; color:#e9edf5; outline:none; font-size:1rem;
    }
    .input:focus { border-color:#60a5fa; }

    /* Ô nhập số: nhỏ gọn, không full-width */
    .input.num {
      width: clamp(132px, 40vw, 200px);   /* nhỏ lại vì chỉ nhập số */
      text-align: center;
    }

    /* Ẩn spinner của number để đồng nhất UI */
    input[type="number"]::-webkit-outer-spin-button,
    input[type="number"]::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
    input[type="number"] { -moz-appearance: textfield; appearance: textfield; }

    ::placeholder { color:#8fa3c2; opacity:.9; }

    .btn {
      display:inline-flex; align-items:center; justify-content:center; gap:8px;
      padding:12px 16px; border:0; border-radius:12px; cursor:pointer; width:100%;
      background:linear-gradient(135deg,#2563eb,#06b6d4); color:#fff; font-weight:700;
      margin-top:14px; transition:transform .04s ease;
    }
    .btn:active { transform: translateY(1px); }
    .btn[disabled] { opacity:.6; cursor:not-allowed; }

    .progress { height:10px; width:100%; background:#101933; border-radius:999px; overflow:hidden; margin-top:10px; display:none; }
    .bar { height:100%; width:0%; background:linear-gradient(90deg,#22d3ee,#60a5fa); transition:width .2s; }

    .alert { padding:10px 12px; border-radius:12px; font-size:.92rem; margin-top:12px; }
    .alert.err { background:#2a1020; border:1px solid #692344; color:#ffb4c8; }
    .alert.ok  { background:#10231e; border:1px solid #1f7a5d; color:#b5f3dc; }
  </style>
</head>
<body>
  <div class="card">
    <div class="header">
      <div class="logo">CN</div>
      <div>
        <h1>Chia nhóm sinh viên</h1>
        <p class="muted">Upload Excel → chọn tham số → tải <em>chia_nhom.xlsx</em>.</p>
      </div>
    </div>

    <div class="pane">
      <div id="drop" class="drop">
        <strong>Kéo-thả</strong> file <span style="font-family:ui-monospace">.xlsx</span> vào đây<br/>
        hoặc <label for="file" style="display:inline;color:#7dd3fc;cursor:pointer;text-decoration:underline">chọn file</label>
        <input id="file" type="file" accept=".xlsx" />
      </div>

      <div class="fields">
        <div>
          <label for="group_size">Kích thước nhóm</label>
          <input id="group_size" class="input num" type="number" min="2" value="3" />
        </div>
        <div>
          <label for="groups_per_ca">Số nhóm mỗi ca (tùy chọn)</label>
          <input id="groups_per_ca" class="input num" type="number" min="1" placeholder="Để trống để tự tính" />
        </div>
      </div>

      <button id="submit" class="btn">⏱️ Chia nhóm & Tải kết quả</button>
      <div class="progress" id="prog"><div class="bar" id="bar"></div></div>
      <div id="msg"></div>
    </div>
  </div>

  <script>
    const $ = (s)=>document.querySelector(s);
    const drop = $("#drop"), fileInput = $("#file"), btn = $("#submit");
    const msg = $("#msg"), prog = $("#prog"), bar = $("#bar");
    let file = null;

    function setMsg(type, text){ msg.innerHTML = '<div class="alert '+(type==='ok'?'ok':'err')+'">'+text+'</div>'; }
    function clearMsg(){ msg.innerHTML = ""; }
    function setProgress(v){ prog.style.display = (v>=0 && v<=100) ? "block":"none"; bar.style.width = v+"%"; }

    ["dragenter","dragover"].forEach(ev=>{
      drop.addEventListener(ev, e=>{ e.preventDefault(); e.stopPropagation(); drop.classList.add("dragover"); });
    });
    ["dragleave","drop"].forEach(ev=>{
      drop.addEventListener(ev, e=>{ e.preventDefault(); e.stopPropagation(); drop.classList.remove("dragover"); });
    });
    drop.addEventListener("drop", e=>{
      const dt = e.dataTransfer;
      if (dt && dt.files && dt.files.length){
        file = dt.files[0]; fileInput.files = dt.files;
        clearMsg(); setMsg("ok", "Đã chọn: <b>"+file.name+"</b>");
      }
    });
    fileInput.addEventListener("change", ()=>{
      if (fileInput.files.length){ file = fileInput.files[0]; clearMsg(); setMsg("ok","Đã chọn: <b>"+file.name+"</b>"); }
    });

    btn.addEventListener("click", async ()=>{
      clearMsg();
      if(!file){ setMsg("err","Vui lòng chọn file .xlsx trước."); return; }
      const group_size = parseInt($("#group_size").value || "3", 10);
      const groups_per_ca_raw = ($("#groups_per_ca").value || "").trim();

      const form = new FormData();
      form.append("file", file);
      form.append("group_size", isNaN(group_size)||group_size<2 ? 3 : group_size);
      if (groups_per_ca_raw !== "") form.append("groups_per_ca", groups_per_ca_raw);

      btn.disabled = true; setProgress(15);
      try{
        const res = await fetch("/api/group", { method:"POST", body:form });
        if(!res.ok){
          let err = "Lỗi không xác định.";
          try{ const j = await res.json(); err = j.error || err; }catch(_){}
          setProgress(0); btn.disabled=false; setMsg("err", err); return;
        }
        setProgress(80);
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a"); a.href=url; a.download="chia_nhom.xlsx"; document.body.appendChild(a); a.click(); a.remove();
        URL.revokeObjectURL(url); setProgress(100); setTimeout(()=>setProgress(-1), 800);
        setMsg("ok","✅ Đã tạo & tải <b>chia_nhom.xlsx</b>.");
      }catch(e){
        setProgress(0); setMsg("err","Không thể kết nối server: "+(e?.message||e));
      }finally{ btn.disabled=false; }
    });
  </script>
</body>
</html>
    """




@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/group")
async def api_group(
    file: UploadFile = File(...),
    group_size: int = Form(3),
    groups_per_ca: Optional[int] = Form(None),
):
    # Đọc file Excel người dùng upload
    content = await file.read()
    try:
        df_raw = pd.read_excel(io.BytesIO(content), engine="openpyxl")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Không đọc được Excel: {e}"})

    # Làm sạch dữ liệu
    df = clean_dataframe(df_raw)
    if df.empty:
        return JSONResponse(status_code=400, content={"error": "Dữ liệu hợp lệ rỗng sau khi làm sạch (thiếu 'Ca học' hoặc 'Tên')."})

    # Chuyển thành dict records & chạy chia nhóm
    dssv = df.to_dict("records")
    ket_qua = chia_nhom_tuong_dong(
        dssv,
        so_nhom_mong_muon_moi_ca=groups_per_ca,
        so_luong_moi_nhom=group_size
    )

    # Build kết quả + ghi vào buffer Excel
    rows, thong_ke = [], []
    for ca, cac_nhom in ket_qua.items():
        for idx, nhom in enumerate(cac_nhom, start=1):
            gpa_tb = round(gpa_tb_nhom(nhom), 2)
            thong_ke.append({'Ca': ca, 'Nhóm': idx, 'Số lượng': len(nhom), 'GPA TB Nhóm': gpa_tb})
            for sv in nhom:
                rows.append({
                    'Mã SV': sv.get('Mã SV'),
                    'Tên': sv.get('Tên'),
                    'Lớp': sv.get('Lớp'),
                    'Ca': ca,
                    'GPA': sv.get('GPA'),
                    'Nhóm': idx,
                    'GPA TB Nhóm': gpa_tb
                })

    if not rows:
        return JSONResponse(status_code=400, content={"error": "Không tạo được kết quả chia nhóm."})

    df_out = pd.DataFrame(rows, columns=['Mã SV', 'Tên', 'Lớp', 'Ca', 'GPA', 'Nhóm', 'GPA TB Nhóm'])
    df_summary = pd.DataFrame(thong_ke, columns=['Ca', 'Nhóm', 'Số lượng', 'GPA TB Nhóm'])

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="ChiaNhom", index=False)
        df_summary.to_excel(writer, sheet_name="ThongKe", index=False)
    buf.seek(0)

    filename = "chia_nhom.xlsx"
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

# Cho phép vẫn chạy CLI cũ: `python group_server.py -i gpa.xlsx ...`
if __name__ == "__main__":
    import sys
    # nếu chạy kèm đối số CLI thì giữ nguyên hành vi cũ
    if any(a in sys.argv for a in ["-i", "--input", "-o", "--output", "-g", "--groups-per-ca", "-s", "--group-size"]):
        main()
    else:
        import uvicorn
        uvicorn.run("group_students:app", host="0.0.0.0", port=8000, reload=True)
