# -*- coding: utf-8 -*-
# grouping.py
import math
import itertools
import re
from typing import List, Dict, Any, Optional
from collections import defaultdict

import pandas as pd
import csv


# ================== Chuẩn hoá cột & helpers ==================
CANON_COLS = ['Mã SV', 'Tên', 'Lớp', 'GPA', 'Ca học', 'Công việc IT', 'Sở thích']

RENAME_MAP = {
    'Mã SV': 'Mã SV', 'Ma SV': 'Mã SV', 'MaSV': 'Mã SV', 'MãSV': 'Mã SV',
    'Tên': 'Tên', 'Ten': 'Tên', 'Họ tên': 'Tên', 'Ho ten': 'Tên', 'Ho_ten': 'Tên',
    'Lớp': 'Lớp', 'Lop': 'Lớp', 'Lớp ': 'Lớp',
    'GPA': 'GPA', 'Điểm GPA': 'GPA',
    'Ca học': 'Ca học', 'Ca': 'Ca học', 'Ca ': 'Ca học', 'Ca học ': 'Ca học',
    'Công việc IT': 'Công việc IT', 'Cong viec IT': 'Công việc IT', 'Cong viec': 'Công việc IT',
    'Sở thích': 'Sở thích', 'So thich': 'Sở thích', 'So_thich': 'Sở thích'
}

def _to_float_gpa(x) -> Optional[float]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if not s:
        return None
    # giữ dấu chấm cho số thập phân; cho phép dùng dấu phẩy
    s = s.replace(',', '.')
    s = re.sub(r'[^0-9\.\-]+', '', s)
    try:
        return float(s)
    except Exception:
        return None

def _clean_text(x) -> Optional[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    return s if s else None

def _split_hobbies(x: str) -> List[str]:
    """Tách sở thích theo , ; |  (không đụng tới dấu chấm để tránh phá số thập phân)."""
    if not isinstance(x, str):
        return []
    tmp = x.replace('|', ',').replace(';', ',')
    parts = [p.strip() for p in tmp.split(',')]
    return [p for p in parts if p]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Chuẩn hoá tên cột
    cols = {c: RENAME_MAP.get(c, c) for c in df.columns}
    df = df.rename(columns=cols)

    # Bổ sung cột thiếu và giữ đúng thứ tự cần thiết
    for c in CANON_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[CANON_COLS].copy()

    # Làm sạch từng cột
    df['Mã SV'] = df['Mã SV'].apply(_clean_text)
    df['Tên'] = df['Tên'].apply(_clean_text)
    df['Lớp'] = df['Lớp'].apply(_clean_text)
    df['Ca học'] = df['Ca học'].apply(_clean_text)
    df['Công việc IT'] = df['Công việc IT'].apply(_clean_text)
    df['Sở thích'] = df['Sở thích'].apply(_clean_text)
    df['GPA'] = df['GPA'].apply(_to_float_gpa)

    # Loại bỏ hàng không hợp lệ (thiếu Ca học hoặc Tên)
    df = df[~df['Ca học'].isna() & (df['Ca học'].str.strip() != '')]
    df = df[~df['Tên'].isna() & (df['Tên'].str.strip() != '')]
    df = df.reset_index(drop=True)
    return df


def read_students_csv(path: str) -> pd.DataFrame:
    """Đọc CSV dữ liệu sinh viên an toàn (giữ chuỗi có dấu phẩy/ngoặc kép)."""
    df_raw = pd.read_csv(
        path,
        dtype=str,
        encoding='utf-8',
        engine='python',
        sep=',',
        quotechar='"'
    )
    return clean_dataframe(df_raw)


# ================== Thuật toán chia nhóm ==================
def tinh_diem_tuong_dong(sv1: Dict[str, Any], sv2: Dict[str, Any]) -> float:
    g1 = sv1.get('GPA')
    g2 = sv2.get('GPA')
    gpa_sv1 = (g1 / 4.0) if g1 is not None else 0.0
    gpa_sv2 = (g2 / 4.0) if g2 is not None else 0.0
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
    return w_gpa * diem_gpa + w_cong_viec * cong_viec_chung + w_so_thich * so_thich_chung


def gpa_tb_nhom(nhom: List[Dict[str, Any]]) -> float:
    gpas = [sv.get('GPA') for sv in nhom if sv.get('GPA') is not None]
    return (sum(gpas) / len(gpas)) if gpas else 0.0


def tinh_diem_trung_binh_nhom(nhom: List[Dict[str, Any]]) -> float:
    if not nhom or len(nhom) < 2:
        return 0.0
    tong = 0.0
    so_cap = 0
    for sv1, sv2 in itertools.combinations(nhom, 2):
        tong += tinh_diem_tuong_dong(sv1, sv2)
        so_cap += 1
    return (tong / so_cap) if so_cap > 0 else 0.0


def chia_nhom_tuong_dong(
    dssv: List[Dict[str, Any]],
    so_nhom_mong_muon_moi_ca: Optional[int] = None,
    so_luong_moi_nhom: Optional[int] = None
) -> Dict[str, List[List[Dict[str, Any]]]]:
    """Chia nhóm theo 'Ca học' với tối ưu tương đồng + cân bằng GPA."""
    sinh_vien_theo_ca = defaultdict(list)
    for sv in dssv:
        sinh_vien_theo_ca[sv['Ca học']].append(sv)

    ket_qua: Dict[str, List[List[Dict[str, Any]]]] = {}

    for ca, lst in sinh_vien_theo_ca.items():
        if not lst:
            continue

        # Sắp xếp giảm dần theo GPA
        lst.sort(key=lambda x: x['GPA'] if x['GPA'] is not None else 0.0, reverse=True)

        # Số nhóm
        if so_luong_moi_nhom and so_luong_moi_nhom > 0:
            num_groups = max(1, math.ceil(len(lst) / so_luong_moi_nhom))
        else:
            base = so_nhom_mong_muon_moi_ca if (so_nhom_mong_muon_moi_ca and so_nhom_mong_muon_moi_ca > 0) else 3
            num_groups = min(len(lst), base) if len(lst) >= base else max(1, len(lst) // 2 or 1)

        groups: List[List[Dict[str, Any]]] = [[] for _ in range(num_groups)]

        # Phân bổ snake
        for i, sv in enumerate(lst):
            idx = i % num_groups
            if (i // num_groups) % 2 == 1:
                idx = num_groups - 1 - idx
            groups[idx].append(sv)

        # Tối ưu tương đồng nội nhóm
        for _ in range(60):
            changed = False
            for i1, i2 in itertools.combinations(range(num_groups), 2):
                n1, n2 = groups[i1], groups[i2]
                for a, sv1 in enumerate(n1):
                    for b, sv2 in enumerate(n2):
                        before = tinh_diem_trung_binh_nhom(n1) + tinh_diem_trung_binh_nhom(n2)
                        n1m = n1[:a] + n1[a+1:] + [sv2]
                        n2m = n2[:b] + n2[b+1:] + [sv1]
                        after = tinh_diem_trung_binh_nhom(n1m) + tinh_diem_trung_binh_nhom(n2m)
                        if after > before:
                            d_tr = gpa_tb_nhom(n1) - gpa_tb_nhom(n2)
                            d_sa = gpa_tb_nhom(n1m) - gpa_tb_nhom(n2m)
                            if abs(d_sa - d_tr) < 0.25:
                                groups[i1], groups[i2] = n1m, n2m
                                changed = True
                                break
                    if changed:
                        break
                if changed:
                    break
            if not changed:
                break

        # Cân bằng GPA TB giữa các nhóm
        gpas_ca = [sv.get('GPA') for sv in lst if sv.get('GPA') is not None]
        gpa_toan_ca = (sum(gpas_ca) / len(gpas_ca)) if gpas_ca else 0.0
        eps = 1e-9

        for _ in range(120):
            improved = False
            for i1, i2 in itertools.combinations(range(num_groups), 2):
                n1, n2 = groups[i1], groups[i2]
                best_delta = 0.0
                best_pair = None
                var_before = ((gpa_tb_nhom(n1) - gpa_toan_ca) ** 2 + (gpa_tb_nhom(n2) - gpa_toan_ca) ** 2)
                for a, sv1 in enumerate(n1):
                    for b, sv2 in enumerate(n2):
                        n1m = n1[:a] + n1[a+1:] + [sv2]
                        n2m = n2[:b] + n2[b+1:] + [sv1]
                        var_after = ((gpa_tb_nhom(n1m) - gpa_toan_ca) ** 2 + (gpa_tb_nhom(n2m) - gpa_toan_ca) ** 2)
                        delta = var_before - var_after
                        if delta > best_delta + eps:
                            best_delta = delta
                            best_pair = (a, b)
                if best_pair is not None:
                    a, b = best_pair
                    n1[a], n2[b] = n2[b], n1[a]
                    improved = True
            if not improved:
                break

        ket_qua[ca] = groups

    return ket_qua


# ================== API tiện ích: chạy/ghi/đọc CSV kết quả ==================
def run_grouping_from_csv(csv_path: str, group_size: Optional[int], groups_per_ca: Optional[int]) -> Dict[str, Any]:
    """
    Đọc gpa.csv -> chia nhóm -> trả về {'ok', 'rows', 'summary'}.
    rows: đúng 7 cột ['Mã SV','Tên','Lớp','Ca','GPA','Nhóm','GPA TB Nhóm']
    """
    df = read_students_csv(csv_path)
    if df.empty:
        return {'ok': False, 'message': 'Dữ liệu sau khi làm sạch rỗng hoặc chưa có.'}

    dssv = df.to_dict('records')
    result = chia_nhom_tuong_dong(
        dssv,
        so_nhom_mong_muon_moi_ca=groups_per_ca,
        so_luong_moi_nhom=group_size
    )

    rows: List[Dict[str, Any]] = []
    stats: List[Dict[str, Any]] = []

    for ca, groups in result.items():
        for idx, nhom in enumerate(groups, start=1):
            gpa_tb = round(gpa_tb_nhom(nhom), 2)
            stats.append({'Ca': ca, 'Nhóm': idx, 'Số lượng': len(nhom), 'GPA TB Nhóm': gpa_tb})
            for sv in nhom:
                rows.append({
                    'Mã SV': sv.get('Mã SV'),
                    'Tên': sv.get('Tên'),
                    'Lớp': sv.get('Lớp'),
                    'Ca': ca,  # Lưu đúng Ca học (vd: Ca1, Ca2)
                    'GPA': (float(sv['GPA']) if sv.get('GPA') is not None else None),
                    'Nhóm': idx,
                    'GPA TB Nhóm': gpa_tb
                })

    if not rows:
        return {'ok': False, 'message': 'Không tạo được kết quả chia nhóm.'}

    return {'ok': True, 'rows': rows, 'summary': stats}


def write_chianhom_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    """
    Ghi kết quả ra CSV đúng 7 cột & đúng thứ tự:
    ['Mã SV','Tên','Lớp','Ca','GPA','Nhóm','GPA TB Nhóm']
    """
    headers = ['Mã SV', 'Tên', 'Lớp', 'Ca', 'GPA', 'Nhóm', 'GPA TB Nhóm']
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=headers,
            quoting=csv.QUOTE_MINIMAL,
            quotechar='"',
            delimiter=','
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ('' if r.get(k) is None else r.get(k)) for k in headers})


def read_chianhom_csv(out_path: str) -> List[Dict[str, Any]]:
    """Đọc lại `chianhom.csv` -> list[dict] đúng 7 cột; ép kiểu số cho GPA / Nhóm / GPA TB Nhóm."""
    try:
        df = pd.read_csv(out_path, dtype=str, encoding='utf-8', engine='python', sep=',', quotechar='"')
    except FileNotFoundError:
        return []

    rows = df.to_dict('records')
    for r in rows:
        r['Nhóm'] = int(float(r['Nhóm'])) if r.get('Nhóm') not in (None, '', 'NaN') else None
        r['GPA'] = float(r['GPA']) if r.get('GPA') not in (None, '', 'NaN') else None
        r['GPA TB Nhóm'] = float(r['GPA TB Nhóm']) if r.get('GPA TB Nhóm') not in (None, '', 'NaN') else None
    return rows
