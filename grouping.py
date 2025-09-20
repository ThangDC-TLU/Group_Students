# -*- coding: utf-8 -*-
import math, itertools, re
import pandas as pd
from typing import List, Dict, Any

# Chuẩn hoá cột
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
    if isinstance(x, (int,float)): return float(x)
    s = str(x).strip().replace(',', '.')
    s = re.sub(r'[^0-9\.\-]+','', s)
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
    return df.reset_index(drop=True)

# ---- Thuật toán tính điểm & chia nhóm ----
def tinh_diem_tuong_dong(sv1, sv2):
    g1, g2 = sv1.get('GPA'), sv2.get('GPA')
    gpa_sv1 = (g1/4.0) if g1 is not None else 0.0
    gpa_sv2 = (g2/4.0) if g2 is not None else 0.0
    diem_gpa = 1.0 - abs(gpa_sv1 - gpa_sv2)

    cong_viec_chung = 1 if sv1.get('Công việc IT') and sv1['Công việc IT'] == sv2.get('Công việc IT') else 0

    so_thich_chung = 0
    if sv1.get('Sở thích') and sv2.get('Sở thích'):
        so1 = set(map(str.lower, _split_hobbies(sv1['Sở thích'])))
        so2 = set(map(str.lower, _split_hobbies(sv2['Sở thích'])))
        so_thich_chung = len(so1 & so2)

    return 0.4*diem_gpa + 0.3*cong_viec_chung + 0.3*so_thich_chung

def gpa_tb_nhom(nhom):
    gpas = [sv.get('GPA') for sv in nhom if sv.get('GPA') is not None]
    return (sum(gpas)/len(gpas)) if gpas else 0.0

def tinh_diem_trung_binh_nhom(nhom):
    if not nhom or len(nhom)<2: return 0.0
    tong=0.0; so_cap=0
    for sv1,sv2 in itertools.combinations(nhom,2):
        tong += tinh_diem_tuong_dong(sv1, sv2); so_cap+=1
    return tong/so_cap if so_cap>0 else 0.0

def chia_nhom_tuong_dong(dssv, so_nhom_mong_muon_moi_ca=None, so_luong_moi_nhom=None):
    from collections import defaultdict
    sinh_vien_theo_ca = defaultdict(list)
    for sv in dssv:
        sinh_vien_theo_ca[sv['Ca học']].append(sv)

    ket_qua = {}
    for ca,lst in sinh_vien_theo_ca.items():
        if not lst: continue
        lst.sort(key=lambda x: x['GPA'] if x['GPA'] is not None else 0.0, reverse=True)

        if so_luong_moi_nhom and so_luong_moi_nhom>0:
            num_groups = max(1, math.ceil(len(lst)/so_luong_moi_nhom))
        else:
            base = so_nhom_mong_muon_moi_ca if so_nhom_mong_muon_moi_ca and so_nhom_mong_muon_moi_ca>0 else 3
            num_groups = min(len(lst), base) if len(lst)>=base else max(1, len(lst)//2 or 1)

        groups=[[] for _ in range(num_groups)]
        # snake distribution
        for i,sv in enumerate(lst):
            idx=i%num_groups
            if (i//num_groups)%2==1: idx=num_groups-1-idx
            groups[idx].append(sv)

        # tối ưu tương đồng
        for _ in range(60):
            changed=False
            for i1,i2 in itertools.combinations(range(num_groups),2):
                n1,n2=groups[i1],groups[i2]
                for a,sv1 in enumerate(n1):
                    for b,sv2 in enumerate(n2):
                        before=tinh_diem_trung_binh_nhom(n1)+tinh_diem_trung_binh_nhom(n2)
                        n1m=n1[:a]+n1[a+1:]+[sv2]; n2m=n2[:b]+n2[b+1:]+[sv1]
                        after=tinh_diem_trung_binh_nhom(n1m)+tinh_diem_trung_binh_nhom(n2m)
                        if after>before:
                            if abs(gpa_tb_nhom(n1m)-gpa_tb_nhom(n2m))<0.25:
                                groups[i1],groups[i2]=n1m,n2m
                                changed=True; break
                    if changed: break
                if changed: break
            if not changed: break

        ket_qua[ca]=groups
    return ket_qua
