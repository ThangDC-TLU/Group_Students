# -*- coding: utf-8 -*-
import argparse
import math
import pandas as pd
from collections import defaultdict
import itertools
import re

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

if __name__ == "__main__":
    main()
