# -*- coding: utf-8 -*-
import argparse
import random
import pandas as pd

MA_LOP = ['64HTTT1', '64HTTT2', '64HTTT3', '64HTTT4']
CA_HOC = ['Ca1', 'Ca2']
CONG_VIEC = [
    'Full-stack Developer', 'Data Analyst', 'UX/UI Designer',
    'DevOps Engineer', 'Cybersecurity Analyst'
]
SO_THICH = ['Cầu lông', 'Bóng đá', 'Đọc sách', 'Âm nhạc', 'Du lịch', 'Chơi game']


def tao_du_lieu_sinh_vien_fake(so_luong, seed=None):
    """Tạo danh sách sinh viên giả lập (tương thích Python 3.7)."""
    if seed is not None:
        random.seed(seed)

    danh_sach_sinh_vien = []
    for i in range(so_luong):
        ma_sv = "225116{}".format(random.randint(1000, 9999))
        ten = "Sinh Vien {}".format(i + 1)
        lop = random.choice(MA_LOP)
        gpa = round(random.uniform(2.5, 4.0), 2)
        ca = random.choice(CA_HOC)
        it_job = random.choice(CONG_VIEC)
        hobbies = random.sample(SO_THICH, random.randint(1, 3))

        danh_sach_sinh_vien.append({
            'Mã SV': ma_sv,
            'Tên': ten,
            'Lớp': lop,
            'GPA': gpa,
            'Ca học': ca,
            'Công việc IT': it_job,
            'Sở thích': ', '.join(hobbies),
        })
    return danh_sach_sinh_vien


def main():
    parser = argparse.ArgumentParser(
        description="Sinh dữ liệu sinh viên và xuất ra Excel (gpa.xlsx)."
    )
    parser.add_argument("-n", "--num", type=int, default=50,
                        help="Số lượng sinh viên (mặc định: 50)")
    parser.add_argument("-o", "--output", type=str, default="gpa.xlsx",
                        help="Tên/đường dẫn file Excel xuất ra (mặc định: gpa.xlsx)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed để cố định kết quả random (tùy chọn)")

    args = parser.parse_args()

    ds = tao_du_lieu_sinh_vien_fake(args.num, seed=args.seed)
    df = pd.DataFrame(ds, columns=[
        'Mã SV', 'Tên', 'Lớp', 'GPA', 'Ca học', 'Công việc IT', 'Sở thích'
    ])

    # Dùng openpyxl 3.0.x cho Python 3.7
    df.to_excel(args.output, index=False, engine="openpyxl", sheet_name="Students")
    print("✅ Đã tạo {} với {} dòng.".format(args.output, len(df)))


if __name__ == "__main__":
    main()
