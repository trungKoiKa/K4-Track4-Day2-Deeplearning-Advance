# Bộ triển khai Lab Day 2 — DeepWeeds

Code hoàn thiện nằm trong `lab_solution/`; `starter/` và `eval.py` gốc được giữ nguyên.
Đây là pipeline chạy thí nghiệm, không phải bộ kết quả đã huấn luyện.

## Colab

Mở `01_DeepWeeds_Full_Lab_Colab.ipynb` trên Colab. Chạy từ trên xuống theo từng phần;
không chọn Run all lần đầu. Notebook đã gắn với thư mục Drive `Lab_Day2_DeepWeeds`.
Bundle `lab_day2_code.zip` chứa code, eval.py gốc, tests và notebook.

1. Kết nối Drive, giải nén bundle code, cài requirements. Colab phải được bật GPU khi train.
2. Chuẩn bị ảnh, EDA, kiểm tra pipeline; xem ảnh augmentation/CutMix.
3. Preflight chọn batch chung cho cả 5 mạng và ghi tag trọng số.
4. Chạy từng B01–B05, chọn backbone; từng T01–T06, chọn recipe.
5. So sánh I00–I04 trên val, đo latency, khóa cấu hình.
6. Huấn luyện riêng T00/F01 mỗi seed 0,1,2. Hoàn tất cả 6 trước khi test.
7. Test từng cấu hình/seed, xuất kết quả; viết nhận xét ảnh lỗi và đóng gói theo MSSV/họ tên.

Sau ngắt phiên: chạy lại phần kết nối/cài đặt/chuẩn bị dữ liệu, tạo cùng Study, rồi chạy lại
ô thí nghiệm bị ngắt. Các ô EDA/pipeline/preflight đã hoàn tất sẽ được bỏ qua.
Không sửa code/config/CSV giữa một lần chạy; checkpoint kiểm tra hash và cấu hình trước resume.
Checkpoint cuối epoch là điểm tiếp tục; epoch đang chạy khi ngắt có thể được chạy lại từ checkpoint.
Test lưu từng batch, phần đã lưu không suy luận lại. GPU không được đảm bảo trên Colab miễn phí.

## Thiết kế thí nghiệm

- 5 backbone; 6 ablation/kết hợp; 6 lần train chung kết: 17 lần train chính, 10 epoch/lần.
- Batch thực 16, accumulation 4; nếu preflight hết bộ nhớ, toàn bộ nhóm dùng 8/8.
- Khởi tạo/augmentation/loss là 3 trục, mỗi lần chỉ đổi một yếu tố trừ thí nghiệm kết hợp.
- Macro-F1 val chọn cấu hình; test không được dùng để chọn hoặc khớp T.
- Focal gamma0 bằng CE; weight decay không áp dụng norm/bias cả head; frozen backbone giữ BN eval.
- Five-crop từ ảnh 256 trước crop224. TTA gộp logit, temperature khớp đúng pipeline.
- Đo latency thực: warmup10, 100 lần, sync, batch1/32, gồm forward/gộp/softmax; loại đọc ảnh/CPU preprocessing.
- MAC dùng fvcore, ghi toán tử chưa hỗ trợ. Một multiply-add tính một; không coi MAC là latency.
- ConvNeXt khóa tag convnext_tiny.fb_in1k; cả 5 backbone dùng ImageNet-1k, ghi đầy đủ tag và khác biệt recipe pretrained.
- Một nhãn mâu thuẫn có trong nguồn gốc: ảnh 20170714-110407-3.jpg, train0 labels1. Giữ CSV và báo cáo.

## Chạy kiểm tra tại máy

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
python -m unittest discover -s tests -v
python -m unittest discover -s solution_tests -v
```

Code cần torch, torchvision, timm và requirements-colab.txt. Các unit test dùng fixture nhỏ,
không phải kết quả lab. Notebook kiểm tra pipeline dùng ảnh thật trong train.

## Sản phẩm

`study/` trên Drive chứa runs (checkpoint + log), curves, predictions, inference, eda và pipeline.
Sau khi đủ kết quả, exporter tạo 7 sheet Excel và report.md từ dự đoán thực, dùng eval.py gốc.
Report chưa sẵn sàng nộp nếu thiếu `analysis_notes.md`; notebook có ô nhập nhận xét quan sát ảnh.
ZIP nộp gồm code/notebook/versions, Excel, báo cáo, ảnh, dự đoán, eval_out và log nhỏ;
không có dataset hoặc trọng số. MSSV/họ tên nhập khi đóng gói; không tự gửi bài/PR.

Các số bài báo là tham khảo, không phải kết quả chạy của bạn. Không cam kết mốc accuracy95.7%.

## Nghiệm thu bổ sung

Best/last checkpoint có đủ state train/eval, optimizer, scheduler, scaler, EMA, RNG và metadata.
Summary có bảng baseline/chung kết, Final có macro-F1 và top-1 val mean/std.
`study.validate()` liệt kê phần còn thiếu; exporter chỉ đóng gói khi ready_to_submit.
Sau kết quả cuối: khởi động lại phiên Colab, chạy setup và prepare_data, rồi
`study.verify_fresh_session()` để kiểm tra nạp 6 model và compatibility mà không rerun test.
Chạy export lại sau kiểm tra phiên mới và sau khi viết analysis_notes.md.
